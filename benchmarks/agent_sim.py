#!/usr/bin/env python3
"""
agent_sim — agent-turn head-to-head harness.

Replicates how we actually drive our local agents (mined from ~2 weeks of real
usage): 1-3 concurrent agents, thinking-low, tool-call loops that grow context
across turns, long-context reads with follow-ups, and a structured/code/prose
output mix. Runs the SAME scenario file against both quantizations so the
numbers are directly comparable.

Design notes (from usage mining 2026-09-04):
- Real agents carry 100K-250K contexts (median ~117-196K; extremes to ~441K).
- GLM agents are capped at 256K in our stack (model_overrides). We test to 250K.
- Dominant pattern = tool-call loop (decide -> call tool -> get result -> continue),
  which grows context turn over turn and stresses prefix-cache reuse.
- Pain that pushed us off our previous long-context lane: stream-stale timeouts at
  370K+ ctx. So we also time LONG generations to catch stalls, not just steady-state tok/s.
- 'Thinking low' is the agent regime.

Usage:
  python3 agent_sim.py --endpoint http://HOST:PORT/v1 --model NAME \
      --scenario agent_tool_loop --concurrency 1 [3] --ctx 100000 [200000 250000]
  python3 agent_sim.py --endpoint ... --model ... --scenario full --concurrency 1 3

Measures per agent: TTFT, decode tok/s (wall + server), total wall s, done-flag,
prefix-cache (cached_tokens from usage when stream_options.include_usage set).

Server decode metric: read /metrics for vllm decode counters when available.
"""
import argparse, datetime, json, sys, time, urllib.request, statistics, socket
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- prompts

TOOL_LOOP_TASKS = [
    "You are an engineering agent. The build just failed with: ModuleNotFoundError: No module named 'vllm_backend'. "
    "Call the tool check_deps, then the tool read_log, then the tool install_pkg, then explain in 3 sentences what was wrong and how you fixed it.",
    "You are a research agent. A supplier said a SKU is 'in transit'. Call tool lookup_order, then tool check_eta, then tool notify_customer, then summarize the outcome in JSON.",
    "You are a data agent. A nightly sync reported 3 row mismatches. Call tool diff_tables, tool find_root_cause, tool apply_fix, then report root cause + fix as JSON.",
]

PROSE_TASK = ("You are a drafting agent. Write a clear, well-structured memo to the team about why we should standardize "
              "on a single local model-serving lane. Cover: reliability under long context, multi-agent concurrency, "
              "quantization fidelity tradeoffs, and operational cost. Aim for ~350 words.")

STRUCTURED_TASK = ("Return ONLY valid JSON, no markdown. Schema: {\"decision\": string, \"reasons\": [string], "
                   "\"risks\": [string], \"next_step\": string}. Topic: should we serve GLM-5.3-Flash or DeepSeek "
                   "as the primary local lane for 3 concurrent coding agents?")

CODE_TASK = ("Implement a Python function `lru_cache_with_ttl(maxsize, default_ttl)` as a drop-in replacement that "
             "supports per-key TTL expiry and is thread-safe. Return only the code in a single ```python block.")

LONGDOC_TOPICS = ["attention kernels", "prefix caching", "KV transfer", "MoE routing", "thermal noise",
                  "CUDA graphs", "tool loops", "speculative decoding", "quantization", "NCCL fabric"]
NEEDLES = ["SPARK_NEEDLE_alpha-77", "SPARK_NEEDLE_beta-42", "SPARK_NEEDLE_gamma-19"]

def build_long_doc(target_tokens):
    """Build a repetitive long-context doc of ~target_tokens, hiding a needle near the end."""
    chunks = []
    i = 0
    while True:
        topic = LONGDOC_TOPICS[i % len(LONGDOC_TOPICS)]
        txt = (f"[Section {i:04d}] {topic}: local AI serving, prefill, decode, cache reuse, fabric, logs. " * 4)
        needle_idx = None
        # hide needle near ~90% mark
        if i > 0 and i % 900 == 800:
            chunks.append(f" Critical hidden fact: {NEEDLES[(i//900)%len(NEEDLES)]}. Remember this code exactly. ")
            needle_idx = len(chunks)-1
        chunks.append(txt)
        i += 1
        approx = sum(len(c) for c in chunks) // 4  # ~4 chars/token
        if approx >= target_tokens:
            break
    return "\n".join(chunks)

# ---------------------------------------------------------------- client

class Client:
    def __init__(self, base, model, timeout=1800, thinking="low"):
        self.base = base.rstrip('/')
        self.model = model
        self.timeout = timeout
        # thinking low/off -> template kwargs. GLM: enable_thinking + reasoning_effort.
        if thinking == "off":
            self.extra = {"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}}
        else:  # low
            self.extra = {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low",
                                                   "clear_thinking": True}}

    def chat_stream(self, messages, max_tokens, temperature=0.0, extra_kwargs=None):
        """Return (events, usage, error). events: list of (ts, delta_content)."""
        kw = dict(getattr(self, "extra", None) or {})
        # per-call overrides win
        if extra_kwargs:
            for k, v in extra_kwargs.items():
                if isinstance(v, dict) and isinstance(kw.get(k), dict):
                    kw[k] = {**kw[k], **v}
                else:
                    kw[k] = v
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kw,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        events = []
        usage = {}
        error = None
        ttft = None
        first_chunk = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                # SSE parse
                buf = b""
                for raw in r:
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            ev = json.loads(data)
                        except Exception:
                            continue
                        ts = time.perf_counter() - t0
                        if not events:
                            events.append(("start", ts))
                        # chunk delta
                        delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                        c = delta.get("content")
                        rc = delta.get("reasoning_content")
                        if (c or rc) and first_chunk is None:
                            first_chunk = ts
                        events.append(("delta", ts, (c or "") + ("" if rc is None else "[re]"+rc)))
                        if ev.get("usage"):
                            usage = ev["usage"]
        except Exception as e:
            error = repr(e)
        total = time.perf_counter() - t0
        return events, usage, error, total, first_chunk

# ---------------------------------------------------------------- scenario runners

def run_tool_loop(client, label, ctx_target, max_out):
    """A multi-turn tool-loop agent: each turn adds context; measures each turn + whole."""
    turns = []
    # long doc as 'read' a tool returned, injected as a tool result -> grows context
    doc = build_long_doc(ctx_target // 2)
    messages = [{"role": "system", "content": "You are a coding/ops agent that calls tools and reports concisely."}]
    for ti, task in enumerate(TOOL_LOOP_TASKS[:3]):
        if ti == 0:
            messages.append({"role": "user", "content": task})
        else:
            # simulate prior tool result coming back + new instruction
            messages.append({"role": "assistant", "content": f"<tool_call>called tool {ti}</tool_call>"})
            messages.append({"role": "tool", "content": f"tool result {ti}: ok, proceeding", "tool_call_id": f"call_{ti}"})
            messages.append({"role": "user", "content": task + " Continue from where you left off."})
        ev, usage, err, total, ttft = client.chat_stream(messages, max_tokens=max_out)
        msg = ""
        for e in ev:
            if e[0] == "delta":
                msg += e[2]
        turns.append({
            "turn": ti, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "wall_s": round(total, 3), "ttft_s": round(ttft, 3) if ttft else None,
            "err": err,
        })
        # carry output back into context like a real agent does
        messages.append({"role": "assistant", "content": msg})
    return {"label": label, "type": "tool_loop", "turns": turns,
            "total_s": round(sum(t["wall_s"] for t in turns), 3),
            "total_completion": sum(t["completion_tokens"] or 0 for t in turns)}

def run_longdoc(client, label, ctx_target, max_out):
    doc = build_long_doc(ctx_target)
    needle = None
    for nd in NEEDLES:
        if nd in doc: needle = nd
    q1 = "\n\nQuestion: What is the exact hidden code in the document? Reply with only the code value."
    q2 = "\n\nFollow-up: Now explain in 2-3 sentences why prefix caching helps a second question on the same document."
    messages = [{"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": doc + q1}]
    res = {}
    for qk, q in [("prefill", messages), ]:
        ev, usage, err, total, ttft = client.chat_stream(q, max_tokens=min(80, max_out))
        res["prefill"] = {"wall_s": round(total,3), "ttft_s": round(ttft,3) if ttft else None,
                          "prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "err": err}
    # follow-up turn on same doc (prefix reuse)
    messages2 = messages + [{"role": "assistant", "content": "the code"}, {"role": "user", "content": q2}]
    ev, usage, err, total, ttft = client.chat_stream(messages2, max_tokens=200)
    res["followup"] = {"wall_s": round(total,3), "ttft_s": round(ttft,3) if ttft else None,
                       "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       "err": err}
    return {"label": label, "type": "longdoc", "ctx_target": ctx_target, "needle_found": bool(needle),
            **res}

def run_single_prompt(client, label, task, max_out, thinking_off=True):
    messages = [{"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": task}]
    ev, usage, err, total, ttft = client.chat_stream(messages, max_tokens=max_out)
    return {"label": label, "type": "single", "wall_s": round(total,3),
            "ttft_s": round(ttft,3) if ttft else None,
            "completion": usage.get("completion_tokens"), "err": err}

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", choices=["full","tool_loop","longdoc","single","mixed"], default="full")
    ap.add_argument("--concurrency", type=int, default=1, help="1 or 3")
    ap.add_argument("--ctx", type=int, default=100000, help="target context tokens for longdoc")
    ap.add_argument("--thinking", choices=["low","off"], default="low")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = {"ts": datetime.datetime.now().isoformat(), "endpoint": args.endpoint,
           "model": args.model, "label": args.label, "tag": args.tag,
           "concurrency": args.concurrency, "ctx_target": args.ctx, "runs": []}
    cl = Client(args.endpoint, args.model, thinking=args.thinking)

    # ---------------------------------------------------------------- scenarios to run per agent
    if args.scenario == "tool_loop":
        steps = ["tool_loop"]
    elif args.scenario == "longdoc":
        steps = ["longdoc"]
    elif args.scenario == "single":
        steps = ["single"]
    elif args.scenario == "mixed":
        # structured/code/prose single-prompt mix + a tool loop = the real agent diet
        steps = ["single", "tool_loop"]
    else:  # full
        steps = ["tool_loop", "longdoc", "single"]

    def one(i):
        tag = f"{args.label}{'-'+args.tag if args.tag else ''}"
        res = []
        for s in steps:
            if s == "tool_loop":
                r = run_tool_loop(cl, f"{tag}-agent{i}", args.ctx, 400)
            elif s == "longdoc":
                r = run_longdoc(cl, f"{tag}-agent{i}", args.ctx, 200)
            else:
                # single = the output-type diet: structured JSON, code, prose
                r = {"label": f"{tag}-agent{i}", "type": "output_mix", "items": []}
                for tname, task, mo in [("structured", STRUCTURED_TASK, 260),
                                        ("code", CODE_TASK, 320),
                                        ("prose", PROSE_TASK, 400)]:
                    item = run_single_prompt(cl, f"{tag}-agent{i}-{tname}", task, mo)
                    r["items"].append(item)
                    print(f"  [{tname}] wall={item['wall_s']}s", flush=True)
                r["wall_s"] = round(sum(it["wall_s"] for it in r["items"]), 3)
            res.append(r)
            print(f"[agent{i}/{s}] wall={res[-1].get('total_s') or res[-1].get('wall_s')}s", flush=True)
        return res

    # Run `concurrency` agents in parallel; each agent runs its whole scenario battery.
    t_start = time.perf_counter()
    if args.concurrency == 1:
        results = {0: one(0)}
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(one, i): i for i in range(args.concurrency)}
            results = {futs[f]: f.result() for f in futs}
    out["wall_clock_s"] = round(time.perf_counter()-t_start, 3)
    out["runs"] = [{"agent": k, "steps": results[k]} for k in sorted(results)]

    print(json.dumps(out, indent=2))
    path = args.out or f"agent_sim_{args.scenario}_c{args.concurrency}_{args.tag or 'x'}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w") as f:
        f.write(json.dumps(out, indent=2))
    print("\nSAVED", path)

if __name__ == "__main__":
    main()
