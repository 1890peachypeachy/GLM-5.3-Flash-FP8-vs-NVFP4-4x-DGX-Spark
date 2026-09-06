#!/usr/bin/env python3
"""
bench_sustained_decode — sustained concurrent-decode benchmark.

Holds N agents in CONTINUOUS sustained decode simultaneously (long prose/code
outputs, no idle gaps between turns), measuring per-agent and aggregate accepted
tok/s. This is our real workload: 3-4 tool-calling agents generating continuously.

Usage:
  python3 bench_sustained_decode.py --endpoint http://HOST:8001/v1 --model glm-5.3-flash \
      --agents 4 --max-tokens 2000 --concurrency 4 --label fp8-c4 --out fp8-sustained.json
"""
import argparse, datetime, json, statistics, time, urllib.request, threading

PROMPTS = {
    "prose": ("Write a long, detailed technical essay explaining how large-scale "
              "distributed inference works across tensor-parallel GPU clusters: prefill, "
              "decode, KV cache, prefix caching, speculative decoding, and MoE routing. "
              "Be very thorough and do not stop until you have covered every topic."),
    "code": ("Write a complete, well-commented Python library implementing a concurrent "
             "task scheduler with a thread pool, priority queue, dependency resolution, "
             "retry logic, and progress reporting. Include full docstrings and unit "
             "tests. Write as much as you can."),
    "structured": ("Generate a detailed JSON configuration file describing a 4-node "
                   "inference cluster: node names, IPs, roles, model serving params, "
                   "and health checks. Output only valid JSON, thorough."),
}

def thinking_kwargs(mode):
    if mode == "off":
        return {"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}}
    return {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low", "clear_thinking": True}}

def stream_once(base, model, prompt_cat, max_tokens, mode, timeout=1800):
    prompt = PROMPTS[prompt_cat]
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
               "stream_options": {"include_usage": True}, **thinking_kwargs(mode)}
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; last_content = None; usage = {}
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = b""
            for raw in r:
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.decode(errors="replace").strip()
                    if not line.startswith("data:"): continue
                    data = line[5:].strip()
                    if data == "[DONE]": continue
                    try: ev = json.loads(data)
                    except Exception: continue
                    ts = time.perf_counter() - t0
                    d = (ev.get("choices") or [{}])[0].get("delta") or {}
                    if (d.get("content") or d.get("reasoning_content")) and first is None:
                        first = ts
                    if d.get("content"): last_content = ts
                    if ev.get("usage"): usage = ev["usage"]
    except Exception as e:
        err = repr(e)
    total = time.perf_counter() - t0
    comp = int((usage or {}).get("completion_tokens") or 0)
    decode_s = (last_content - first) if (first and last_content) else None
    return {"wall_s": round(total, 3), "ttft_s": round(first, 3) if first else None,
            "completion_tokens": comp,
            "decode_s": round(decode_s, 3) if decode_s else None,
            "tok_s": round((comp - 1) / decode_s, 2) if (decode_s and comp > 1) else None,
            "err": err}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--thinking", default="low", choices=["low", "off"])
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cats = ["prose", "code", "structured"]
    out = {"ts": datetime.datetime.now().isoformat(), "endpoint": a.endpoint,
           "model": a.model, "agents": a.agents, "max_tokens": a.max_tokens,
           "thinking": a.thinking, "label": a.label, "runs": []}

    # warmup (sequential, short) so graphs are live before sustained concurrency
    print("warmup (3 short sequential)...", flush=True)
    for c in cats:
        stream_once(a.endpoint, a.model, c, 64, a.thinking)

    # Launch N concurrent sustained decode streams (mix of categories)
    results = [None] * a.agents
    def worker(i):
        cat = cats[i % len(cats)]
        results[i] = stream_once(a.endpoint, a.model, cat, a.max_tokens, a.thinking)
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(a.agents)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and r.get("tok_s")]
    total_tok = sum(r["completion_tokens"] for r in ok)
    per_agent = [r["tok_s"] for r in ok]
    agg = sum(r["completion_tokens"] for r in ok) / wall
    out.update({
        "wall_s": round(wall, 2), "total_completion_tokens": total_tok,
        "aggregate_tok_s": round(agg, 1),
        "per_agent_tok_s": per_agent,
        "per_agent_median_tok_s": round(statistics.median(per_agent), 1) if per_agent else None,
        "per_agent_min_tok_s": round(min(per_agent), 1) if per_agent else None,
        "runs": results})
    path = a.out or f"sustained_{a.agents}ag_{a.max_tokens}t_{a.label or a.thinking}.json"
    with open(path, "w") as f: f.write(json.dumps(out, indent=2))
    print(f"SUSTAINED {a.agents}-agent: aggregate={agg:.1f} tok/s over {wall:.0f}s | "
          f"per-agent={[round(x,1) for x in per_agent]} median={out['per_agent_median_tok_s']}", flush=True)
    print("SAVED", path, flush=True)

if __name__ == "__main__":
    main()
