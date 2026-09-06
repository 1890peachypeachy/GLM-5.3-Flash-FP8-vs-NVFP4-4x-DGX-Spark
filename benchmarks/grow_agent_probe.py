#!/usr/bin/env python3
"""
grow_agent_probe — growing-context agent-tool-loop probe (the real 3-4 agent pattern).

Models how an agent ACTUALLY grows: starts small, and each tool-loop turn
appends a tool result (a growing injected document) + a new instruction, so the
context climbs from ~10K toward ~400K across turns. Measures per-turn TTFT,
decode, wall, and whether the prefix cache keeps each turn cheap (tail-only
prefill) or the agent re-prefills a growing chunk.

This is the gap neither agent_sim.tool_loop (stays ~500 tok) nor agent_sim.longdoc
(one-shot read, identical across C3 agents) covers: distinct agents, each growing
its OWN large context turn-over-turn.

Read-only. Safe on a live lane: issues chat completions, no restart.

Usage:
  python3 grow_agent_probe.py --endpoint http://HOST:8001/v1 --model glm-5.3-flash \
      --thinking low --agents 1 [3] --max-ctx 400000 --doc-step 40000 --out out.json
"""
import argparse, datetime, json, time, urllib.request, threading

DOC_TEMPLATE = (
    "[Section {i:06d}] topic={topic}: distributed TP4 serving, prefill, decode, "
    "prefix caching, KV, speculative decoding, RoCE fabric, MoE routing, logs. " * 5
)
TOPICS = ["attention","prefix-cache","KV-transfer","MoE-routing","thermal",
          "cuda-graphs","tool-loops","spec-decode","quantization","nccl"]

def build_growth(step_bytes_target):
    """Return a doc sized ~step_bytes_target chars (~1/4 = tokens)."""
    chunks=[]; i=0; approx=0
    while approx < step_bytes_target:
        chunks.append(DOC_TEMPLATE.format(i=i, topic=TOPICS[i%len(TOPICS)]))
        i+=1; approx=sum(len(c) for c in chunks)
    return "\n".join(chunks)

def thinking_kwargs(mode):
    if mode=="off":
        return {"chat_template_kwargs":{"enable_thinking":False,"clear_thinking":True}}
    return {"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low","clear_thinking":True}}

def stream_once(base, model, messages, max_tokens, mode, timeout=1800):
    payload={"model":model,"messages":messages,"max_tokens":max_tokens,
             "temperature":0.0,"stream":True,"stream_options":{"include_usage":True},
             **thinking_kwargs(mode)}
    req=urllib.request.Request(base+"/chat/completions",data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); usage={}; first=None; err=None
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            buf=b""
            for raw in r:
                buf+=raw
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1); line=line.decode(errors="replace").strip()
                    if not line.startswith("data:"): continue
                    data=line[5:].strip()
                    if data=="[DONE]": continue
                    try: ev=json.loads(data)
                    except Exception: continue
                    ts=time.perf_counter()-t0
                    if first is None:
                        d=(ev.get("choices") or [{}])[0].get("delta") or {}
                        if d.get("content") or d.get("reasoning_content"): first=ts
                    if ev.get("usage"): usage=ev["usage"]
    except Exception as e: err=repr(e)
    total=time.perf_counter()-t0
    return {"wall_s":round(total,3),"ttft_s":round(first,3) if first else None,
            "usage":usage,"err":err}

def run_agent(base, model, idx, mode, max_ctx, doc_step):
    """Grow context 0 -> max_ctx by appending a doc+instruction each turn, ~doc_step tokens."""
    # system + short opener
    messages=[{"role":"system","content":"You are a coding/ops agent that calls tools and reports concisely."},
              {"role":"user","content":"Start working. Call check_deps and report."}]
    res=stream_once(base,model,messages,120,mode)
    turns=[{"turn":0,"label":"base",**res,
            "pt":res["usage"].get("prompt_tokens"),
            "cached":(res["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
            "comp":res["usage"].get("completion_tokens")}]
    # grow context
    acc=0; t=1
    step_chars=doc_step*4  # ~4 chars/token
    while acc < max_ctx:
        doc=build_growth(step_chars)
        messages.append({"role":"assistant","content":"<tool_call>read_doc</tool_call>"})
        messages.append({"role":"tool","tool_call_id":"g","content":doc})
        messages.append({"role":"user","content":f"Context now includes a large tool result (~section index {t*20}). "
                          f"Summarize the key facts in 2 sentences and call check_deps again."})
        res=stream_once(base,model,messages,120,mode)
        pt=res["usage"].get("prompt_tokens"); cached=(res["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        comp=res["usage"].get("completion_tokens")
        turns.append({"turn":t,"pt":pt,"cached":cached,"comp":comp,**res})
        acc=pt or acc+step_chars
        print(f"  [agent{idx} turn{t}] pt={pt} cached={cached} ttft={res['ttft_s']} wall={res['wall_s']} err={res['err']}",flush=True)
        t+=1
        # cap turns to avoid runaway
        if t>40: break
    return {"agent":idx,"mode":mode,"turns":turns,"peak_ctx":max(t["pt"] for t in turns if t["pt"])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--endpoint",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--thinking",default="low",choices=["low","off"])
    ap.add_argument("--agents",type=int,default=1)
    ap.add_argument("--max-ctx",type=int,default=400000)
    ap.add_argument("--doc-step",type=int,default=40000,help="tokens added per turn")
    ap.add_argument("--out",default="")
    ap.add_argument("--label",default="")
    a=ap.parse_args()
    out={"ts":datetime.datetime.now().isoformat(),"endpoint":a.endpoint,"model":a.model,
         "thinking":a.thinking,"agents":a.agents,"max_ctx":a.max_ctx,"doc_step":a.doc_step,
         "label":a.label,"runs":[]}
    t0=time.perf_counter()
    def one(i):
        r=run_agent(a.endpoint,a.model,i,a.thinking,a.max_ctx,a.doc_step)
        out["runs"].append(r)
    threads=[]
    for i in range(a.agents):
        th=threading.Thread(target=one,args=(i,)); th.start(); threads.append(th)
    for th in threads: th.join()
    out["wall_clock_s"]=round(time.perf_counter()-t0,3)
    path=a.out or f"grow_agent_{a.agents}ag_{a.max_ctx}ctx_{a.label or a.thinking}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(path,"w") as f: f.write(json.dumps(out,indent=2))
    print("SAVED",path,"wall",out["wall_clock_s"])

if __name__=="__main__":
    main()
