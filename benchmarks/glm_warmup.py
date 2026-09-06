#!/usr/bin/env python3
"""
glm_warmup — post-boot warmup sweep: compile the kernels that otherwise JIT
during inference.

Root cause (2026-09-05): growing-context decode stalls past ~288K are Triton/TileLang
kernels JIT-compiling at first-encounter shapes DURING live inference
(BuildPrefillChunkMetadataKernel, _resample_kernel, _rejection_kernel,
_prepare_dflash_inputs_kernel, mhc_pre_big_fuse_with_norm_tilelang, _compute_local_logits_stats_kernel).
vLLM's jit_monitor: "Triton kernel JIT compilation during inference ... consider extending warmup."

Fix: after boot, drive the engine through the real workload shapes so those kernels
compile BEFORE serving traffic. Sweep growing contexts with a chunked-prefill pattern
+ spec-decode (thinking-low so reasoning path doesn't mask DFlash2) to force each
kernel family to compile once.

Usage: python3 glm_warmup.py --endpoint http://HOST:8001/v1 --model glm-5.3-flash
"""
import argparse, time, urllib.request, json

def warmup_req(base, model, content_len, max_tokens=8, timeout=600):
    """Send a request whose prompt is ~content_len chars (~content_len/4 tokens) so the
    chunked-prefill + spec-decode kernels for that shape JIT. max_tokens small = warmup cost only."""
    # build ~content_len char prompt (repetitive filler + a real question at end)
    filler = ("Distributed LLM inference over RoCE fabrics, prefix caching, KV transfer, "
              "speculative decoding, MoE routing, memory bandwidth. ") * (content_len//120 + 1)
    prompt = filler[:content_len] + "\nQuestion: reply OK."
    payload = {"model": model, "messages": [{"role":"user","content":prompt}],
               "max_tokens": max_tokens, "temperature": 0.0,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}}
    req = urllib.request.Request(base+"/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); got=False; err=None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf=b""
            for raw in r:
                buf+=raw
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    if line.startswith(b"data:"):
                        d=line[5:].strip()
                        if d==b"[DONE]": got=True
    except Exception as e: err=repr(e)
    return round(time.perf_counter()-t0,1), got, err

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-ctx", type=int, default=400000)
    ap.add_argument("--step", type=int, default=50000, help="context chars... use tokens")
    ap.add_argument("--wait", type=float, default=1.0)
    a=ap.parse_args()
    # Sweep token lengths that trigger the chunked-prefill + spec-decode kernels.
    # The stall fired past ~288K, so emphasize the high end. Each ~30-40K.
    targets_tokens = [5000, 20000, 60000, 120000, 180000, 240000, 288000, 320000, 360000, a.max_ctx]
    print(f"=== GLM warmup sweep, {a.endpoint} model={a.model} max_ctx={a.max_ctx} ===", flush=True)
    for tk in targets_tokens:
        # ~4 chars/token
        t0=time.perf_counter()
        wall, got, err = warmup_req(a.endpoint, a.model, tk*4)
        status = "OK" if got else ("ERR:"+err if err else "NO-DONE")
        print(f"  {tk} tok: {wall}s {status}", flush=True)
        time.sleep(a.wait)
    print("=== warmup done ===")

if __name__=="__main__":
    main()
