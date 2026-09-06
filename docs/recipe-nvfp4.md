# The NVFP4 recipe (4x DGX Spark, TP4)

This is the quantization that ran as our production GLM-5.3-Flash lane from
2026-08-28 until FP8 replaced it on 2026-09-06. It is **superseded for
production** (see [switch-adaptation.md](switch-adaptation.md) for why FP8 won)
but fully documented here so you can serve either — and it remains our rollback
lane: launchers + weights are kept staged on all four nodes.

The NVFP4 recipe descends from
[tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
(forked and adapted for our fleet). **That repo is the deep engineering record** —
SM121 crash forensics, the TopK oversubscription patch, DFlash2 bring-up, the
KV-memory ladder, and every hard-won boot rule. This page carries only what you
need to launch it.

## Weights and files to download

| Artifact | Source | Size | Notes |
| --- | --- | --- | --- |
| Main weights | [`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | ~182 GiB, 11 shards | compressed-tensors build. **Do not use ModelOpt/LibertAIDAI builds** — they emit corrupted token IDs ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)); we reproduced it and the RedHat build is clean (U+FFFD probe: 0/0/0 vs 4/9/8). |
| DFlash2 drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ rev `bf582e4e` | ~2.3 GB | Pin the revision; verify `model.safetensors` sha256 after download. CC BY-NC-ND 4.0 (non-commercial). |
| Container image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v12-dflash2-topkfix` | ~31 GB | Pull once, fan out (`docker save \| ssh \| docker load`), verify the image **ID** matches on all 4 nodes. |
| SM121 indexer patch | [`sparse_attn_indexer_kpool_sm121.py`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) | — | Without it the engine boots, answers short prompts, then dies on every decode past ~24K context. |
| Vision chat template | `chat_template_mm.jinja` (in the tonyd2wild repo) | — | Must sit **inside the weights dir** or image requests 500. |

## Launch (our fleet values)

Same topology as the FP8 recipe: 4x GB10, TP4, one switched RoCE fabric
(`10.73.0.0/24`, single HCA `rocep1s0f1` per node), worker-first boot
(rank 3 → 2 → 1 → 0). Our optimized launch configuration:

| Setting | Value |
| --- | --- |
| Context window | 491,520 (480K) — was 1M before the 2026-09-05 right-size |
| `--max-num-seqs` | 12 — was 64 |
| KV cache | fp8 e4m3, pinned 24 GiB/rank (`--kv-cache-memory`), pool ≈ 3.8M tokens |
| MoE backend | **`--moe-backend marlin` — REQUIRED, never `auto`** (native FlashInfer CUTLASS FP4 path silently emits garbage on GB10) |
| Spec decode | DFlash2 k=7 (k=3 in the final A/B config) |
| CUDA graphs | FULL_AND_PIECEWISE (eager off) |
| Chunked prefill | on, threshold 2048 |
| Flusher | **unconditional page-cache flusher on every node, started before the launcher, for the whole boot** |
| Served identity | `glm-5.3-flash` on `:8001` |

Launch order and the memory ritual (swap must exist but swappiness=0, drop
caches, flusher running) are documented step-by-step in the
[tonyd2wild repo README](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark).
Its hard-won rules all apply: tear down all ranks before relaunching any, verify
image IDs not tags, gate with a long prompt AND a long answer, and treat
"serving is not the bar" — a config that boots is not a config that works.

## Limitations (measured on our fleet)

- **Sustained-decode collapse.** At 3 agents in continuous decode, per-agent
  throughput fell to ~0.5-2 tok/s (aggregate 1.4-5.5), with DFlash2 draft
  throughput cratering to 3-8 tok/s. Single-agent clean decode is fine
  (~45 tok/s code, ~27 prose/structured) — the collapse only appears under
  sustained concurrent load, which is exactly the multi-agent workload.
- **Cannot serve 4 growing agents.** In the C4 probe (4 agents growing to 416K)
  it wedged at 192K with 79-94 s/turn turns and had to be killed. 3 agents
  growing to 416K complete fine (~7-11 s/turn warm).
- **KV-pool edge near 400K×3.** Pool ≈ 3.8M tokens: three agents at ~400K
  context approach the edge and per-turn cost rises 10-13 s (pool/window
  fragmentation, not a decode ceiling). KV pool usage stays ~6-8% on the L1
  ladder — the 4-agent wall is engine scheduling contention, not KV capacity.
- **JIT warmup is mandatory.** Growing-context decode stalls past ~288K
  (22-31 s/turn) were 100% Triton/TileLang kernels JIT-compiling at
  first-encounter shapes during inference — not config, not OOM. Run
  [`glm_warmup.py`](../benchmarks/glm_warmup.py) after every boot; a warmed lane
  runs 2.9-5.8 s/turn at the same contexts. Never judge a cold first run.
- **W4A4 quality cost.** The RedHat conversion quantizes activations to 4-bit;
  expect a few points lower on hard reasoning vs the native FP8 checkpoint.
- **DFlash2 drafter is CC BY-NC-ND 4.0** — non-commercial use only, same as FP8.

## Why it lost the A/B (and when you might still want it)

FP8 beat it at every concurrency level and was the only config to complete C4 —
see [switch-adaptation.md](switch-adaptation.md) for the full table. The NVFP4
lane's remaining edge is **KV density** (368 B/token/layer vs fp8's 512 with the
b12x NVFP4-KV route): if you need maximum pool capacity per GiB — e.g. many
concurrent near-1M-context sessions — the NVFP4-KV flex lane documented in the
tonyd2wild repo is still the tool for that job. For 3-4 agents at 400K+, FP8 wins.
