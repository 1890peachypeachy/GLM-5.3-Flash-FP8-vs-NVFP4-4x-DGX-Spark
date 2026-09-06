# GLM-5.3-Flash FP8 vs NVFP4 on four DGX Sparks

One cluster, two quantizations, one head-to-head. This repository serves
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B MoE)
tensor-parallel across four NVIDIA GB10 (DGX Spark) nodes and documents **both recipes
we ran, our measured results, and why FP8 won production**.

| | **FP8** (current production) | **NVFP4** (superseded, rollback lane) |
| --- | --- | --- |
| Checkpoint | zai-org native FP8 QAT | RedHatAI GLM-5.3-Flash-NVFP4 (compressed-tensors) |
| Weights | ~306 GiB, 62 shards | ~182 GiB, 11 shards |
| Recipe | **this repo** — [`docs/production-recipe.md`](docs/production-recipe.md) + [`docs/switch-adaptation.md`](docs/switch-adaptation.md) | [`docs/recipe-nvfp4.md`](docs/recipe-nvfp4.md) + the tonyd2wild repo |
| Speculative decoding | DFlash2 drafter, adaptive-k 3-5 | DFlash2 drafter, k=7 (k=3 in final A/B) |
| MoE kernels | Triton FP8 + GB10 JSON | Marlin (required on GB10) |
| Context / KV | 480K window, fp8 KV pool ≈ 2.42M tokens | 480K window, fp8 KV pool ≈ 3.8M tokens |
| Verdict | **Winner** — faster everywhere, only config that survives 4 agents | Sustained-decode collapse at 3+ agents |

**The one-line result:** at the same 4-node TP4 harness, FP8 was ~1.14x faster with 1
agent, ~1.45x faster with 3, completed 4 concurrent agents growing to 416K context
(where NVFP4 wedged and had to be killed), and held ~22.8 tok/s per agent in sustained
3-way decode where NVFP4 collapsed to ~0.5-2 tok/s per agent.

## Start here

An agent or operator must read [`AGENTS.md`](AGENTS.md) first. Then:

| Need | Read |
| --- | --- |
| Install from zero (OS + drivers + Docker + image + weights) | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Operate, deploy, start/stop, recover, run gates | [`docs/operations.md`](docs/operations.md) |
| Fabric and networking (switchless ring upstream / switched star adaptation) | [`docs/fabric.md`](docs/fabric.md) + [`docs/switch-adaptation.md`](docs/switch-adaptation.md) |
| The FP8 runtime recipe and its customizations | [`docs/production-recipe.md`](docs/production-recipe.md) |
| **Serve NVFP4 instead (or roll back)** | [`docs/recipe-nvfp4.md`](docs/recipe-nvfp4.md) |
| **Reproduce our benchmarks** | [`benchmarks/README.md`](benchmarks/README.md) |
| Our FP8-vs-NVFP4 results | [`docs/switch-adaptation.md` §4](docs/switch-adaptation.md) |

## What you need to download

For the **FP8 lane** (this repo's default):

- **Weights:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
  at revision `690b7052` — ~306 GiB, 62 shards. `scripts/fetch-fp8-weights.sh`
  downloads once on rank 0 and fans out to the other nodes over the fabric with
  manifest verification. Allow ≥330 GiB free per node.
- **DFlash2 drafter:** [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  at revision `bf582e4e` — ~2.3 GB, per node. CC BY-NC-ND 4.0: the whole lane is
  non-commercial.
- **Container image:** `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
  (~31 GB) — pull once, fan out, verify identical image IDs on all nodes.
- **Host extras:** stock `libnccl.so.2` extracted from the image (switch adaptation —
  no patched-NCCL build needed), the SM121 sparse-attn indexer patch, the GB10 Triton
  MoE JSON, and the adaptive-k scheduler — all shipped in this repo under `scripts/node/`.

For the **NVFP4 lane**: different weights (~182 GiB RedHatAI), a newer image tag
(`sm121-v12-dflash2-topkfix`), and a vision chat template — see
[`docs/recipe-nvfp4.md`](docs/recipe-nvfp4.md) for the exact list.

## How to launch (short version — both lanes)

```sh
# 0. Preflight (read-only) and site config
TP4_HOSTS='user@node0 user@node1 user@node2 user@node3' \
  ./scripts/agent-preflight.sh --report /tmp/tp4-preflight.json
cp cluster.env.example cluster.env && $EDITOR cluster.env

# FP8 on a switched (star) fabric — boot directly, worker-first:
./scripts/deploy.sh            # installs launcher + patches + MoE JSONs to nodes
# on each node, in ~/tp4:      RANK=3 bash ./launch-glm53-tp4.sh 3   # worker
#                              RANK=2 ...  RANK=1 ...  then head last:
#                              RANK=0 bash ./launch-glm53-tp4.sh 0
# The launcher runs the memory ritual (swappiness=0, drop_caches) itself.

# 1. Wait for readiness: GET /health == 200 (NOT /v1/models). Cold boot ~16 min.
# 2. WARM THE LANE before real traffic or benchmarks (JIT — see limitations):
python3 benchmarks/glm_warmup.py --endpoint http://<head>:8001/v1 --model glm-5.3-flash

# 3. Smoke it
curl http://<head>:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","temperature":0,"max_tokens":64,
       "messages":[{"role":"user","content":"Reply with READY."}]}'
```

The NVFP4 lane uses the tonyd2wild launcher + unconditional page-cache flusher on
every node (see [`docs/recipe-nvfp4.md`](docs/recipe-nvfp4.md)); boot order and
teardown rules are identical. **Never restart one serving rank in isolation** —
tear down all four and boot worker-first.

The API is OpenAI-compatible, exposed by rank 0, and has **no API key, TLS, rate
limit, or caller isolation**. Keep the management network and fabric private; expose
only through a trusted VPN or an authenticating reverse proxy.

## Our testing and findings

Full method, commands, and tables: [`benchmarks/README.md`](benchmarks/README.md)
(harnesses) and [`docs/switch-adaptation.md` §4](docs/switch-adaptation.md) (results).
Harness = growing-context agent probes (agents' contexts grow 32K→416K turn over
turn, thinking-low, exactly how we drive real agents) + a sustained-decode bench.
Both lanes warm, same endpoint/model identity, same flags.

| Metric | NVFP4 | FP8 |
| --- | --- | --- |
| C1 — 1 agent, →416K, total | 50.4 s | **44.1 s** (~1.14x) |
| C3 — 3 agents, →416K, total | 161.8 s | **111.3 s** (~1.45x) |
| C4 — 4 agents, →416K | **wedged at 192K** (79-94 s/turn), killed | **130.3 s**, all 4 complete |
| C3 danger zone (>256K), per-turn max | 10.4-16.2 s | **6.5-9.3 s** |
| Sustained decode, 3 agents | **collapse** — 1.4-5.5 aggregate (~0.5-2/agent) | **53.7 aggregate / 22.8 median per-agent**, steady |
| Sustained decode, 4 agents | collapse | 59.9 aggregate / 18.8 median per-agent |
| Growing-context aggregate tok/s | C1 8.6 / C3 10.0 / C4 n/a | **C1 9.7 / C3 12.7 / C4 13.2** |

**Why NVFP4 lost:** it cannot sustain 4 concurrent growing agents (C4 wedges) and its
sustained 3-way decode collapses to ~0.5-2 tok/s/agent — the exact shape of a real
multi-agent workload. Spec-decode tells the story: FP8 accepted 44-54 tok/s with 60-74%
draft acceptance (mean accept 3.4-4.0 of k=5) under load; NVFP4's draft cratered to
3-8 tok/s. FP8's native QAT also avoids the W4A4 conversion's reasoning-quality cost.

**Non-obvious things we learned** (each is documented in the repo):

1. **Warm the lane or your numbers are fiction.** First-run-after-boot probes read
   ~5x slow from Triton/TileLang kernels JIT-compiling mid-inference (~20 s/turn).
   We nearly mislabeled FP8 prefill because of it. Always warm, then measure.
2. **Per-agent turn cost is dominated by prefill re-serve, not decode** — ~7-11 s/turn
   at 200-300K context, roughly flat thanks to prefix caching. A strictly faster
   decode engine (we also A/B'd EXL3: 99 tok/s clean structured decode) did NOT win
   the agent benchmark, which is how we knew decode throughput was not the wall.
3. **Right-sizing the window (1M→480K) + seq (64→12) helped low-mid contexts ~3x** —
   but its apparent "high-context stall" was also JIT, not the config.
4. **Checkpoint provenance matters:** ModelOpt NVFP4 builds emit corrupted token IDs
   (tool-call parser desyncs); the RedHatAI compressed-tensors build is clean.
5. **Measure your own ceiling.** Our KV pin (16 GiB FP8 lane) is where *our* fleet
   lands; headroom varies GiB-to-GiB between identical nodes. A config that boots
   and answers a short prompt is not a config that works.

## Limitations (both recipes)

- **Context / KV pool.** FP8 lane: 480K window, fp8-KV pool ≈ 2.42M tokens
  (max concurrency ≈ 4.93x at the full 480K window). NVFP4 lane: same 480K window,
  pool ≈ 3.8M tokens — but three agents near 400K context ride the pool edge
  (10-13 s/turn from fragmentation). NVFP4's higher density is real (368 vs 512
  B/token/layer) but did not convert into more concurrent agents.
- **Sustained multi-agent decode.** NVFP4 collapses (above). FP8 holds 18.8-22.8
  tok/s/agent at 3-4 agents — good, not unlimited: plan on ~20 tok/s/agent.
- **JIT warmup after every boot.** ~16 min cold boot, then a warmup sweep before
  real traffic. Unwarmed lanes stall 20-30 s/turn at new shapes.
- **The DFlash2 drafter is CC BY-NC-ND 4.0** — both lanes are non-commercial.
- **No authentication on the API.** Trusted LAN/VPN only.
- **We run 4 nodes at TP4.** The image also runs at TP2 (two Sparks, 262K context) —
  see the tonyd2wild 2x repo linked in Credits. Our numbers are 4-node only.

## Credits

This repository adapts two community recipes onto our switched-fabric fleet; the heavy
lifting belongs to them:

- **FP8 lane:** [jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
  — the entire runtime recipe (Triton FP8 MoE + GB10 JSON, adaptive-k scheduler,
  DFlash2 integration, FP8 KV, switchless ring architecture). Our changes adapt
  the network layer to a switched RoCE fabric ([`docs/switch-adaptation.md`](docs/switch-adaptation.md)).
  jnardiello's lane descends from [Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark](https://github.com/Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark),
  informed by [alexellis's four-node recipe](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
  and [jspark3](https://github.com/jakejharris/jspark3) (acceptance instrumentation).
- **NVFP4 lane:** [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
  (+ his [2x TP2 repo](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark))
  — the SM121 container image, DFlash2 vLLM route, SM121 patches, and the deep
  engineering record. Full attribution and third-party terms:
  [`CREDITS.md`](CREDITS.md).
- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) ·
  NVFP4 quant: [RedHatAI](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) ·
  DFlash2 drafter: [inco.ai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0).
- Benchmark harnesses in [`benchmarks/`](benchmarks/) are ours, built to mimic our
  agent workload — reuse them, but expect different absolute numbers on different
  hardware.

## Documentation

| Need | Read |
| --- | --- |
| Install from zero | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Operate the cluster | [`docs/operations.md`](docs/operations.md) |
| Fabric / networking | [`docs/fabric.md`](docs/fabric.md) · [`docs/switch-adaptation.md`](docs/switch-adaptation.md) |
| FP8 runtime recipe | [`docs/production-recipe.md`](docs/production-recipe.md) |
| NVFP4 recipe + rollback | [`docs/recipe-nvfp4.md`](docs/recipe-nvfp4.md) |
| Benchmarks + results | [`benchmarks/README.md`](benchmarks/README.md) |
| Attribution + licenses | [`CREDITS.md`](CREDITS.md) |

Changes are recorded in [`CHANGELOG.md`](CHANGELOG.md). Offline validation:
`./scripts/check.sh` (no GPU, Docker, SSH, or weights needed).
