# Switch-Adapted FP8 Deployment (Scope A) — deployment + benchmark notes

**Fork:** `1890peachypeachy/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless`
**Upstream:** `jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless`
**Date:** 2026-09-06 · **Status:** verified live + benchmarked

This doc records how this fork deploys jnardiello's GLM-5.3-Flash FP8 (zai-org) stack
on a **switched** RoCE fabric, and the results we measured against the NVFP4 lane it
replaced. Read `docs/production-recipe.md` upstream first for the engine rationale; this
file only records what differs on a switch and how it performed.

---

## 1. Why a fork / what changed for a switch

jnardiello's upstream stack is a self-contained **switchless ring** deployment (four
uncabled diagonals, patched NCCL, per-node netplan ring re-addressing, `tp4ctl
fabric-check` expecting two fabric interfaces per node). On a **switched** fabric
(CRS504-style star, one CX-7 fabric port per node) that scaffolding is unnecessary and
incompatible. This fork keeps the entire runtime recipe (image, Triton FP8 MoE + GB10
JSON, adaptive-k scheduler, DFlash2 drafter, FP8 KV) and adapts only the network layer.

**Switch adaptations (all in this fork):**

| File | Change |
| --- | --- |
| `cluster.env.example` | Single fabric HCA `rocep1s0f1`; `MGMT_IF` = fabric iface; `FABRIC_TARGETS` informational only; stock `NCCL_DIR`; API 8001 / `glm-5.3-flash`; 480K window / seq 12 |
| `scripts/launcher/launch-glm53-tp4.sh` | HCA validation relaxed from ≥2 comma-separated HCAs to **≥1** (a switch has one fabric HCA per node) |
| `docs/switch-adaptation.md` | this file |

**Deliberately NOT used on a switch:** `render-netplan.sh` + per-node `40-cx7.yaml`
(keep the switch netplan), the upstream patched-NCCL build, `tp4ctl up` /
`fabric-check` (expects the switchless ring + a `tp4-flusher` systemd unit), and
`deploy-host.sh`.

### Key network facts verified on our fleet (rank 0..3)
- Fabric: `10.73.0.0/24` switched star, one fabric link per node = `enp1s0f1np1`
  (HCA `rocep1s0f1`, GID index 3). Rank n carries `10.73.0.<n+1>`.
- The second active RDMA link (`roceP2p1s0f1`, on the LAN `192.168.0.x`) is **not** the
  fabric and must NOT join `NCCL_IB_HCA`.
- The launcher's fixed NCCL block (`NCCL_ALGO=Ring`, `NCCL_SKIP_TREE_CONNECT=1`,
  `NCCL_P2P_LEVEL=SYS`, `NCCL_NET=IB`, GID 3) is switch-compatible **as-is** — ring over
  a switch is the standard DGX config.
- **Stock NCCL satisfies the launcher's `NCCL_DIR` preflight.** Extract it from the image
  (not a patched build): `docker run --rm <IMAGE> cp /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 $HOME/nccl-patched/`.
  The launcher `LD_PRELOAD`s + mounts this. (The image also ships NCCL at the same path;
  deep_ep logs a benign "duplicate NCCL" import warning — non-fatal.)

## 2. Launching on a switch (bypassing `tp4ctl up`)

`tp4ctl up` is unusable here (fabric-check + systemd flusher assumptions). Boot directly
with the per-node launcher, **worker-first** (rank 3 → 2 → 1 → 0):

```bash
# on each node, in ~/tp4 (deployed by scripts/deploy.sh):
RANK=<3|2|1|0> bash ./launch-glm53-tp4.sh <3|2|1|0>
```

The launcher itself does the memory ritual (swappiness=0, drop_caches) + local teardown +
`docker run`. Rank 0 exposes the OpenAI-compatible API on `0.0.0.0:$API_PORT`.

**Passwordless-sudo note:** the launcher calls `sudo docker`, `sudo sysctl -qw ...`, and
`sudo tee /proc/sys/vm/drop_caches`. A tight-scoped passwordless-sudo drop-in covering
exactly those commands is required on a non-NOPASSWD node:
```
<user> ALL=(root) NOPASSWD: /usr/bin/docker, /usr/sbin/sysctl, /usr/bin/tee
```

**Weight fan-out:** pull FP8 weights (zai-org/GLM-5.3-Flash, ~306 GiB) ONCE on one node,
then fan out over the fabric (`rsync -a` on `10.73.0.x`, never LAN) to the other three.
Verify shard count (62 safetensors + config.json) before boot.

## 3. Boot signatures (verified live 2026-09-06)

A healthy first boot shows, in rank-0 logs:
- `Using TRITON Fp8 MoE backend` (from `fp8.py`, choosing TRITON over the fallback list)
- `Using configuration from ...E=288,N=512,...NVIDIA_GB10,dtype=fp8_w8a8,...json` (GB10 MoE JSON mounted)
- `adaptive-k: AdaptiveKScheduler active (k_lo=3 k_hi=5 ...)` (per-request adaptive-k)
- NCCL world formed: `world_size=4 rank=<n> ... backend=nccl ... tcp://10.73.0.1:29521`
- CUDA graphs: PIECEWISE 21/21, FULL 17/17, DFlash2 drafter graphs captured
- `/health` → 200; `GET /v1/models` lists `glm-5.3-flash`

Cold boot ~16 min (306 GiB weight load + graph capture). **Kernels JIT during the first
inference at new shapes** — a single warm-up sweep (growing context 5K→400K) after boot
removes the ~20 s/turn JIT stall on the first growing-context probe. A first C1 probe run
right after boot will look ~5× slow (JIT pollution); re-run warm before judging.

## 4. Benchmark results — FP8 vs NVFP4 (apples-to-apples)

Harness: `benchmarks/grow_agent_probe.py` (vendored in this repo)
(`--max-ctx 400000 --doc-step 30000 --thinking low`, growing 32K→416K by +30K/turn).
Both lanes warm, same endpoint/model name. NVFP4 = RedHat W4A4 conversion, marlin,
DFlash2 k=3. FP8 = zai-org native QAT, Triton MoE, adaptive-k 3-5.

### Wall-clock (decode+prefill combined latency)
All runs at **thinking-low** (`enable_thinking: true, reasoning_effort: "low"`) —
the agent regime; reasoning tokens are not accelerated by spec decode. Exact
request shape: [`benchmarks/README.md`](../benchmarks/README.md).

| Run | NVFP4 | FP8 | Winner |
| --- | --- | --- | --- |
| C1 (1 agent) | 50.4 s | **44.1 s** | FP8 ~1.14× |
| C3 (3 agents) | 161.8 s | **111.3 s** | FP8 ~1.45× |
| C4 (4 agents) | wedged at 192K (79-94 s/turn), killed | **130.3 s**, 4/4 to 416K | FP8 only |

Danger zone (>256K) per-turn max: NVFP4 C3 10.4-16.2 s vs FP8 C3 6.5-9.3 s; FP8 C4
10.8-13.6 s. NVFP4 **cannot complete C4** — FP8 is the only config that serves 4
concurrent agents growing to 400K+.

### Sustained decode throughput (tok/s — the agent-feel metric)
Sustained decode bench, 2500-token outputs, mixed prose/code/structured, thinking-low.
| Workload | NVFP4 | FP8 |
| --- | --- | --- |
| 3-agent sustained | **collapse to 1.4-5.5 aggregate (~0.5-2/agent)** | **53.7 aggregate / 22.8 median per-agent**, steady |
| 4-agent sustained | collapse | 59.9 aggregate / 18.8 median per-agent, steady |

Spec-decode under load: FP8 accepted 44-54 tok/s, draft accept 60-74%, mean accept length
3.4-4.0 of k=5 — healthy. NVFP4 craters to 3-8 tok/s draft under sustained 3-way decode.

### Verdict
FP8 (this fork) is the production winner on our 4×DGX-Spark / switched-fabric setup for a
3-4 concurrent agent workload at 400K+ context: it is ~1.14× faster single-agent, ~1.45×
faster at 3 agents, uniquely completes 4 agents to 416K, and eliminates the sustained-
decode collapse (22.8 tok/s/agent vs NVFP4's ~0.5-2). Native FP8 QAT also avoids the
W4A4-conversion quality loss of NVFP4.
