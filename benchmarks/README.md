# Benchmarks — reproduce our numbers

These are the exact harnesses we used for the **FP8 vs NVFP4 head-to-head** in
[`docs/switch-adaptation.md`](../docs/switch-adaptation.md). They are custom on
purpose: standard decode benchmarks do not predict how a serving lane *feels*
when several agents work on it. These scripts mimic our real agent workload:

- **agents run tool-call loops that grow context turn over turn** (read a doc →
  tool result appends → next instruction), climbing from ~32K toward ~416K tokens;
- **thinking-low** is the agent regime (reasoning is on and is NOT accelerated by
  speculative decoding — a clean-decode benchmark hides this);
- **1, 3, and 4 concurrent agents** are the shapes that matter;
- **the lane must be warm** — a first run after boot is JIT-polluted (see
  [Limitations](#limitations) below) and will lie to you by ~5x.

All scripts are read-only against a live endpoint: they issue chat completions
and nothing else. No restarts, no config changes. Endpoint, model name, and all
sizing come from CLI flags — nothing is hardcoded.

## Scripts

| Script | What it measures | When to use |
| --- | --- | --- |
| [`agent_sim.py`](agent_sim.py) | Full agent-turn battery per agent: tool-loop, longdoc read + follow-up, and a structured/code/prose output mix. Per-turn TTFT, decode tok/s, wall, prefix-cache hit. | The headline A/B harness. Run identically against both lanes. |
| [`grow_agent_probe.py`](grow_agent_probe.py) | The real 3-4 agent pattern: N **distinct** agents each growing their OWN context 32K→416K (+30K/turn). Per-turn wall time and where it stalls. | The C1/C3/C4 growing-context numbers in our results table. |
| [`bench_sustained_decode.py`](bench_sustained_decode.py) | N agents in CONTINUOUS sustained decode (2500-token outputs, mixed prose/code/structured, no idle gaps). | The metric that exposed NVFP4's sustained-decode collapse. |
| [`glm_warmup.py`](glm_warmup.py) | Post-boot warmup sweep (5K→400K growing contexts). | Run after EVERY boot, before ANY benchmark or real traffic. |

## Exact commands we ran

Both lanes served the same endpoint shape (`http://<head-node>:8001/v1`, model
`glm-5.3-flash`) so agents — and benchmarks — see one identity.

```bash
# 0. ALWAYS warm the lane first (also required after every boot; see limitations)
python3 benchmarks/glm_warmup.py --endpoint http://<head>:8001/v1 \
  --model glm-5.3-flash --max-ctx 400000

# 1. Growing-context probe — 1 agent (C1), then 3 (C3), then 4 (C4)
python3 benchmarks/grow_agent_probe.py --endpoint http://<head>:8001/v1 \
  --model glm-5.3-flash --thinking low --agents 1 \
  --max-ctx 400000 --doc-step 30000 --out fp8-c1.json
python3 benchmarks/grow_agent_probe.py ... --agents 3 --out fp8-c3.json
python3 benchmarks/grow_agent_probe.py ... --agents 4 --out fp8-c4.json

# 2. Sustained decode — the agent-feel metric
python3 benchmarks/bench_sustained_decode.py --endpoint http://<head>:8001/v1 \
  --model glm-5.3-flash --agents 3 --max-tokens 2500 --thinking low \
  --label fp8-c3 --out fp8-sustained-c3.json
python3 benchmarks/bench_sustained_decode.py ... --agents 4 --label fp8-c4 --out fp8-sustained-c4.json

# 3. Full agent battery (optional deeper dive)
python3 benchmarks/agent_sim.py --endpoint http://<head>:8001/v1 \
  --model glm-5.3-flash --scenario full --concurrency 1 --ctx 250000 \
  --thinking low --label fp8-c1 --out fp8-sim-c1.json
python3 benchmarks/agent_sim.py ... --concurrency 3 --label fp8-c3 --out fp8-sim-c3.json
```

Repeat the identical commands against the other lane (NVFP4) before comparing.
A/B rules we hold ourselves to:

1. **Warm both lanes** before measuring (JIT pollution otherwise).
2. **Same harness, same flags, same endpoint/model identity** on both lanes.
3. **Re-run anything anomalous.** A cold first C1 run once read ~5x slow and
   nearly mislabeled FP8 prefill; the stall was kernel JIT, not the config.
4. Quote the workload with every number. Single-stream tok/s is a statement
   about the prompt, not the engine (draft acceptance is content-driven).

## Limitations

- These harnesses measure **our agent shape** (tool loops, growing context,
  thinking-low). A chatbot or batch-inference workload will weight things
  differently — run your own mix before concluding.
- `agent_sim.py`'s C3 longdoc uses a deterministic doc builder: identical agents
  can hit each other's prefix cache. Treat its concurrent longdoc "cold prefill"
  as a cache-hit measure; use `grow_agent_probe.py` for the honest distinct-agent
  contention number.
- The server does not populate `usage.prompt_tokens_details.cached_tokens`, so
  prefix-cache reuse is inferred from follow-up TTFT, not read directly.
- Numbers are from a 4x GB10 (DGX Spark) TP4 cluster on a switched RoCE fabric.
  Different node counts, fabrics, or clocks will shift results.
