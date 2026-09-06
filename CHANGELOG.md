# Changelog

All repository changes are recorded here incrementally. Entries describe concrete
effects; release sections are created only when the owner explicitly authorizes a
release.

## Unreleased

### Added

- Added `benchmarks/` with the four harnesses used for the FP8-vs-NVFP4
  head-to-head (`agent_sim.py`, `grow_agent_probe.py`, `bench_sustained_decode.py`,
  `glm_warmup.py`) and a README with the exact commands, A/B rules, and method
  limitations, so the published results are reproducible from the checkout.
- Added `docs/recipe-nvfp4.md`: the NVFP4 recipe as a runnable alternative lane
  (weights, image, launch config, measured limitations) with the tonyd2wild
  repository referenced as the deep engineering record.
- Rewrote the README as a two-recipe entry point: what to download for each lane,
  how to launch either, the measured FP8-vs-NVFP4 results, shared limitations, and
  full attribution for both adapted recipes.
- Documented the exact benchmark request shape (thinking-low, temperature, output
  sizes, context growth, concurrency, streaming) in `benchmarks/README.md` and
  pointed the results section at the vendored harnesses instead of the private
  benchmark repository.
- Extended `CREDITS.md` with the NVFP4 recipe source (tonyd2wild's four-node
  repository) and the RedHatAI NVFP4 quant.

### Fixed

- Corrected `SPEC_EXTRA_JSON` in `cluster.env.example`: the switch-adaptation
  release shipped it with escaped inner quotes that fail the launcher's JSON
  validation (and the offline tests). The value now parses as the intended JSON
  fragment.
- Corrected the `FABRIC_TARGETS` template values to a consistent RFC 5737 ring
  plan (last octet = node number = rank + 1, peers ordered by ascending link
  index) so the upstream ring tooling (`render-netplan.sh`) validates them even
  though a switched deployment does not use it.
- Updated the launcher fallback expectations in the offline tests to the
  switch-adapted single-HCA values (`rocep1s0f1`, fabric iface for management)
  instead of the upstream two-port ASUS values.

### Changed

- Documented the workload priorities and the quality, prefill-throughput, and prose
  non-regression requirements.
- Added a README badge linking to the maintainer's X profile.
- Added this changelog and made a matching `Unreleased` entry mandatory for future
  code, configuration, and documentation changes.
- Added one offline validation command for syntax, links, command help, model
  manifests, chat-template rendering, host lifecycle, preflight, and the adaptive-k
  policy.

### Changed

- Renamed the repository from `GLM-5.3-Flash-FP8-4-DGX-Spark` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` and clarified that its verified
  ConnectX-7 fabric connects the nodes directly without a dedicated network switch.
- Renamed the repository from `tp4-glm53-fp8-gx10` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark`.
- Centered the README tables for clearer presentation on GitHub.
- Documented the configured 256K (262,144-token) context window in the README and
  corrected the GitHub About description.
- Consolidated operator documentation into five task-oriented guides and shortened
  the repository and agent entry points.
- Moved the controller, launcher, and public node assets under `scripts/` while
  preserving their installed paths on cluster hosts.
- Moved the remaining ignored node configuration and operator tools under
  `scripts/node/`, leaving no root-level `node/` directory.
- Documented `scripts/node/` as the source of files installed on cluster hosts, including
  runtime patches, host configuration, model manifests, and the patched NCCL build.
- Published the 2026-09-05 loopback benchmark aggregate and its method limits without
  private paths, raw logs, or node addresses.
- Reworked the README benchmark summary into a six-metric comparison against a fixed
  initial baseline, with protocol limits and secondary metrics kept in `docs/bench.md`.
- Consolidated attribution and third-party terms in `CREDITS.md`.
- Recorded the permanent project rule to run checks locally and never use GitHub
  Actions.

### Fixed

- Made lifecycle commands fail closed on unverifiable rank, container, or flusher state;
  failed starts now clean all four ranks, and effective overlays cannot change TP4
  cardinality or the configured container name.
- Rejected out-of-range, malformed, and ambiguous leading-zero fabric IPv4 addresses
  before netplan generation.
- Made help, deploy discovery, and offline tests tolerate optional internal profiling,
  collective microbenchmark, MoE tuning, and publication tooling being absent.
- Updated public links and source comments after the documentation consolidation.
- Clarified parallel-stack discovery, fabric versus manual RDMA checks, boot ordering,
  feature rollback boundaries, and fail-closed backup and validation of a rebuilt NCCL
  candidate.

### Removed

- Removed the repository benchmark harness, published performance results, and
  benchmark-only deployment and validation hooks; post-boot functional gates now live
  in the operations guide.
- Removed historical studies, raw experiment indexes, internal tuning tools, mirror
  tooling, and the outdated performance graphic from the public repository surface;
  local copies remain available outside the published file set.
- Removed the GitHub Actions workflow; offline validation remains available through
  `scripts/check.sh`.
