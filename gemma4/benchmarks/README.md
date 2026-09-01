# Gemma4-E2B TP benchmark templates

These are the reproducible 4x4x4 Pathways benchmarks for the current
Gemma4-E2B GRPO optimization stack. Each entry point renders a JobSet, verifies
the mesh and the complete patch ConfigMap, submits it, waits for completion,
downloads the head log, and prints a phase summary.

For the control-plane bottleneck these runs are meant to test, see
[`../PATHWAYS_DP_BOTTLENECK_SUMMARY.md`](../PATHWAYS_DP_BOTTLENECK_SUMMARY.md).

| Entry point | Rollout mesh | Device allocation | Purpose |
|---|---:|---:|---|
| `run_tp1_dp64.sh` | TP1 x DP64 | trainer gets first 64 and sampler last 64 core-devices | corrected TP1 baseline |
| `run_tp2_dp32.sh` | TP2 x DP32 | trainer gets first 64 and sampler last 64 core-devices | optimized current baseline |
| `run_tp4_dp32.sh` | TP4 x DP32 | sampler/trainer colocated on the full 128 core-devices | full-slice TP4/DP32 benchmark |

All profiles use 64 TPU chips (128 core-devices), 20 training steps, the
immutable packing image digest, and the same model/dataset/sampling contract.
TP4/DP32 needs the full sampler fraction because `128 / TP4 = DP32`; the TP1
and TP2 comparisons retain the historical, disjoint half-slice meshes in one
Pathways JobSet. In MaxText's placement terminology those 0.5/0.5 profiles are
not device-colocated. TP4 uses the actual colocated 1.0/1.0 mode; changing it
to 0.5/0.5 would expose only 64 sampler devices and resolve TP4 x DP16.

## Run

From the repository root:

```bash
gemma4/benchmarks/run_tp1_dp64.sh
gemma4/benchmarks/run_tp2_dp32.sh
gemma4/benchmarks/run_tp4_dp32.sh
```

Safe render-only preflight:

```bash
gemma4/benchmarks/run_tp4_dp32.sh --render-only
```

Useful controls:

```bash
# Choose the JobSet name.
gemma4/benchmarks/run_tp2_dp32.sh --run-name my-g4-tp2

# Submit after validation, but do not wait.
gemma4/benchmarks/run_tp2_dp32.sh --submit-only

# Attach to an already-created JobSet and collect its final result.
gemma4/benchmarks/run_tp2_dp32.sh --run-name my-g4-tp2 --attach
```

The scripts default to:

- context `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap`
- namespace `ubench-regression-tests`
- queue `multislice-queue`, priority `medium`
- artifacts `gs://ubench-logs/alisachen-tpu7x/runs/<run>/`
- local render/log/summary `$HOME/gemma4-benchmark-results/<run>/`

Override those with `BENCH_CONTEXT`, `BENCH_NAMESPACE`, `BENCH_QUEUE`,
`BENCH_PRIORITY`, or `BENCH_RESULT_ROOT`.

## Patch contract

`patches.sha256` pins all 33 files from the repository's top-level `patches/`
directory. The runner verifies their hashes, copies all of them into a temporary
chart, and then verifies that the rendered ConfigMap contains all 33 keys. This
closes the old staging-only gap where the checked-in chart carried only a subset
of the live patch bundle.

The workload launcher executes the 26 patches used by the optimized run and
fails before training if a required patch is absent or prints a failure banner.
Feature-gated patches remain mounted and applied, while the profile env controls
activation. The proven active stack is:

- token padding gap `32768` (generation-correctness gate)
- sequence packing `8192` and train micro-batch `8`
- keep-KV weight sync
- 64 math-verifier workers
- continuous-decode global-EOS reduction removal
- vocab-sharded sampler padded-row fix for TP greater than one
- verifier parse/verify timeout set to five seconds

The verifier timeout patch does not bound the separate raw SymPy
`parse_expr()` fallback. Consequently the result summary calls out reward-phase
outliers instead of silently discarding them. Likewise, `SKIP_JAX_PRECOMPILE=1`
matches the measured runs; unseen prefill buckets may appear as rollout compile
tails and are reported as outliers.

## Files

- `launcher.sh`: frozen, fail-closed workload launcher and patch order
- `profiles/*.yaml`: exact mesh/env overlays
- `run_profile.sh`: render, verify, submit, wait, and collect orchestration
- `validate_render.py`: offline JobSet/ConfigMap contract checks
- `summarize_log.py`: median/mean/p90/max phase and tail report
- `patches.sha256`: exact live patch bundle checksum contract
