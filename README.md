# tpu-rl

TPU v7x (Ironwood) RL benchmarking — runbooks, launchers, and command references.

## Docs
## Operating guide
- [OPERATING.md](OPERATING.md) — **start here**: access, one-command launch/watch/gates, pre-flight,
  known failure modes. Launchers and the helm chart live in `gemma4/`, `chart/`; helpers in `tools/`.

### Background
- [gemma4/GEMMA4_E2B_RUN.md](gemma4/GEMMA4_E2B_RUN.md) — commands for running gemma4-e2b GRPO RL
- [gemma4/GEMMA_BENCH_PLAN.md](gemma4/GEMMA_BENCH_PLAN.md) — benchmark ladder
- [RUNBOOK.md](RUNBOOK.md) — Fix Catalog #1–21

## Launchers
- [smoke/gemma4_e2b_smoke_maxtext_route.sh](smoke/gemma4_e2b_smoke_maxtext_route.sh) — smoke (2 steps, batch 8, 2x2x1), MaxText sampler route — **VALIDATED** (exit 0, 2/2 steps, ckpts, 39/39 math-verified)
- [smoke/gemma4_e2b_smoke_native_route.sh](smoke/gemma4_e2b_smoke_native_route.sh) — smoke, NATIVE tpu_inference `Gemma4ForCausalLM` sampler (exercises the gemma4 weight mapping)
- [gemma4/gemma4_e2b_benchmark_4x4x4_maxtext_route.sh](gemma4/gemma4_e2b_benchmark_4x4x4_maxtext_route.sh) — real benchmark run (30 steps, batch 256, 4x4x4)
- [chart/values-4x4x4.yaml](chart/values-4x4x4.yaml) — helm overlay for the 4x4x4 slice

## Native route — status (2026-08-17)
Weight mapping ([`weight_mapping/gemma4.py`](https://github.com/alisachen-google/maxtext/blob/ironwood-rl-gemma4/src/maxtext/integration/tunix/weight_mapping/gemma4.py), branch
[`ironwood-rl-gemma4`](https://github.com/alisachen-google/maxtext/tree/ironwood-rl-gemma4)
in the maxtext fork) is implemented and offline-validated (22/22 param-group
coverage, HF-name + shape/rank checks, embedding un-scale hook). Registry
dispatch proven in-run. First e2e attempt is **blocked on cluster capacity**:
head pod Pending — cpu-np nodepool full, autoscaler FailedScaleUp "GCE out of
resources" (us-central1-c). Validation gate when it runs: pearson > 0.8.
Full details and code pointers: [scripts/ironwood/README.md](https://github.com/alisachen-google/maxtext/blob/ironwood-rl-gemma4/src/maxtext/trainers/post_train/rl/scripts/ironwood/README.md) in the maxtext branch. Key code: [registry dispatch](https://github.com/alisachen-google/maxtext/blob/ironwood-rl-gemma4/src/maxtext/integration/tunix/weight_mapping/__init__.py), [adapter](https://github.com/alisachen-google/maxtext/blob/ironwood-rl-gemma4/src/maxtext/integration/tunix/tunix_adapter.py), [MaxTextForCausalLM adapter](https://github.com/alisachen-google/maxtext/blob/ironwood-rl-gemma4/src/maxtext/integration/vllm/maxtext_vllm_adapter/), [native Gemma4ForCausalLM](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/gemma4.py), [multimodal prompt-logprobs check](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/tpu_runner.py), [tunix weight push](https://github.com/google/tunix/blob/main/tunix/generate/vllm_sampler.py).

## Run log archive (gs://alisachen/tpu_rl/ironwood/logs/)

Full head-pod logs for every 2026-08-17/18 gemma4-e2b bring-up run:

| Log | Run | Outcome |
|---|---|---|
| `t6g9-head.log` | maxtext route, tmbs=4 | trainer OOM: 130G HLO temporaries (Fix #21) |
| `t6g10-head.log` | maxtext route, tmbs=1 | **PASS** — first gemma4 T6 (exit 0, 2/2 steps) |
| `t6g11-head.log`, `t6g11-worker.log` | maxtext route bisect | Pathways cache-key mismatch (Fix #22 evidence) |
| `t6g12-head.log` | maxtext route, limits restored | same (partial log — creds expired mid-run) |
| `t6g13-head.log` | maxtext route control | **PASS** — proved Fix #22 (cluster clean again) |
| `t6gn4/t6gn5-head.log` | native route attempts | cache-key mismatch window (Fix #22) |
| `t6gn6-head.log` | native, mapping v1 | ran e2e but gibberish (wildcard keys never matched → random sampler weights) |
| `t6gn7-head.log` | native, mapping v2 | 65 unmapped q/k/v (fused qkv_proj discovery) |
| `t6gn8-head.log` | native, mapping v3 | **PASS** — 0 unmapped, weight mapping validated e2e |

Trajectory logs / tensorboard / checkpoints per run: `gs://ubench-logs/alisachen-tpu7x/runs/<run>/`.
