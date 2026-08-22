# tpu-rl

TPU v7x (Ironwood) RL benchmarking — runbooks, launchers, and command references.

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

## Gemma4 vs Qwen3 on the same 4x4x4 — performance comparison and analysis (2026-08-21)

Both runs: 64-chip tpu7x 4x4x4, Pathways, GRPO, 2,048 completions/step
(256×8), full tianyu optimization stack, medians over warm steps.
Full details: [gemma4/GEMMA4_PERF_REPORT.md](gemma4/GEMMA4_PERF_REPORT.md).

| | qwen3-0.6b (`tianyu-rl-17a`) | gemma4-e2b (`g4r1c`) |
|---|---|---|
| Global step (median) | **133.6 s** | **271.2 s** |
| Rollout | 105.4 s | 145.5 s |
| Trainer segment | ~25.1 s | 108.1 s |
| Rewards (in-step) | 2.8 s | 12.6 s (spiky 4–103) |
| Completion cap / truncation | 8192 / ~23% | 4096 / ~81% |
| Tokens generated per step | ~7.6 M (osl_mean 3708) | ~6.8 M (osl_mean 3332) |
| **Rollout throughput** | **~72k tok/s** | **~48k tok/s** |
| tmbs (grad-acc microbatch) | 32 (8 microbatches) | 1 (256 microbatches) |
| CD window / EOS interval | 128 / 8 | 64 / 1 |
| Model route | native `Qwen3ForCausalLM` | MaxText-backed (`MaxTextForCausalLM`) |
| Rewards recipe | math (check_numbers ~0.39) | format-only weights (math_verify still runs) |

### Analysis — where the 2× gap comes from

1. **Mostly legitimate model cost (rollout).** Per generated token gemma is
   only ~1.5× slower (48k vs 72k tok/s) for a model with ~3–4× the compute
   per token (35 unscanned layers vs 28 scanned, head_dim 256/512 vs 128,
   262k vs 152k vocab, per-layer embedder + KV-share lookups). Rollout-side
   the gemma port is running efficiently.
2. **Trainer microbatching is the biggest avoidable gap (108 s vs 25 s).**
   Gemma is pinned at `train_micro_batch_size=1` (Fix #21: the RL trainer
   materializes full 262k-vocab per-token logprobs; tmbs=4 OOM'd at 130 G).
   Qwen runs tmbs=32. The upstream pretrain recipe
   (tpu-recipes gemma4-2b 8k-bf16) fits `per_device_batch_size=2` at seq
   8192 on the same hardware — pretraining's fused cross-entropy never
   materializes the logits. Structural headroom: adopt chunked/fused logprob
   computation in the RL trainer; tactical: tmbs=2 scout (g4s2).
3. **Decode dispatch settings differed** — qwen: CD window 128 + EOS-check
   interval 8; gemma: 64/1. Tested in scout g4s1: deepening to 128/8 did
   NOT improve gemma rollout (~150 s) and inflated reward-verify cost —
   gemma's rollout at this shape is not dispatch-bound; window 64 already
   amortizes host round-trips.
4. **Regime density**: 81% of gemma completions hit the 4096 cap (OSL p50 =
   cap) vs 23% for qwen at 8192 — gemma's decode batch stays maximally full,
   so its per-step token volume is only ~11% lower despite the halved cap.
5. **Reward compute**: qwen's stays at ~2.8 s (parallel pool);
   gemma's math_verify on longer truncated math text is spiky (4→103 s) and
   is the main step-to-step variance source — smoothing it is worth ~15 s
   off the median.

Validity: both runs 20/20 steps, exit 0, zero restarts; qwen rewards in the
0.13–0.16 band; gemma rewards rising 0.017→0.027 with textbook grad-norm
decay (337→6) and kl=0 at β=0.

## Native-route investigation (2026-08-22) — TP>1 enabled; generation quality gap found

Mechanics COMPLETE: shard-interleaved qkv/gate_up fusion (matching
tpu_inference's merged-linear layout law; `GEMMA4_QKV_N_SHARDS`/
`GEMMA4_NATIVE_TP` = rollout TP) runs e2e at TP=4 with 0 unmapped keys and
clean exits (g4ns). Fork carries the same fix (35b790363).

QUALITY GAP: native-route generations are fluent but base-model-like
(instruction-ignoring, prompt echoes) while the MaxText route answers
correctly on identical prompts. Single-variable elimination (all smokes
batch 8, gemma4-e2b):

| Variable | Test | Verdict |
|---|---|---|
| TP interleave layout | TP=4 vs TP=1 (g4ns vs g4ns1) | identical output — exonerated |
| Chat template | vs coherent g4dlv2, same `<turn|>` markers | exonerated |
| HF weights | both routes load_format=dummy + push | exonerated |
| Continue-decode | CD off (g4ns2b) | still rambles — exonerated |
| Sharded sampling | SHARDED_SAMPLING=0 (g4ns3) | still rambles — exonerated |
| Mapping dst coverage | audit: all linears mapped; unmapped norm scales identity-init | exonerated |
| Embedding scales | token (÷√hidden hook) + per-layer (raw/runtime-scale both sides) | exonerated |
| Final logit softcap | native applies tanh cap | exonerated |

Remaining hypothesis: semantic gap inside tpu_inference's Gemma4 attention
implementation itself (e.g. `self.scaling = 1.0` marked "Assuming ..." in
source) — t6gn8's earlier "coherent" grade likely over-generous. Next step
if pursued: logit-parity harness (push weights, compare per-token logits vs
MaxText forward on identical input). Until then the MaxText route remains
the production/benchmark path.

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
