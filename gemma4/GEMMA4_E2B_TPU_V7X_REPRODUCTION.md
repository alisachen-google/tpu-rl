# CMCS AI Ninja Team — gemma4-E2B TPU v7x GRPO on GKE

**Author:** alisachen@
**Collaborator:** gutianyu@, xiaotongyang@, hosseins@, mattdonati@
**Created:** 2026-09-02
**Last updated:** 2026-09-08

---

## 0. Overview

This guide reproduces the TPU v7x side of the TPU v7x vs. GB200 RL comparison for
**gemma-4-E2B**: GRPO post-training (DAPO clip-higher, ε=0.2/0.28) of gemma-4-E2B-it on
OpenMathInstruct-2 using tunix (MaxText + tpu-inference, colocated).

**Headline result: 175.6 → 35.7 s/step (4.9×), with reward correctness `check_numbers`
0.02 → 0.58 (GPU parity).** Unlike the Qwen3 line, the gemma4 campaign surfaced a
**corrupted-checkpoint bug** that capped reward at ~2% until fixed; the fix is a required
part of this reproduction (§1.2a).

---

## 1. Setup

### 1.1 Image

Same JAX-0.10.2 client/server alignment discipline as the Qwen3 line.

**Required images:**
- MaxText workload image: `maxtext-rl-ironwood:2026-08-24-packing` (sha f7c957d0)
- Pathways server image: `server:20260730-jax_0.10.2`
- Pathways proxy image: `proxy_server:20260730-jax_0.10.2`

The runtime optimizations ship as **self-verifying start-time patches** applied at pod
start (not baked into the image), so the pinned image stays fixed across the whole
optimization ladder and every lever is a single-variable toggle. Full patch catalog:
`gemma4/patches/` (36 files). Deployable manifest: `g4r55-best-perf-all-in-one.yaml`
(all patches embedded).

### 1.2 Model & Dataset

- **Model:** gemma-4-E2B-it — `model_name=gemma4-e2b`
- **Tokenizer:** `google/gemma-4-E2B-it`
- **Checkpoint:** `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/0/items` (the
  **verified** conversion — see §1.2a)
- **Dataset:** OpenMathInstruct-2, `hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet`, split `train_1M`

### 1.2a Checkpoint conversion — REQUIRED (unique to gemma4)

The gemma4 checkpoint originally in use (`gemma4-e2b-raw`) had a **corrupted embedding
table**: every row normalized to √1536 and rotated vs. HF, worst on the turn/stop tokens.
Effect: the model was fluent but reward-blind (`check_numbers` ≈ 0.02) and never emitted its
stop token (85% cap-truncation). Root cause and fix in `GEMMA4_REWARD_BUG_REPORT.md`.

**Fix:** reconvert from the verified bf16 HF safetensors via MaxText's official
`checkpoint_conversion/to_maxtext.py` (native `gemma4-e2b` mapping) → `gemma4-e2b-raw2`.
Gated by two checks:
- **Byte audit** — embedding rows vs HF safetensors, require cos ≥ 0.999 (measured 0.999999)
- **Forward-pass logit check** — `tests/utils/forward_pass_logit_checker.py`, KL < 0.015 vs HF (measured 5.3e-3)

Job: `g4ckpt-reconv-job.yaml`. This is the single highest-impact step in the reproduction —
without it every reward number is invalid.

### 1.3 Preprocessing

Native runtime preprocessing in MaxText `train_rl`. Two gemma4-specific requirements:

**Prompt formatting:** gemma-4-E2B-it is instruction-tuned inside the canonical turn frame
(special tokens `<|turn>`=105 / `<turn|>`=106). We render the gemma-3-style textual prompt
via `gemma_data.py` (`dataset_processor_path` hook). On the **fixed** checkpoint the model
emits the trained stop token **id 106 (`<turn|>`)** natively, retired on-device via
`patch_runner_eos.py` (stop ids `[1,106,50]`). No textual-marker stop is needed (the
`patch_textual_stop.py` lever was retired after the checkpoint fix — validated in g4r69).

**Sequence limits:** response cap 4096 = `max_target_length=8192` − `max_prefill_predict_length=4096`.
(Note: prompts average ~155 tokens, so prefill is heavily oversized — see §4.)

### 1.3 Reward function

`check_numbers` (+1.0 correct via exact / whitespace / math-verify) + `match_format_exactly`
(+0.1). Cross-validated against the GPU/verl port (`maxtext_math_reward.py`): weights
identical (+1.0/+0.1), equivalence paths agree at the worker level. Guarded by
`patch_verify_timeout.py` (per-item sympy timeout, kills a 156 s reward spike the Qwen line
never fixed).

---

## 2. Launching Runs

**Job manifest:** `gemma4/g4r55-best-perf-all-in-one.yaml`
**Launch:** `kubectl apply -f gemma4/g4r55-best-perf-all-in-one.yaml`

### 2.1 Job manifest breakdown

| File section | Functionality |
|---|---|
| ConfigMap `g4r55-patches` | 36 start-time patches: continue-decode fix (PR #3374), RPA v3 block sizes (#3102), no-EOS early exit, **rr DP routing** (the −70 s lever), async detok/logprobs/reward, textual stop (retired), verify-timeout spike fix, gemma-3 prompt, + phasetime/lendist/reward-dump instrumentation |
| ConfigMap `g4r55-launcher` | the launch script (applies the patch roster, then `train_rl`) |
| JobSet `g4r55` | pathways-head + 16 pathways-workers on one 4×4×4 TPU v7x slice |

**Frozen invariants** (parity with the Qwen line and the GPU side):
`batch_size=256`, `rl.num_generations=8`, `rl.grpo_beta=0.0`, `rl.grpo_epsilon=0.2`,
`rl.epsilon_high=0.28`, response cap 4096, `decode_sampling_temperature=0.8`,
`decode_sampling_top_k=50`, `decode_sampling_nucleus_p=0.95`, `learning_rate=1e-6`,
`gradient_clipping_threshold=1.0`, `num_test_batches=0`. Topology **TP1 × DP128**, CD
window 512.

### 2.2 Collecting results

Each run writes TensorBoard events. Step time is the steady-state **median over steps ≥3**
(steps 0–2 excluded: XLA/shard_map compile, checkpoint restore, profiler capture; a
mid-run rollout wobble excluded). Validation disabled, no checkpoint writes in the window.

Each step decomposes into 4 phases (no reference inference, β=0):
- **Rollout** — generation submission → last sequence returned
- **Other/Rewards** — reward scoring (async, off the critical path)
- **Train** — one optimizer update (TMBS 8, fsdp128)
- **Weight sync** — refit updated weights into the rollout engine

Retrieve: `tensorboard --logdir gs://ubench-logs/g4r68/tensorboard/`

### 2.3 Scaling table

Single-slice 4×4×4 only (64 chips / 128 devices). One topology probe: TP4×DP32 = 88–92 s
vs TP1×DP128's L1-class config — on healthy weights **TP4 beats TP1 by ~40 s** at matched
config (reverses the Qwen-era TP1-wins result), but the campaign's best (window 512) is TP1.

### 2.4 Step time breakdown (BEST config, g4r68, 20 steps)

| Phase | Median (s) | Share | Notes |
|---|---:|---:|---|
| Rollout | 31.4 | 88% | Generation of 2,048 sequences (256×8), DP128×TP1, continue-decode depth 512, sync scheduling, vocab-sharded sampling, rr DP routing |
| Other/Rewards | 0.23 | 0.6% | Reward scoring — async + parallel (64 procs) + verify-timeout; was 12 s serial in the baseline |
| Train | 3.8 | 11% | One optimizer update, TMBS 8, fsdp128, flash attention |
| Weight sync | 0.45 | 1.3% | Weight refit into the rollout engine (keep-KV + jit-unroll) |
| **Total** | **35.7** | **100%** | median; mean 35.8, std 0.78 s (n=16 clean) |

### 2.5 Workload shape (fixed checkpoint)

- Mean ISL ~155 tokens (p50 ~133)
- **Mean OSL ~1,450–1,570 tokens, with 3–11% of sequences hitting the 4,096 cap** — and
  falling as RL trains (OSL 1750 → 1350 over 8 steps). Contrast the broken checkpoint:
  OSL ~3,550, 85% cap-truncation.
- **Every completion terminates naturally on id 106** (`…<answer>X</answer><turn|>`);
  `eos_at_cap` = 0, `with_eot` (textual marker) ≈ 0.
- Reward stable: `check_numbers` 0.538–0.620 (median 0.582) across all 20 steps — GPU parity
  (GPU reference pure-correctness ≈ 0.44–0.55).

---

## 3. Perf Optimizations

Baseline is g4r19 (our-launcher tp1×dp128, window 64, serial host, **broken checkpoint**).
All step times are clean-step medians. Two eras: **A** (mechanical, broken ckpt, rewards
~0.02) and **B** (re-measured on the fixed checkpoint, rewards ~0.58).

| run id | What each run added incrementally | Median step (s) | rollout | train | weight sync | reward+misc | check_numbers | PR / patch |
|---|---|---:|---:|---:|---:|---:|---:|---|
| g4r19 | Baseline: tp1×dp128, win64, serial host, canonical prompt | 175.6 | 127 | ~15 | ~1.0 | 12 | 0.00 | OOM trio |
| g4r30 | Reference recipe: win512 + 29-flag XLA ladder + async host work | 50.2 | — | — | — | — | 0.02 | #3374, ladder |
| g4r33 | Reference bundle held @ win64 (isolates window) | 51.7 | — | — | — | — | 0.02 | — |
| g4r32/32b | + textual `<end_of_turn>` stop @ win512 | 43.4 / 40.7 | 33.5 | 4.5 | 0.45 | 0.2 | 0.02 | patch_textual_stop |
| g4r35 | − async reward (leave-one-out) | 46.3 | — | — | — | — | 0.02 | patch_reward_parallel |
| g4r32c | + reward-spike fix (per-item sympy timeout) | 40.1 | 33.5 | 4.5 | 0.45 | 0.2 | 0.02 | **patch_verify_timeout** |
| — | **CHECKPOINT FIX → gemma4-e2b-raw2** (§1.2a) | — | — | — | — | — | **0.02→0.58** | g4ckpt-reconv |
| g4r20b (L1) | reward fix on fixed ckpt: gemma-3 prompt + stop ids, win64 | 130.5 | 118 | ~8 | 0.5 | 2.0 | 0.53 | patch_stop_tokens/runner_eos |
| g4r56 | + **rr DP routing** ← the dominant host-work lever | 58.4 | ~50 | 4.0 | 0.45 | 1.9 | 0.54 | **patch_dp_rr_routing** |
| g4r57 | + async detok | 59.8 | ~51 | — | — | — | 0.54 | patch_async_detok |
| g4r38 (L2) | + async logprobs + fast-extract | 56.3 | 48.5 | 4.2 | 0.45 | 1.8 | 0.54 | patch_async_logprobs/fast_extract |
| g4r40 (L4) | + TMBS 8 / HBM 0.3 (null lever) | 56.3 | 48.5 | 4.2 | 0.45 | 1.8 | 0.55 | DL_TMBS/DL_HBM |
| **g4r55 ★** | **+ CD window 512 + XLA ladder (BEST)** | **35.6** | 31.5 | 4.0 | 0.45 | 0.21 | **0.58** | win512 + ladder |
| g4r61 | + async_scheduling (double-buffer, perf-neutral) | 35.6 | 31.0 | 4.0 | 0.45 | 0.20 | 0.58 | DL_ASYNC (verified safe, ~0 gain) |
| **g4r68 ★★** | **BEST config, 20-step confirmation, textual stop retired** | **35.7** | 31.4 | 3.8 | 0.45 | 0.23 | **0.58** | — |

**Attribution (fixed ckpt, win64):** reward/prompt fix −20 s · **rr DP routing −70 s (96%
of the host-work bundle; async detok/logprobs/fast-extract near-null)** · window 512 −21 s ·
TMBS/HBM null · async_scheduling null. The checkpoint fix bought ~4–5 s of perf *for free*
(shorter, naturally-terminating completions) on top of the reward transform.

**Key differences from the Qwen3 line:**
1. **Checkpoint corruption** — gemma4's shared checkpoint was reward-blind; required a
   verified reconversion. Qwen3's mature converter had no such issue.
2. **rr DP routing is the single dominant lever** here (−70 s), where the Qwen line's gains
   were spread across colocation, TP/DP shape, and continue-decode.
3. **async_scheduling** — the Qwen line's projected "~15 s async saving": we measured it
   **perf-neutral** on gemma4 (device-time hidden, host still dominates), and its documented
   generation-degradation was the checkpoint bug, not the scheduler.
4. **Native stop token** — gemma4 stops on id 106 once the checkpoint is fixed; no textual
   stop hack needed.

---

## 4. Next Steps

**Concrete:**
- **Window double-buffering with deferred host-processing** (engine reorder, NOT the
  async_scheduling flag) — the sole remaining high-value lever. Rollout is 88% of the step
  and ~26 s of it is serial-head per-window IPC. Target: rollout 31 → ~9 s ⇒ **global ~13 s**.
- **Batched request submission** — one bulk `add_request` vs 2048 IPCs/window.
- **Prefill rebalance** — prompts are ~155 tokens but prefill reserves 4096; drop to 1024
  → response cap 4096 → 7168 for free (truncation → ~0, reward density up). Measured
  quality lever, +15 s at cap 8192.
- Do **not** invest further in trainer/weight-sync/reward levers or env knobs
  (TMBS/HBM/mnbt) — all measured null at this window.

**Roofline:** ideal ≈ 5.7 s/step (decode 4.4 HBM-bound + trainer 0.8 + sync 0.5); current
best 35.7 s = **6.3× gap, entirely rollout serial-head IPC**. Full analysis:
`GEMMA4_HEADROOM_ANALYSIS.md`.

**Open questions:** production topology (TP4 vs TP1 on healthy weights), response-cap policy
(4096 benchmarking vs 8192 training-quality), multiprocess-DP feasibility under Pathways
(the only lever that touches the roofline itself).

---

## Artifacts

- Deployable manifest: `gemma4/g4r55-best-perf-all-in-one.yaml`
- Ladder detail: `GEMMA4_PERF_LADDER_DETAILED.md`, `OPTIMIZATION_175_to_34.md`
- 20-step best-perf report: `G4R68_20STEP_BEST_PERF.md`
- Reward bug + fix: `GEMMA4_REWARD_BUG_REPORT.md`, `GPU_TPU_REWARD_ALIGNMENT.md`
- Roofline: `GEMMA4_HEADROOM_ANALYSIS.md`
- Patches: `gemma4/patches/` (36 files) · TensorBoard: `gs://ubench-logs/g4r68/tensorboard/`
