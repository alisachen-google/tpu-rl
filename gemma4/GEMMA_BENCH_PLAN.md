# Benchmarking Plan — Gemma-4-2B DAPO/GRPO on TPU v7x (Ironwood)

**Status:** PLAN (pre-run) · **Base methodology:** tianyu-rl-05 (Qwen3-0.6B, 274.9 s median
vs Magellan 299 s) + our smoke ladder · **Image:** same `2026-08-03b` digest
`sha256:c96e3219a5…` (headline runs; one optional rung needs a derived image — see §3).

## 1. Objective

Extend the validated Qwen3-0.6B DAPO/GRPO pipeline to **gemma-4-2B** on `tpu7x:4x4x4`
(16 hosts × 8 cores), producing: (a) a STRICT-BF16 steady-state step time with full phase
breakdown (rollout / train / sync / residual), measured to the tianyu bar (median over
**n≥5 warm steps**, step 1 excluded); (b) artifacts comparable against the **GB200 sweep
convention** (per-device train batch frozen, sequences-per-accelerator as the swept lever).

## 2. What carries over unchanged

- **Image + PR stack**: `2026-08-03b` already contains #2507, #3002, #3101, #3102, #3117,
  #3156, #3174 (merged set) and #3100, #3191 (open, via fork `be9cdf04`) — independently
  re-verified by the tianyu guide by source-line extraction from the shipped image.
- **Manifest skeleton**: tianyu-rl-05.yaml (namespace `priority-dev`, `multislice-queue`,
  `maxRestarts: 20`, embedded-ConfigMap patches, XLA flags on the proxy argv,
  `instance_type=tpu7x:4x4x4` bare, TPU_SKIP_MDS_QUERY, no explicit ACCELERATOR_TYPE).
- **Runtime patches** (embedded ConfigMaps): `patch_cd_proper.py` (PR #3374 — required for
  continue-decode on flax_nnx), `patch_vllm_sampler.py` (patches 4+5), `patch_lendist.py`
  (ISL/OSL instrumentation — MORE important for gemma, see §5 gates).
- **The four validity gates** + reward-collapse watch (§7).
- **Kueue-deactivation budgeting**: runs die externally (`active=false`) — front-load the
  steps you need; watch pod count.

## 3. Requirement (i): the PR / reference matrix

| Item | Type | In `2026-08-03b`? | Action for gemma runs |
|---|---|---|---|
| #2507 + #3117 (continue decode + async) | code | ✅ (incl. d449d04 correctness fix) | use; `enable_continue_decode:true, max_decode_steps:128` |
| #3102 RPA block-size env | code | ✅ | lever: `RPA_V3_DECODE_BLOCK_SIZES` — re-tune for gemma (head_size 256 doubles KV row width; Qwen3 values are NOT presumed transferable) |
| #3100 replay-exact sampler | code (open PR, via fork) | ✅ env-gated | OFF for headline; optional determinism rung |
| #3101 mutable-metadata fix | code | ✅ | inherited |
| #3174 EOS-check interval | code | ✅ env-gated | rung variable (see ladder R3) |
| #3191 KV persist across sync | code (open PR, via fork) | ✅ env-gated | OFF for headline (mirrors tianyu); optional rung if rollout proves KV-alloc-bound |
| **#3165 DAPO/GRPO TPU recipe** | **recipe only** (examples/dapo_tpu: README, config_template, launch.sh — no runtime code) | n/a | **adopt its knobs**: grpo_beta=0, epsilon_high=0.28, CD ladder; note its `GRR_*` env names are internal-prefix leaks — real names are `RPA_V3_DECODE_BLOCK_SIZES` / `CONTINUE_DECODE_EOS_CHECK_INTERVAL` |
| **maxtext #4646 fp8 KV cache** | code, **OPEN, in no image** | ❌ | **excluded from headline** (breaks STRICT-BF16; Magellan: "for the reported throughput → bf16; fp8 KV is an optional capacity knob"). Gemma re-opens the question — 262k vocab & head 256 grow KV 2× per token vs Qwen3 — so include as **optional rung R5 on a derived image** `2026-08-03b-fp8kv` (cherry-pick #4646, new tag, full T0c+T6 gate) **only if** R1 measures rollout as KV-capacity-bound. Do the arithmetic first (tianyu's method): `total_hbm_avail_gb` vs `layers × kv_heads × head_dim × 2 × 2B × max_len × max_num_seqs`. |
| Docs (Ironwood/tpu7x, Ironwood stack blog, MaxText, Tunix, Meta infra ×4) | references | n/a | cite in the report's context section |

## 4. Requirement (ii): the GB200 convention, mapped to TPU

The GB200 gemma-4-2B sweep freezes **per_device_train_batch_size (pd) = 2** — not swept —
because gemma-4-2B's `head_size=256` + **262k-token vocab** make the training-side logits
tensor ~4× larger per sequence than Qwen3-0.6B (logits = batch × seq × 262144 × 2 B), and
pd>2 OOMs on GB200's 186 GB HBM. Their only swept lever is **SPG = 512/W** (sequences per
GPU, W = world size). For comparability we:

1. **Report the TPU analog explicitly**: per-device train micro-batch = `train_micro_batch_size
   × seq_positions / trainer_devices` and sequences-per-chip at each rung — so a reader can
   line our table against the GB200 sweep without unit archaeology.
2. **Expect the same wall to bind us**: the tianyu ROOT CAUSE (tmbs=32 → 108 G > 94.7 G HBO
   temporaries at 0.6B/151k vocab) scales ~4× worse here. Plan numbers (§6) start tmbs=4
   with an OOM-scout, not at Qwen3's 16.
3. **Freeze what they froze**: batch geometry per device is fixed once the scout finds the
   ceiling; sweep only total batch via data-parallel width — mirroring pd-frozen/SPG-swept.

## 5. Gemma-specific deltas and risks (each with a planned probe)

| Delta vs Qwen3-0.6B | Consequence | Probe |
|---|---|---|
| 2B params (~4 GB bf16) vs 0.6B | trainer memory & FSDP all-gathers ×3.3; sampler weight-push ×3.3 | T6-mini timing; `weight_sync` phase |
| vocab 151k → 262k | logits+loss memory ~×1.7/token; **sampling compute** in decode grows | train OOM-scout; rollout per-token time vs Qwen3 |
| head_size 128 → 256 (KV row ×2/head) | KV cache per token ×~2 (modulo kv_heads count — read from config at T0c) | recompute tianyu's KV headroom arithmetic before touching #3191/#4646 |
| model code path: `gemma4.py` not `qwen3.py` | our RLConfig/multimodal patch was proven for qwen3; gemma may read *different* missing fields — same bug class | **T0c-gemma**: instantiate the gemma model class against `types.RLConfig` on CPU; fix schema gaps BEFORE cluster time |
| tokenizer/dataset fit | prompt lengths differ → truncation risk against `max_prefill_predict_length` | LENDIST gate: `max(isl) < cap` per step |
| checkpoint | **PREREQUISITE, currently unknown**: need a pathways-compat MaxText gemma-4-2B checkpoint (raw TensorStore, `use_ocdbt=False`) | §9 ask #1 — blocker until answered |

## 6. The run ladder

**P0 — prerequisites (no TPU):** checkpoint path confirmed & readable (T3); **T0c-gemma**
on CPU: model-class instantiation vs RLConfig, tokenizer load, config-schema dry parse of
the full override string; PR-presence audit unchanged (image is the same digest).

**P1 — OOM scout (cheapest possible TPU probe):** 1 step, `num_batches=1`, candidate
`train_micro_batch_size ∈ {2, 4, 8}` descending from predicted ceiling; watch the
`Compiled train_step` line and RESOURCE_EXHAUSTED at ~40 s post-rollout (the deterministic
tianyu signature). Pick the largest tmbs that fits with ≥30% headroom, preferring exact
divisors of batch_size (no ragged second compile shape).

**P2 — T6-mini-gemma (2 steps, 2x2x1 if a small checkpoint variant fits, else 4x4x4):**
full loop validation, pearson gate >0.8, LENDIST sanity, reward > 0.

**P3 — headline runs (STRICT-BF16, 4x4x4, n≥5 warm steps × 2 repeats):**

| Rung | Config delta | Purpose |
|---|---|---|
| R0 | CD off, EOS interval 1, no RPA override | control |
| R1 | + continue-decode (`max_decode_steps=128`) | the #2507/#3117 rung |
| R2 | + RPA_V3_DECODE_BLOCK_SIZES (gemma-tuned; start from Qwen3's `1,16384,1,4096` then probe) | #3102 rung |
| R3 | + `CONTINUE_DECODE_EOS_CHECK_INTERVAL=8` | #3174 rung — tianyu warning applies: this REGRESSED 3× at their old config; if R3 ≫ R2, drop it and report R2 |
| R4 (optional) | + `KV_CACHE_PERSIST_ACROSS_WEIGHT_SYNC=1` | #3191 — only if weight-sync/KV-realloc shows in the phase split |
| R5 (optional, derived image) | fp8 KV (#4646 cherry-pick) | capacity knob — separate tag, separate table, NEVER mixed into the bf16 headline |

Baseline config (R0 draft) = tianyu-rl-05 launcher with:
`model_name=gemma-4-2b` (verify exact maxtext name), tokenizer per checkpoint,
`batch_size=256, train_micro_batch_size=<scout>, rollout_micro_batch_size=256,
max_num_seqs=256, rl.num_generations=8, rl.grpo_beta=0.0, rl.epsilon_high=0.28,
max_target_length=16384, max_prefill_predict_length=8192 (revisit after first LENDIST),
max_num_batched_tokens=max_target_length+kv_cache_buffer, num_batches=20, debug=True,
profiler=xplane (skip 1, profile 1)` — new run names per attempt (≤22 chars), fresh
`gcs_scratch_location` per §2 hygiene rules, ARTIFACT_DIR = `gs://ubench-logs/alisachen-tpu7x/gemma`.

**P4 — report** (§8 format).

## 7. Gates (all four + one, checked before believing any number)

1. **Patches applied**: `CD-PROPER PATCH: APPLIED` + `LENDIST PATCH: APPLIED` in head log
   (+ our T0c-gemma schema check passed at P0).
2. **Single-pool placement**: 16 workers in exactly one gke-nodepool (capture the mapping
   WHILE ALIVE).
3. **No prompt truncation**: `max(isl) < max_prefill_predict_length` in every lendist step
   file (gemma tokenizer may inflate ISLs — this gate is the reason `8192` is provisional).
4. **Steps completed**: `perf/train/global_step_time` scalars exist.
5. **No reward collapse**: reward trending > 0 with grpo_beta=0 + epsilon_high=0.28 (value
   itself is not comparable to Qwen3's 0.145 — different model/task difficulty).

## 8. Reporting format

Per rung: step table (global/rollout/train/sync + residual named), median over warm steps,
tmbs & per-device geometry (§4.1), LENDIST summary (isl_max, osl_mean, %-at-cap), reward
trace, gates checklist, and the deviations-from-recipe table (tianyu style: every knob that
differs from #3165's template, with a reason). References section cites: Google tpu7x docs,
Ironwood stack blog, MaxText, Tunix, the 9 tpu-inference PRs + #3165, maxtext #4646, and
the four Meta infra posts (context for the GB200 comparison).

## 9. Open questions — ask before burning TPU time

1. **Gemma-4-2B checkpoint**: which GCS path (pathways-compat, raw TensorStore)? If none
   exists, converting one is a work item that precedes P1.
2. **Dataset**: keep `nvidia/OpenMathInstruct-2` for internal comparability, or
   DAPO-Math-17k for recipe fidelity? (Rollout is ~64% of step and OSL-driven — the single
   biggest comparability variable; tianyu left it open for Qwen3 too.)
3. **remat_policy**: rl.yml forces 'custom' — same question tianyu escalated; for a 2B
   model the answer moves the train phase materially more.
4. **GB200 numbers to compare against**: which published gemma-4-2B step time / SPG values
   is this benchmarked against? (Requirement (ii) implies a specific sweep exists.)
5. Sampling params for gemma (temp/top-k/top-p) — copy Qwen3's 0.8/50/0.95 or recipe values?

## 10. Budget & scheduling realism

Per attempt at 4x4x4: ~5 min admission + ~10 min init (warm cache) + ~20-step budget where
**external Kueue deactivation is the expected end** — plan every run to deliver its metric
by step ~8. OOM-scout attempts are ~15 min each. Estimated total for P0–P3 with 2 repeats
and one re-tune loop: 1.5–2 cluster-days of intermittent 4x4x4 occupancy. Capacity check
(`kubectl get nodes -l cloud.google.com/gke-tpu-topology=4x4x4`) before every submit —
NAP creates/deletes pools.
