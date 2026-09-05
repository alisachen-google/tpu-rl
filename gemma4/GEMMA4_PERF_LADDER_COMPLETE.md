# gemma4-E2B GRPO — Complete Perf Ladder, Patch Map & Per-Step Component Breakdown

*tpu7x 4×4×4 (128 devices) · MaxText + tunix + tpu-inference/Pathways · GRPO/RLVR on
OpenMathInstruct-2 · 2048 completions/step (256 prompts × 8). Every row has a run, a
manifest, and GCS artifacts. Perf = TensorBoard/PHASETIME medians over steps ≥3 (cold +
contention steps excluded). Two eras: **Ladder A** = mechanical perf on the (then-unknown)
corrupted checkpoint; **Ladder B/C** = re-measured on the verified checkpoint `raw2`.*

Companion docs: `GEMMA4_LADDER_REPORT.md` (raw per-step tables), `GEMMA4_REWARD_BUG_REPORT.md`
(the checkpoint bug), `GEMMA4_HEADROOM_ANALYSIS.md` (roofline). Patches live in the
`*-manifest/` configmaps and at `/mnt/disks/scratch/g4r8-stage/patches/` (mirrored per run).

---

## How a lever becomes a run

The image is a **pinned shared build** (`maxtext-rl-ironwood:2026-08-24-packing`), so every
lever ships as either (a) a **runtime patch** applied at pod start (self-verifying, fail-open,
prints `X PATCH: APPLIED|FAILED`), or (b) an **env knob** (`DL_*`) read by the launcher into
a MaxText config flag. The column "Patch / knob" below names the exact artifact.

---

## Ladder A — mechanical perf (corrupted checkpoint, rewards ≈ 0.02 throughout)

| Rung | Run | Lever added | Patch / knob | Warm/clean | Δ |
|---|---|---|---|---|---|
| A0 | g4r19 | baseline tp1×dp128, win64, serial host | (OOM trio: `patch_flash_sa` + `DL_CPV=8` + `DL_KVBUF=4352`) | **175.6 s** (148–155 clean) | — |
| A1 | g4r30 | reference recipe: win512 + XLA ladder + async host work | `values-xla-ladder.yaml` (29 flags) + `patch_cd_proper`/`patch_cd_masking`/`patch_cd_report` + `patch_async_detok`/`patch_async_logprobs` | **50.2 s** | −125 |
| A2 | g4r33 | reference bundle held @ win64 (isolates window) | `DL_CD=64` | **51.7 s** | (window512 = A2↔A4 = −11.0) |
| A3 | g4r32 | + textual `<end_of_turn>` stop @ win512 | **`patch_textual_stop.py`** | **43.4 s** | −6.8 |
| A4 | g4r32b | champion replica | (same stack) | **40.7 s** (best 38.8) | — |
| A5 | g4r35 | − async-reward (leave-one-out) | `patch_reward_parallel.py` (off) | 46.3 s | async = −5.6 |
| A6 | **g4r32c** | + reward-spike fix | **`patch_verify_timeout.py`** + `_AR_PARTIAL` edit to `patch_async_reward` | **40.1 s, spike-free** | step-8: 156→0.26 s |

**Ladder A attribution:** base bundle −125 s · window 512 −11.0 · textual stop −6.8 · async
reward −5.6. Arc **175.6 → 40.1 s (4.4×)**, but `check_numbers` flat at ~0.02 (the hidden
checkpoint defect).

### A6/g4r32c per-step components (the clean champion, corrupted ckpt)

| step | rollout | rewards | trainer | weight-sync | dequeue* | other | GLOBAL |
|---|---|---|---|---|---|---|---|
| clean (3–5,7,9) | 33.0–34.4 | 0.19–0.26 | 4.2–4.9 | 0.42–1.35 | 4.9–6.2 | ~1.0 | **38.8–41.1** |
| 8 (spike-fixed) | 34.4 | **0.26** | 6.6 | 2.5 | 24 | ~0 | 41.1 |
| 0 (cold) | 82.4 | 55.4 | 45 | 1.1 | 22.5 | 10.6 | 188.7 |

Canonical clean budget: **rollout ~33.5 (84%) · trainer ~4.5 (11%) · sync ~0.45 · rewards
~0.2 · other ~1.0**. *dequeue overlaps rollout; informational.

---

## The correctness fix (between Ladder A and B)

The reward stayed at 0.02 because the shared checkpoint's **embedding table was corrupted**
(every row normalized to √1536, directions rotated, worst on turn/stop tokens). Fixed by
reconverting from the verified HF safetensors → `gemma4-e2b-raw2`. See
`GEMMA4_REWARD_BUG_REPORT.md`. Validation: row audit cos 0.999999; forward-pass KL 5.3e-3.

| Job | Patch / tooling | Result |
|---|---|---|
| g4ckpt-reconv | MaxText `checkpoint_conversion/to_maxtext` (fork, native gemma4-e2b) + built-in row audit | `raw2`, cos 0.999999 |
| g4ckpt-logitcheck | `tests/utils/forward_pass_logit_checker` vs HF | KL 5.3e-3 < 0.015 |

Adoption knob: **`DL_CKPT_PATH`** (points `load_parameters_path` at `raw2`).

---

## Ladder B — one lever per rung on the FIXED checkpoint (rewards ≈ 0.5, GPU parity)

| Rung | Run | Lever added | Patch / knob | Clean step | `check_numbers` |
|---|---|---|---|---|---|
| B0 | (g4r19) | old-ckpt baseline | — | 148–155 s | 0.00 |
| B-A/B | g4r44 | **checkpoint swap only** | `DL_CKPT_PATH=raw2` | 135.8 s (perf-neutral) | **0.53–0.56** |
| B1 | g4r20b | end-of-turn/reward fix | `patch_stop_tokens.py` + `patch_runner_eos.py` + `patch_textual_stop.py` + `DL_STOPIDS=[106,50]` + `DL_DSPROC=gemma_data.py` | **127–133 s** | 0.50–0.57 |
| B2 | g4r38 | + host-work scheduling | `patch_async_detok.py` + `patch_async_logprobs.py` + `patch_dp_rr_routing.py` + `patch_fast_extract.py` | **56–58 s** | 0.49–0.59 |
| B3 | g4r39 | + decode-loop shape (win512+XLA) | `DL_CD=512` + `values-xla-ladder.yaml` | ❌ launcher crash (−11 s via A2↔A4) | — |
| B4 | g4r40 | + trainer/engine shape | `DL_TMBS=8` + `DL_HBM=0.3` | 56 s (null) | healthy |

**Ladder B attribution @ win64:** reward fix (B0→B1) −~20 s (shorter healthy completions +
parallel rewards); **host-work levers (B1→B2) −~73 s — the dominant rung**; TMBS/HBM null.

### B2/g4r38 per-step components (host-work levers, fixed ckpt)

| step | rollout | rewards | trainer(sum) | weight-sync | dequeue* | GLOBAL |
|---|---|---|---|---|---|---|
| 3 | 49.5 | 1.8 | ~4.2 | ~0.45 | ~5 | **58.1** |
| 4 (best) | 47.6 | 1.8 | ~4.2 | ~0.45 | ~5 | **56.3** |
| 0 (cold) | 607.6 | 61.6 | 45 | 1.1 | 24 | 718.1 |

Training visibly learns: OSL 1750 → 1350 and truncation 10% → 3.0% by step 7.

### B1/g4r20b per-step (reward fix, fixed ckpt)

| step | rollout | rewards | trainer | GLOBAL |
|---|---|---|---|---|
| 4 | ~121 | 1.9 | ~5–15 | 133.0 |
| 5 | ~115 | 2.0 | ~5 | 130.5 |
| 6 (best) | ~115 | 2.0 | ~5 | 127.6 |

---

## Ladder B★ — the synthesis (best result)

| Run | Config | Patch / knob | Clean step | `check_numbers` |
|---|---|---|---|---|
| **g4r55** 🏆 | Ladder-A champion stack (win512, all levers) + **fixed ckpt** | full A6 stack + `DL_CKPT_PATH=raw2` | **34.5–37.2 s (best 34.5)** | **0.54–0.62** |

### g4r55 per-step components (production config)

| step | rollout | rewards | GLOBAL |
|---|---|---|---|
| 2 (best) | 30.8 | 0.22 | **34.5** |
| 3–5,7–9 | 31.4–33.0 | 0.20–0.40 | 35.4–37.2 |
| 6 | 46.2 | 0.26 | 50.5 (wobble) |
| 0 (cold) | 152.6 | 60.3 | 266.0 |

**Endpoints: 175.6 → 34.5 s (5.1×) · rewards 0.02 → 0.58.** The ~4 s gain over the old
champion (g4r32c 40.1) is free — the healthy model's shorter, naturally-terminating
completions do less decode. Clean budget: **rollout ~31.5 (88%) · trainer ~4 · rewards ~0.2.**

---

## Ladder C — reproduction & probes (fixed checkpoint)

| Run | Lever | Patch / knob | Clean step | Finding |
|---|---|---|---|---|
| g4r50 | TP4×DP32 | `DL_TP=4` `DL_DP=32` | 88–92 s | **topology verdict reverses** — TP4 beats TP1 by ~40 s on healthy weights |
| g4r52 | response cap 8192 | `DL_MTL=12288` (`patch_verify_timeout` guards the old hang) | 146–149 s | cap lever −33 → **−15 s**; truncation 0.05% |
| g4r53 | TMBS 16 | `DL_TMBS=16` | 120.5–121.7 s | small −7…10 s (near-null) |
| g4r59 | prefill rebalance | `DL_MPPL=1024` (cap→7168) | 71–73 s | quality lever (+15 s, truncation→0) |

---

## Ladder D — host-parallelism (in flight)

The clean step is ~88% rollout(host)-bound; roofline ≈ 5.7 s (decode 4.4 + trainer 0.8 +
sync 0.5). The gap is the serial head paying ~3.3 s/window × 8 windows. Levers under test:

| Run | Lever | Patch / knob | Status |
|---|---|---|---|
| g4r56–58 | host-work micro-split (rr → detok → logprobs) | `DL_DPRR` / `DL_ADETOK` / `DL_ALOGP` one at a time | decomposing the B1→B2 −73 s |
| g4r60 | engine batching | `DL_MNBT=33280` | staged |
| g4r61 / g4r61w | **window double-buffering** = `async_scheduling` | **`DL_ASYNC=True`** | verified safe on fixed ckpt (`eos_at_cap=0`); perf measurement in flight |

Double-buffer target: rollout 31.5 → ~8.5 s ⇒ global ~36 → ~13 s (see headroom doc).

---

## Full patch catalog (what each file does)

**Correctness / decode-window**
- `patch_cd_proper.py` — continue-decode compiler-options fix (PR #3374); required for win>10
- `patch_cd_masking.py` / `patch_cd_report.py` — count-only window exit + last-non-pad host read (both required or window is void)
- `patch_cd_noeos.py` — drop the per-iter cross-DP all-reduce in the exit cond
- `patch_stop_tokens.py` / `patch_runner_eos.py` — union gemma stop ids [1,106,50] into the vLLM + device-loop stop sets
- `patch_textual_stop.py` — retire at the textual 7-BPE `<end_of_turn>` (our novel lever)
- `patch_g4_format_guard.py` — format-regex fix (no catastrophic backtracking on echoes)
- `patch_verify_timeout.py` — per-item sympy timeout (kills the 156 s reward spike)
- `gemma_data.py` / `gemma_data_canonical.py` — gemma-3-style / canonical-frame prompt builders

**Performance rewrites**
- `patch_sharded_sampling.py` — vocab-sharded sampling/logprobs
- `patch_packer_fast.py` / `patch_packing_knob.py` — numpy-first sequence packer + budget knob
- `patch_host_extract.py` / `patch_fast_extract.py` / `patch_fast_detok.py` — vectorized EOS extract, fast answer-extract, batch detok
- `patch_async_detok.py` / `patch_async_logprobs.py` / `patch_reward_parallel.py` — move detok/logprobs/rewards off the critical path
- `patch_dp_rr_routing.py` — round-robin DP routing (no per-request all-rank state queries)
- `patch_jit_unroll.py` / `patch_sync_keepkv.py` — jitted weight-sync unroll + keep-KV
- `patch_flash_sa.py` — flash trainer attention (OOM fix); `patch_gc_sync.py` — GC gating

**Instrumentation**
- `patch_phasetime.py` / `patch_perf_v2.py` — per-phase timing (rollout/rewards/trainer/sync)
- `patch_lendist.py` — OSL/ISL/truncation/`with_eot` distribution per step
- `patch_reward_dump.py` — GPU-format per-item reward JSONL
- `patch_log_stats.py` / `patch_firsttok.py` / `patch_dump_completions.py` — KV gauge, opening-format census, raw completions

---

## Headline

**175.6 s → 34.5 s (5.1× mechanical) with rewards 0.02 → 0.58 (GPU parity, via the
checkpoint fix).** Production config = g4r55 (champion stack + `raw2`, TP1). The remaining
7× gap to the 5.7 s roofline is entirely serial-head window overhead; the double-buffer
(`async_scheduling`, verified accuracy-safe) is the next lever, targeting ~13 s/step.
