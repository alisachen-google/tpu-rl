# gemma4-E2B Perf Ladder — detailed component stats (median & mean)

*tpu7x 4×4×4 (128 devices). Per-rung stats over **clean steps (≥3, contention-excluded)**.
Components: **rollout** = engine_generate, **trainer** = actor_train (sum), **weight-sync**
= weight_sync_time, **global** = run_global_step. Sources: PHASETIME JSONs + TensorBoard
`perf/train/*`. Fixed-checkpoint rungs unless noted. Median | mean in seconds.*

## The ladder (global step time)

| Rung | Run | Config | Global median | Global mean | rewards `check_numbers` |
|---|---|---|---|---|---|
| L0 | g4r19 | baseline, old ckpt | 153.0 | 151.9 | 0.00 |
| L1 | g4r20b | + reward/prompt fix, win64 | 130.5 | 130.4 | 0.53 |
| L2 | g4r38 | + host-work (rr/detok/logprobs/extract) | 56.3 | 57.1 | 0.54 |
| L4 | g4r40 | + TMBS8/HBM0.3 | 56.3 | 56.3 | 0.55 |
| **★** | **g4r55** | **champion win512 + fixed ckpt (BEST)** | **35.6** | **35.7** | **0.58** |
| L5 | g4r32c | champion win512, old ckpt | 40.1 | 40.9 | 0.02 |
| D1 | g4r61 | ★ + async_scheduling | 35.6 | 35.9 | 0.58 |

## Per-rung component detail (median | mean, clean steps)

### L0 — g4r19 (baseline, old ckpt, win64)
| component | median | mean |
|---|---:|---:|
| rollout | 127.0 | 128.5 |
| trainer | ~15 | ~15 |
| weight-sync | ~1.0 | ~1.0 |
| rewards | 12.0 | 12.8 |
| **global** | **153.0** | **151.9** |
*rollout ≈ 84% of the step; rewards serial (8.5–17.5 s).*

### L1 — g4r20b (reward fix, win64)
| component | median | mean |
|---|---:|---:|
| rollout | 118.0 | 118.6 |
| trainer | ~5–15 | ~8 |
| weight-sync | ~0.5 | ~0.5 |
| rewards | 2.0 | 2.1 |
| **global** | **130.5** | **130.4** |
*rewards drop 12 → 2 s (parallel); rollout still dominates.*

### L2 — g4r38 (host-work levers, win64)
| component | median | mean |
|---|---:|---:|
| rollout | 48.5 | 49.6 |
| trainer (sum actor_train) | 4.2 | 4.3 |
| weight-sync | 0.45 | 0.55 |
| rewards | 1.8 | 1.9 |
| **global** | **56.3** | **57.1** |
*rollout −70 s vs L1 (rr routing); trainer/sync at floor.*

### ★ g4r55 — BEST (champion win512 + fixed ckpt)
| component | median | mean |
|---|---:|---:|
| rollout | 31.5 | 31.9 |
| trainer | ~4.0 | ~4.2 |
| weight-sync | ~0.45 | ~0.5 |
| rewards | 0.21 | 0.24 |
| **global** | **35.6** | **35.7** |
*Best clean step 34.5. rollout ≈ 88%; trainer/sync/rewards effectively solved.
Per-step (steps 2–9, excl. cold/wobble): 34.5 / 34.4 / 35.7 / 36.0 / 35.6 / 35.4 / 34.1.*

### D1 — g4r61 (★ + async_scheduling)
| component | median | mean |
|---|---:|---:|
| rollout | 31.0 | 31.4 |
| **global** | **35.6** | **35.9** |
*Statistically identical to g4r55 — async_scheduling perf-neutral (device-time hidden,
host still dominates). Accuracy-safe (rewards 0.58).*

## Component summary across the ladder

| Rung | rollout (med) | trainer (med) | weight-sync (med) | rewards (med) | global (med) |
|---|---:|---:|---:|---:|---:|
| L0 | 127.0 | ~15 | ~1.0 | 12.0 | 153.0 |
| L1 | 118.0 | ~8 | 0.5 | 2.0 | 130.5 |
| L2 | 48.5 | 4.2 | 0.45 | 1.8 | 56.3 |
| ★ g4r55 | 31.5 | 4.0 | 0.45 | 0.21 | **35.6** |

**What moved and what didn't:**
- **rollout: 127 → 31.5 s** — carried the entire optimization (rr routing −70, window 512,
  natural-stop shorter completions). It is still 88% of the step.
- **trainer: 15 → 4 s** — TMBS8/flash/fsdp128; then flat (null lever at win64).
- **weight-sync: 1.0 → 0.45 s** — keep-KV + jit-unroll; at floor.
- **rewards: 12 → 0.2 s** — parallel + async + spike-fix; solved.

The remaining 35.6 s vs the ~5.7 s roofline is **all rollout**, i.e. the serial head's
per-window IPC — addressable only by host-pipeline parallelism (the double-buffer engine
reorder), not by any further trainer/sync/reward work.

*A 20-step confirmation run of the best config (g4r68) is in flight; its full per-step
table will replace the 10-step g4r55 numbers above when complete.*
