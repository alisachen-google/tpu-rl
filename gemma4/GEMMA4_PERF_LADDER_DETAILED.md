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

## Host-work bundle decomposition (L1 → L2, single lever per run, win64, fixed ckpt)

The −73 s L1→L2 host-work jump, split one lever at a time. Each run = the previous + one knob.

| Run | Adds | rollout (med) | global median | global mean | Δ global vs prev |
|---|---|---:|---:|---:|---:|
| g4r20b (L1) | (reward fix only) | 118.0 | 130.5 | 130.4 | — |
| **g4r56** | + **rr DP routing** | ~50 | **58.4** | 58.9 | **−72** |
| g4r57 | + async detok | ~51 | 59.8 | 60.0 | +1.4 (noise) |
| g4r38 (L2) | + async logprobs + fast-extract | 48.5 | 56.3 | 57.1 | −3 (noise) |

**Finding: rr DP routing alone = ~96% of the host-work win (130.5 → 58.4 s).** It removes a
per-request all-rank state query through the Pathways proxy (2048 IPCs per decode window);
async detok / logprobs / fast-extract are near-null on top of it. The step is
serial-head-IPC-bound, not detok/logprob-bound. Rewards 0.53–0.59 (parity) across all four.
Clean steps used (contention-excluded): g4r56 = 58.4/59.4, g4r57 = 59.8/60.2, g4r38 = 56.3/58.1.

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

### ★★ g4r68 — BEST config, 20-STEP CONFIRMATION (champion win512 + fixed ckpt, textual stop retired)

Full 20-step run. Clean stats (n=15, contention step 6 excluded): **global median 35.7,
mean 35.8, min 34.1, max 37.4, std 0.77 s** — exceptionally stable. `check_numbers` median
0.582, mean 0.579 (0.538–0.620) across all 20 steps.

| component | median | mean |
|---|---:|---:|
| rollout | 31.4 | 31.4 |
| trainer | 3.7 | 3.8 |
| weight-sync | ~0.45 | ~0.5 |
| rewards | 0.23 | 0.24 |
| **global** | **35.7** | **35.8** |

Per-step global (steps 4–19): 35.7 / 35.2 / 48.5(wobble) / 36.3 / 35.7 / 35.5 / 35.8 /
35.7 / 36.7 / 34.1 / 37.4 / 34.9 / 36.5 / 35.5 / 35.6 / 36.6. Best step 34.1. One wobble
(step 6, 48.5). This is the authoritative best-config measurement (supersedes the 10-step
g4r55 numbers): **35.7 s median, rewards 0.58, over 20 steps.**

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
| L1+rr (g4r56) | ~50 | ~4 | 0.45 | 1.9 | 58.4 |
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
