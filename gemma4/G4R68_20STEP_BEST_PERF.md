# g4r68 — Best-Config 20-Step Perf Report

**The authoritative measurement of the production configuration.** gemma4-E2B GRPO,
TP1×DP128 on tpu7x 4×4×4 (128 devices), CD window 512, fixed checkpoint
(`gemma4-e2b-raw2`), textual stop retired (native id-106 stop). 20 steps, exit 0.
Manifest: `g4r55-best-perf-all-in-one.yaml` with `num_batches=20`.

## Full per-step table

| step | rollout | trainer | weight-sync | rewards | **global** | note |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 96.3 | 688.7 | — | 58.07 | **845.6** | cold (compile + restore + sync sympy) |
| 1 | 34.4 | 8.5 | ~0.5 | 2.33 | **45.6** | warmup |
| 2 | 30.1 | 3.7 | ~0.45 | 0.20 | **34.3** | clean |
| 3 | 30.8 | 3.7 | ~0.45 | 0.22 | **35.1** | clean |
| 4 | 31.4 | 3.7 | ~0.45 | 0.22 | **35.7** | clean |
| 5 | 30.6 | 4.1 | ~0.45 | 0.21 | **35.3** | clean |
| 6 | 44.2 | 3.7 | ~0.45 | 0.24 | **48.5** | rollout wobble |
| 7 | 31.7 | 3.8 | ~0.45 | 0.37 | **36.3** | clean |
| 8 | 31.3 | 3.8 | ~0.45 | 0.25 | **35.7** | clean |
| 9 | 30.7 | 4.2 | ~0.45 | 0.21 | **35.5** | clean |
| 10 | 31.4 | 3.8 | ~0.45 | 0.27 | **35.8** | clean |
| 11 | 31.6 | 3.6 | ~0.45 | 0.22 | **35.7** | clean |
| 12 | 32.5 | 3.7 | ~0.45 | 0.23 | **36.7** | clean |
| 13 | 29.8 | 3.7 | ~0.45 | 0.21 | **34.1** | clean (best) |
| 14 | 33.2 | 3.7 | ~0.45 | 0.23 | **37.4** | clean |
| 15 | 30.6 | 3.7 | ~0.45 | 0.25 | **34.9** | clean |
| 16 | 31.1 | 4.8 | ~0.45 | 0.21 | **36.4** | clean |
| 17 | 31.2 | 3.7 | ~0.45 | 0.23 | **35.5** | clean |
| 18 | 31.5 | 3.6 | ~0.45 | 0.23 | **35.6** | clean |
| 19 | 32.5 | 3.6 | ~0.45 | 0.23 | **36.6** | clean |

Columns: rollout = `engine_generate`, trainer = `micro_batch_steps − generate_and_advantage`,
rewards = `compute_rewards`, global = `run_global_step` (seconds). Source: PHASETIME JSONs
at `gs://ubench-logs/g4r68/phasetime/`.

## Summary statistics (clean steps 2–19, contention step 6 excluded, n=16)

| component | median | mean | min | max | std |
|---|---:|---:|---:|---:|---:|
| **global** | **35.7** | **35.7** | 34.1 | 37.4 | 0.78 |
| rollout | 31.3 | 31.4 | 29.8 | 33.2 | 0.86 |
| trainer | 3.7 | 3.8 | 3.6 | 4.8 | 0.30 |
| weight-sync | 0.45 | 0.50 | — | — | — |
| rewards | 0.23 | 0.24 | 0.20 | 0.37 | 0.04 |

**Reward quality (TensorBoard `check_numbers`, all 20 steps):** median **0.582**, mean
0.579, range 0.538–0.620 — GPU parity (GPU reference: pure-correctness ≈ 0.44–0.55), no
drift across the run.

## Reading it

- **35.7 s median, std 0.78 s** — exceptionally stable across a full 20-step horizon; one
  wobble (step 6). Best clean step 34.1 s.
- **Component split: rollout 88% · trainer 11% · sync/rewards ~2%.** Trainer, weight-sync,
  and rewards are all at their floor — the entire remaining gap to the ~5.7 s roofline is
  rollout (serial-head per-window IPC). See `GEMMA4_HEADROOM_ANALYSIS.md`.
- **Cold step 0 (845 s)** is compile + checkpoint restore + a one-time synchronous sympy
  pass; amortized to zero over a real training run.
- This supersedes the earlier 10-step g4r55 numbers as the authoritative best-config result,
  and confirms the textual-stop removal (this run has it off) caused no regression.

**Campaign endpoints: 175.6 → 35.7 s (4.9×) · rewards 0.02 → 0.58 (GPU parity).**
