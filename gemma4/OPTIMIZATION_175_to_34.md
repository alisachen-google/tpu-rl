# gemma4-E2B: 175.6 s → 34.5 s, step by step

The complete optimization path for gemma4-E2B GRPO on TPU v7x 4×4×4 (128 devices).
Each row stacks on the previous. Perf = TensorBoard/PHASETIME clean-step medians (cold +
contention steps excluded). Full per-step component tables + patch catalog:
`GEMMA4_PERF_LADDER_COMPLETE.md`. Patch files: `patches/`.

## The ladder

| # | Optimization | Step time | Δ |
|---|---|---:|---:|
| 0 | **Baseline** — tp1×dp128, window 64, serial host work | **175.6 s** | — |
| 1 | **Base bundle** — async host work (detok / logprobs / rewards off the critical path), deep-window scaffolding, TMBS8, flash trainer | ~51.7 s | **−124** |
| 2 | **CD window 512** — 8× fewer serial host round-trips per step | ~43–51 s | **−11.0** |
| 3 | **Textual `<end_of_turn>` stop** — retire at the marker instead of self-dialoging to the cap | 43.4 s | **−6.8** |
| 4 | **Async reward** — score during rollout, off the critical path | 40.7 s | **−5.6** |
| 5 | **Reward-spike fix** — per-item sympy timeout + partial-async rescore | 40.1 s | tail 156→0.3 s |
| 6 | **Checkpoint fix** — reconvert embedding from verified HF safetensors; model stops naturally → ~30% shorter completions | **34.5 s** | **−5.5** |

*Per-optimization patch/knob mapping: see `GEMMA4_PERF_LADDER_COMPLETE.md`.*

**Endpoints: 175.6 → 34.5 s = 5.1× wall-clock. And `check_numbers` 0.02 → 0.58 (GPU parity),
because step 6 fixed a corrupted checkpoint — see `GEMMA4_REWARD_BUG_REPORT.md`.**

## Reading the ladder

- **The base bundle is the dominant lever (−124 s).** It is mostly moving host work — detok,
  logprobs, rewards — off the serial critical path, plus deep decode windows. Everything after
  it refines an already host-bound step (rollout ≈ 88% of every clean step).
- **Step 6 is the only lever that improved *both* axes.** It was pursued for correctness
  (rewards 0.02 → 0.58), but a healthy model emits its stop token, so completions are ~30%
  shorter and do less wasted decode → ~4–5 s perf for free.

## Levers that did NOT help (equally informative)

| Lever | Result | Why |
|---|---|---|
| TMBS 8 / HBM 0.3 | null | trainer already ~4 s; nothing to buy |
| Window 512 on our launcher | crashes | launcher-lineage defect; −11 s only via reference stack |
| `async_scheduling` double-buffer | ~neutral (~1 s) | hides device time, but host processing still dominates; needs an engine-core host-pipeline change |
| Prefill rebalance / cap 8192 | +15 s | quality levers (recover truncated tail), not perf levers |
| TP4×DP32 | 88–92 s (worse than TP1's win512 path) | but reverses the old TP1-wins verdict at matched config |

## Where the remaining headroom is

Roofline ≈ 5.7 s/step (decode 4.4 HBM-bound + trainer 0.8 + sync 0.5). The 34.5 s is a **6×
gap**, entirely the serial Python head paying ~3 s/window × 8 windows. Closing it needs
host-parallelism, not kernels: window double-buffering with **deferred host-processing**
(the engine reorder, not the `async_scheduling` flag), batched request submission, and
parallel per-window output processing — targeting ~13 s/step.

**Production config: g4r55** — the full stack above on the fixed checkpoint, TP1×DP128,
window 512.
