# Gemma4-E2B GSM8K: GPU↔TPU accuracy parity of the **24 s performance configuration** (2026-10-29)

Companion to `GEMMA4_E2B_GPU_TPU_PARITY_REPORT.md` (the 35 s parity recipe). Same standard, same
benchmark method, same comparison items; the only change is that the TPU run carries the performance
levers of the optimized ~24 s/step recipe.

## 1. What was run

| item | value |
|---|---|
| run | `g4pt1c2048p` (seed 42), manifest `parity/manifests/g4pt1c2048p-seed42-24s-config.yaml` |
| base | the 35 s GSM8K parity recipe (`g4pt1`): gemma4-E2B-it, GRPO + token TIS (cap 3.0), 1 tpu7x 4x4x4 slice (64 chips), Pathways, pinned image `@sha256:f7c957d0…` |
| 24 s levers applied | response cap 2048 (`max_target_length` 2560 = prefill 512 + 2048, splash blocks 512), async reward scoring (`ASYNC_REWARD=on`), round-robin DP routing, approx-max-k sampling, EOS-check interval 32, `free_kv_cache_during_weight_sync=false`, 128-token KV blocks |
| not applied | the OpenMathInstruct-2 prompt set of the 24 s recipe (parity is measured on the GSM8K training recipe so the curve is comparable to the GPU reference) |
| schedule | 250 updates, eval every 20 updates + final, checkpoint every 20 updates |
| DVFS | updates 0–140 ran under the original manifest (`xla_tpu_dvfs_p_state=7`); updates 141–250 resumed from checkpoint 140 under the corrected manifest (`xla_tpu_dvfs_p_state=6`, the only value allowed on this cluster) |
| admissions | 4 (120-minute Kueue limit per admission; resumed from checkpoints 20/60/140 via `chain-seeds3.sh`); last segment 10:44–12:25 UTC, exit 0 |

## 2. Standard (unchanged)

Scorer-v2 accuracy on the 1319 GSM8K test questions, greedy decode, same scorer on both platforms.
GPU reference: frozen 3-seed curve, step-250 mean 92.42 ± 0.42.
Gate: TPU step-250 ≥ 91.42 (GPU mean − 1 pp) **and** every evaluated point within ±1 pp of the GPU mean.

## 3. Result: **PASS**

| step | GPU mean ± SD | TPU 24 s config | Δ vs GPU mean | TPU 35 s recipe (seed 42, overlay) |
|---:|---|---|---:|---|
| 0 | 80.31 ± 0.24 | 81.27 | +0.96 | 81.50 |
| 20 | 91.03 ± 0.53 | 91.66 | +0.63 | 92.04 |
| 40 | 91.69 ± 1.07 | 91.05 | −0.64 | 91.43 |
| 60 | 91.36 ± 0.23 | 91.43 | +0.07 | 92.19 |
| 80 | 91.84 ± 0.53 | 91.36 | −0.48 | 91.66 |
| 100 | 91.64 ± 0.16 | 91.66 | +0.02 | 91.89 |
| 120 | 92.04 ± 0.55 | 92.57 | +0.53 | 92.12 |
| 140 | 92.67 ± 0.12 | 91.96 (restart eval; 92.34 at 141 before the pause) | −0.71 | 92.57 |
| 160 | 92.77 ± 0.19 | 92.19 | −0.58 | — |
| 180 | 92.52 ± 0.23 | 91.96 | −0.56 | 93.18 |
| 200 | 92.65 ± 0.15 | 91.81 | −0.84 | 92.49 |
| 220 | 92.60 ± 0.24 | 92.12 | −0.48 | 92.27 |
| 240 | 92.65 ± 0.42 | 92.27 | −0.38 | 92.65 |
| **250** | **92.42 ± 0.42** | **92.42** | **0.00** | 92.49 |

* Gate: 92.42 ≥ 91.42 → pass; the step-250 accuracy equals the GPU 3-seed mean to two decimals.
* All 14 points within ±1 pp of the GPU mean (largest |Δ| 0.96 pp at step 0, i.e. before any update).
  Single-run Wilson 95 % half-width is ≈1.5 pp, so none of the deltas is distinguishable from noise.
* Boxed-answer rate ≥ 99.5 % from step 20 on (same as GPU); no-box 0.15–0.45 %.
* Question-level transitions vs step 0 (wrong→right ≈ 160–175, right→wrong ≈ 20–25) match the GPU
  seeds' envelope (panel 4).

### Comparison items (same nine panels as the 35 s report; `parity/figures_24s_config/`)

| panel | 24 s config vs GPU |
|---|---|
| 1 accuracy curve | inside the GPU band at every point; final 92.42 = GPU mean |
| 2 gain vs step 0 | +11.2 pp vs GPU +12.1 pp (TPU starts 1 pp higher at step 0, ends at the same level) |
| 3 boxed rate | 99.5–100 % both |
| 4 transitions | same envelope |
| 5 rollout length | 480 → ~520 tokens (cap 2048), between the 35 s run (~430) and GPU (600–670); GPU's longer rollouts come from its higher response cap, not from different learning |
| 6 step time | median 32.1 s (p-state 6 segment, n = 98 non-eval steps) vs GPU seeds 31.7 / 33.1 / 35.0 and the 35 s TPU recipe 34.6 s |
| 7 TIS weights | max at the 3.0 cap on most steps, mean 1.000, clipped fraction 4×10⁻⁶ |
| 8 sampler↔trainer k3 KL | mean 4.9×10⁻⁴, ESS 0.9991 over updates 141–250; no spike (the 10⁶ spike in the overview at step ~160 belongs to the 35 s seed-42 overlay, documented in that report) |
| 9 training loss | same range and noise as the 35 s run |

Note on step time: this is the GSM8K parity recipe with the 24 s levers, with eval/checkpoint
overheads excluded; the 24.4 s (p-state 7) / 25.3 s (p-state 6) figures of the optimized recipe
itself are measured on OpenMathInstruct-2 prompts (`GEMMA4_E2B_OPTIMIZED_RECIPE_PARITY_REPORT.md`).
Only the final segment's stdout (updates 141–250) was retained, so panels 5–9 show the TPU 24 s
curve from update 141; the evaluation dumps (panels 1–4) cover all 15 evaluations.

## 4. Evidence

* Per-question dumps: `parity/evals_24s_config/eval_{000,021,…,140,141,160,…,250}.jsonl`
  (`gs://ubench-logs/g4pt1c2048p/evals/`).
* Key log lines (evals, checkpoint restores, resume weight push, exit): `parity/logs/seed42_g4pt1c2048p_24s_config_key_lines.log`.
  Resume check: `Restored params from step: 140`, `RESUME-SYNC pushing restored actor weights (global_steps=140)`,
  then `EVAL-BREAKDOWN step=140 … acc=91.96%` against 92.34 % at step 141 before the pause.
* Figures: `parity/figures_24s_config/` (`hires/` at 300 dpi), summary table `parity/figures_24s_config/parity_summary.md`.
* Regenerate: `python3 parity/tools/make_parity_figures.py --gpu <aggregate_summary.json> --tpu "seed42_35s=<35s dumps>:<35s log>" "c2048p_24s=parity/evals_24s_config:<log>" --out figs --dpi 300`.

## 5. Conclusion

The performance levers of the 24 s recipe do not move the GSM8K accuracy curve: the configured run
tracks the GPU reference within ±1 pp at every evaluation and finishes at the GPU mean. Combined with
the reward-band parity of the optimized recipe on its own prompt set, the optimized configuration
meets the same parity standard as the 35 s recipe.
