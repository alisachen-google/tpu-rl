# The Optimization Ladder — per-rung detailed results

One lever per run, our launcher lineage, gemma4-E2B GRPO on tpu7x 4×4×4 (128 devices),
8 steps per run. Each rung's table: per-step global time + phase split (rollout = engine
generate, rewards = compute_rewards, trainer = actor train, sync = weight sync; sources:
PHASETIME JSONs + TensorBoard `perf/train/*`). Plan: `GEMMA4_FORWARD_PLAN.md`.
Chain control: `ladder-chain.sh` (each rung launches only after the previous completes).

## Rung status board

| Rung | Run | Lever added | Warm median | Status |
|---|---|---|---|---|
| L0 | g4r19 | — (baseline) | **175.6 s** | ✅ measured |
| L1 | g4r20b | end-of-turn/reward fix | — | queued |
| L2 | g4r38 | host-work scheduling | — | queued |
| L3 | g4r39 | decode-loop shape (cd512 + XLA ladder) | — | staged (auto) |
| L4 | g4r40 | trainer/engine shape (TMBS8, HBM0.3) | — | staged (auto) |
| L5 | g4r32c | champion + spike fix (target) | **~40.1 s, spike-free** | ✅ done |

---

## L0 — g4r19: the baseline (measured 2026-09-01)

Config: canonical `<|turn>` prompt, CD window 64, serial host work, TMBS 4, HBM 0.25,
flash trainer, chips_per_vm 8, kv_cache_buffer 4352. Manifest: `gemma4/g4r19-all-in-one.yaml`.

| step | global | note |
|---|---|---|
| 1 | 367.4 | warmup |
| 2 | 175.6 | warm |
| 3 | 153.0 | clean |
| 4 | 155.2 | clean |
| 5 | 148.5 | clean (best) |
| 6 | 427.0 | reward spike (serial sympy, 127.6 s rewards) |
| 7 | 350.2 | rollout wave |

**Warm median 175.6 s; clean band 148–155 s.** Phase split on clean steps: rollout
125–130 s (~85%), rewards 8.5–17.5 s (serial), trainer ~15 s (TMBS 4). Task health:
`check_numbers` = 0.0 every step (untrained `<|turn>` markers — the defect L1 fixes),
truncation 64→78% rising.

## L1 — g4r20b: + end-of-turn/reward fix

*(pending — table filled on completion; run is fighting cluster infra crash-loops, 20 restarts)*

## L1-probe — g4r41: canonical `<|turn>` prompt (GPU-parity) ✅ COMPLETED 2026-09-02 20:31Z

g4r20b config + `gemma_data_canonical.py` (byte-exact GPU prompt render, ids 105/106
verified single-token by the in-run `GEMMA-CANON-IDS` banner). 8 steps, exit 0.

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 68.7 | 745.8 | 1909.0 | cold (fresh compile + cluster turbulence) |
| 1 | 3.0 | 120.2 | 138.7 | clean |
| 2 | 3.1 | 256.5 | 268.1 | infra wobble (worker crash-loops cluster-wide) |
| 3 | 4.9 | 120.2 | 134.2 | clean |
| 4 | 3.1 | 120.8 | 133.0 | clean (best) |
| 5 | 3.2 | 128.6 | 140.6 | clean |
| 6 | 3.4 | 285.3 | 297.2 | infra wobble |
| 7 | 3.9 | 282.2 | 295.0 | infra wobble |

**Perf:** clean band ~133–141 s at window 64 — ~10% under g4r19's 148–155 clean band
(shorter OSL 1650 vs ~2000+, parallel rewards 3–5 s vs 8.5–17.5).
**Quality: unchanged — the decisive negative result.** 0/33 dump samples correct,
truncation 35–37%, textual `<end_of_turn>` in ~63–65% of completions, no id-106 stops —
despite the trained-frame prompt provably reaching the model. This ruled out the prompt
contract and triggered the byte-level weights audit that found the corrupted embedding
table (see `GPU_TPU_REWARD_ALIGNMENT.md`, "ROOT CAUSE FOUND"). Quality rungs are capped
by the checkpoint defect until reconversion; perf rungs remain valid.

## L2 — g4r38: + host-work scheduling — FIRST PASS ON FIXED CKPT (overnight 09-02/03, harvested from TB after cluster sweep)

Config: L1 (textual prompt + stop ids + textual stop) + async detok/logprobs + rr routing
+ fast extract, **checkpoint = gemma4-e2b-raw2 (fixed)**, window 64. Ran 8 steps before the
cluster recycler deleted the jobset; TB + GPU-format reward JSONL survived.

| step | global (s) | note |
|---|---|---|
| 1 | 1415.3 | cold |
| 2 | 199.3 | infra turbulence |
| 3 | 227.0 | infra turbulence |
| 4 | **54.6** | clean |
| 5 | **53.5** | clean (best) |
| 6–8 | 171.7 / 192.9 / 195.4 | turbulence |

Trainer ~4.2 s (sum actor_train), weight sync ~0.45 s.

**Reward parity headline: `check_numbers` = 0.491–0.584 across all 8 steps (mean ≈0.54,
full 2048 completions/step) — inside the GPU band (0.54–0.65).** fmt 73%. First
multi-step, at-scale confirmation that the fixed checkpoint closes the GPU↔TPU reward gap
— and on the TEXTUAL prompt, showing the prompt frame is a minor variable on healthy
weights. Perf: clean band ~53–55 s at window 64 (was 133–141 s on the corrupted ckpt) —
natural stopping recovered most of the historical window-512 gain. Noisy steps reflect
overnight cluster instability; the rerun will set the official warm median.

## L3 — g4r39: + decode-loop shape

*(pending — known risk: this lineage crashed at cd512 in g4r21d; crash = chain stops,
delta stays priced by g4r33↔g4r32b = −11.0 s)*

## L4 — g4r40: + trainer/engine shape

*(pending)*

## L5 — g4r32c: champion + spike fix (target) ✅ COMPLETED 2026-09-02 ~19:5xZ

Champion stack (g4r32b) + `patch_verify_timeout` (VERIFY_ITEM_TIMEOUT=5) + `_AR_PARTIAL`
partial-async rescoring. 10 steps, exit 0.

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 55.4 | 82.4 | 188.7 | cold (sync scoring, 51 s math-verify — expected) |
| 3 | 0.22 | 33.1 | 39.6 | clean |
| 4 | 0.22 | 33.2 | 39.1 | clean (best) |
| 5 | 0.19 | 34.0 | 40.1 | clean |
| 6 | 0.24 | 46.5 | 53.9 | rollout wobble (same step as g4r32b's 54.7) |
| 7 | 0.49 | 34.9 | 41.9 | clean |
| 8 | **0.26** | 34.5 | **41.1** | **SPIKE GONE** (was 156 s rewards / 199 s global in g4r32 AND g4r32b) |
| 9 | 0.21 | 34.4 | 40.1 | clean |

(steps 1–2 lost to head-log rotation across an infra restart; pattern identical to g4r32b's.)

**Warm median ~40.1 s; the reproducible step-8 reward spike is eliminated** — async scorer
healthy every step (`errors=0`, wait ~6 ms, 2048/2048 ready). Mean is no longer
spike-dominated: the champion config is now production-clean. Health: OSL ~1950–2040,
truncation 42–45%, reward dumps consistent with the champion band (fmt≈62%, sparse correct —
the known textual-prompt ceiling that the g4r41/g4r42 canonical-prompt runs address).
