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
| L1 | g4r20b | end-of-turn/reward fix | 127–133 s clean · rewards 0.50–0.57 | ✅ done |
| L2 | g4r38 | host-work scheduling | **56–58 s clean** · rewards 0.49–0.59 | ✅ done |
| L3 | g4r39 | decode-loop shape (cd512 + XLA ladder) | ❌ launcher defect (−11 s via ref pair) | crash-confirmed |
| L4 | g4r40 | trainer/engine shape (TMBS8, HBM0.3) | 56 s — null lever @win64 | ✅ done |
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

## L1 — g4r20b: + end-of-turn/reward fix ✅ MEASURED (2026-09-03, best attempt via TB; job retired amid cluster churn)

Baseline config + textual prompt (`gemma_data.py`) + stop ids [106,50] + textual stop,
checkpoint = gemma4-e2b-raw2, window 64.

| step | global (s) | note |
|---|---|---|
| 1 | 689.8 | cold |
| 2–3 | 160.2 / 163.1 | contention |
| 4 | 133.0 | clean |
| 5 | 130.5 | clean |
| 6 | 127.6 | clean (best) |

**Clean band 127.6–133.0 s** (rollout ~115–123, rewards 2–3, trainer ~5–15).
`check_numbers` **0.50–0.57** every step — GPU parity at the very first ladder rung.
L0→L1 on healthy weights: 148–155 → 127–133 clean (−~20 s: shorter healthy completions +
parallel rewards). L1→L2 (host-work levers): 130 → 56–58 clean (**−~73 s**, the ladder's
biggest measured mechanical lever at window 64).

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

## L1-fixed — g4r44: canonical prompt on FIXED checkpoint ✅ COMPLETED 2026-09-03

g4r41's exact config (canonical GPU-parity prompt, window 64) with only the checkpoint
swapped to `gemma4-e2b-raw2` — the single-variable checkpoint A/B. 3 steps, exit 0.

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 65.7 | 800.8 | 1920.0 | cold (compile + restore) |
| 1 | 2.9 | 116.5 | **135.8** | clean |
| 2 | 2.2 | 245.2 | 253.6 | rollout wobble |

**Quality (full population, TB): `check_numbers` = 0.533–0.560 (fmt 0.083–0.090) — GPU
band, sustained.** Dumps 61/97 correct (63%). Truncation 7.1%, `with_eot`=0, OSL ~1607,
every completion closing `<answer>X</answer><turn|>`.
**Perf A/B vs g4r41 (same config, broken ckpt): clean step 135.8 vs 133–141 — the
checkpoint fix is perf-neutral at the L1 config**, exactly as predicted (mechanics
unchanged; quality 0 → 0.54). The ~54 s band seen in g4r38's first pass is therefore
attributable to the L2 host-work levers, not the checkpoint.

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

## L3 — g4r39: + decode-loop shape — ❌ CONFIRMED CRASH (2026-09-03)

Window 512 + 29-flag XLA ladder on our launcher lineage: 4 attempts, 0 steps completed,
each dying in the first deep-window rollout with the IFRT-proxy-termination signature —
byte-for-byte the g4r21d failure, now reproduced **on the fixed checkpoint**, ruling out
weights as the cause. Verdict: a real launcher-lineage defect vs the reference stack's
deep-window path (their 34-patch variant runs cd512 fine). The window-512 delta stays
priced by the reference-side pair: **−11.0 s** (g4r33 ↔ g4r32b). Root-causing the parity
gap is a standing engineering item, not a ladder blocker.

## L2-official — g4r38 rerun (2026-09-03, alongside g4r20b — cross-slice contention noted)

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 61.6 | 607.6 | 718.1 | cold |
| 1–2 | 2.4 / 2.1 | 200.1 / 162.5 | 215.7 / 170.6 | contention |
| 3 | 1.8 | 49.5 | **58.1** | clean |
| 4 | 1.8 | 47.6 | **56.3** | clean (best) |
| 5 | 1.9 | 83.4 | 92.1 | mid |
| 6–7 | 2.3 / 3.0 | 206.2 / 216.5 | 232.8 / 226.9 | contention |

`check_numbers` 0.494–0.585 (reproduces the overnight pass exactly). **Training is
learning:** OSL mean fell 1750 → 1350–1400 and truncation 10% → **3.0%** by steps 6–7 —
the policy is tightening its reasoning under real reward signal. Clean band **56–58 s**;
the L1→L2 host-work delta at window 64 on healthy weights ≈ **−73 s** (130 → 57).

## L4 — g4r40: + trainer/engine shape (rebased onto L2, window 64) ✅ COMPLETED 2026-09-03

L2 + TMBS 8 + HBM 0.3, fixed checkpoint. 8 steps, exit 0.

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 75.6 | 713.6 | 1814.8 | cold (fresh trainer compile at TMBS 8) |
| 1–2 | 3.4 / 2.6 | 219.9 / 179.3 | 237.6 / 189.4 | contention |
| 3 | 1.8 | 48.3 | **56.2** | clean (best) |
| 4 | 1.9 | 48.4 | **56.4** | clean |
| 5–7 | 2.5–3.4 | 202–220 | 211.0 / 228.9 / 223.9 | contention |

**Verdict: null lever at window 64** — clean band 56.2–56.4 vs L2's 56.3–58.1. The trainer
was already ~4–5 s/step, so TMBS 8 + HBM 0.3 had nothing left to buy (mirrors the
reference chain's own "TMBS 16: null" probe). Rewards healthy (44% of dumps correct).

## Reproduction series (teammate-chain logic on the fixed checkpoint)

Following the reference chain's paired-submission logic, rerun on healthy weights.
Controls already measured: TP1×DP128 / cap 4096 = L1 (g4r20b, clean 127–133 s).

### R1 — g4r50: TP4×DP32 (their round-k pair member) ✅ COMPLETED 2026-09-03

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 80.8 | 839.8 | 1970.0 | cold (fresh TP4 compile) |
| 1–2 | 3.3 / 2.8 | 191.0 / 127.3 | 210.4 / 137.2 | warmup/contention |
| 3 | 2.9 | 77.5 | **88.2** | clean (best) |
| 4 | 2.2 | 79.7 | **89.5** | clean |
| 5 | 2.5 | 82.2 | **92.2** | clean |
| 6 | 3.0 | 197.9 | 209.1 | contention |
| 7 | 3.8 | 82.3 | 97.5 | clean-ish |

**RESULT — the reference chain's topology verdict REVERSES on healthy weights.**
Their round-k: TP1 beat TP4 by ~70 s (TP4 ~173 vs TP1 91–106) in the never-stopping
regime, where raw decode throughput ruled. On the fixed checkpoint: **TP4×DP32 = 88–92 s
clean vs TP1×DP128 = 127–133 s at matched (L1-class) config — TP4 wins by ~40 s**,
because the binding constraint is now the serial head's round-trips (32 engines to
service instead of 128), not decode FLOPs. Quality unchanged (54% of dumps correct,
truncation 2.6% by step 7 — training continuing to tighten lengths).

Follow-up staged: g4r54 = TP4×DP32 + L2 host-work levers (compound of the two winners).

### R2 — g4r52: response cap 8192 (their 004 analog) ✅ COMPLETED 2026-09-03

L1 config with `max_target_length=12288` (response budget 8192), packing budget 12288,
protected by verify-timeout + partial-async (their cap-8192 line hung unfixed on item 1128).

Clean band **146–149 s** (steps 3–5) vs **127–133 s** at cap 4096 → the cap lever is
**≈ −15 s**, less than half the reference chain's −33 s — because the model now stops
naturally, only the straggler tail uses the extra headroom. **Truncation at cap 8192:
1/2048 = 0.05%** — exactly the GPU's regime (0.1–0.5%): the truncated-completion tail is
fully recovered, and dump correctness rose to **22/33 = 67%** (vs ~54% at cap 4096, where
7–11% of completions were auto-zero). No math-verify hang (their unfixed failure mode):
the run completed 8/8 steps with rewards 1.9–3.4 s/step.

**Trade summary:** cap 8192 costs ~15 s/step and buys back ~5–10% reward density. For
training-quality-first work, take cap 8192; for pure step-time benchmarking, cap 4096.

## Ladder verdict on the fixed checkpoint (window-64 lineage)

**L0 148–155 → L1 127–133 (reward fix, healthy lengths) → L2 56–58 (host-work levers,
the dominant rung: −73 s) → L4 56 (null)**. L3 (window 512) is blocked by a confirmed
launcher-lineage defect; its −11 s remains available via the reference stack (whose
champion config measured 40.1 s in g4r32c). Quality is uniform across all fixed-ckpt
rungs: `check_numbers` ≈ 0.49–0.59 — GPU parity everywhere.

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
