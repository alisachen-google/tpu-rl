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
| L5 | g4r32c | champion + spike fix (target) | — | queued (priority) |

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

*(pending — table filled on completion)*

## L2 — g4r38: + host-work scheduling

*(pending)*

## L3 — g4r39: + decode-loop shape

*(pending — known risk: this lineage crashed at cd512 in g4r21d; crash = chain stops,
delta stays priced by g4r33↔g4r32b = −11.0 s)*

## L4 — g4r40: + trainer/engine shape

*(pending)*

## L5 — g4r32c: champion + spike fix (target)

*(pending — expect ~40.7 s median with step-8 spike ≤ ~15 s)*
