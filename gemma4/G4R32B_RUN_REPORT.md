# g4r32b — Champion Confirmation Run Report

**Date:** 2026-09-02 (launched 02:15Z, completed ~04:40Z, one infra restart absorbed) ·
**Result:** warm median **40.7 s**, best clean step **38.8 s** — champion band (38.8–43.4 across
g4r32/g4r32b) confirmed. Campaign: 125.8 → ~40.7 = **3.1×**.

## 1. What this run is

Byte-identical replication of g4r32 (the champion) plus the fmt-fixed `patch_reward_dump`:

- **Base:** tianyu g4-003 recipe verbatim (image `maxtext-rl-ironwood@sha256:f7c957d0…` = tag
  `2026-08-24-packing`; Pathways servers `20260730-jax_0.10.2`; tp1×dp128 full-slice 4×4×4;
  CD window 512; hbm 0.3; TMBS 8; chips_per_vm 8; kv_cache_buffer 4352; 29-flag XLA proxy ladder;
  gemma-3-style textual prompt via `gemma_data.py`; async reward/detok/logprobs; rr routing;
  prefix off; block 128; eos-interval 8; their 34-patch stack)
- **+ `patch_textual_stop`** (`TEXTUAL_STOP=1`): retire each CD request at the first textual
  `<end_of_turn>` (ids `[236820,643,236779,1340,236779,887,236813]`) — our novel lever
- **+ `patch_reward_dump`** (`REWARD_DUMP_N=4`) with the inline-fmt fix (pool-worker visibility bug)

Full e2e manifest (patches configmap + launcher configmap + JobSet): **`gemma4/g4r32b-manifest/`**
— reproduce with `kubectl apply -f patches.yaml -f launcher.yaml -f jobset.yaml`.

## 2. Step table

| step | rewards | rollout | global | note |
|---|---|---|---|---|
| 0 | 61.7 | 84.2 | 202.6 | cold (compile cached) |
| 1 | 1.69 | 34.8 | 46.4 | warmup |
| 2 | 0.22 | 33.5 | 40.4 | clean |
| 3 | 0.23 | 34.4 | 40.4 | clean |
| 4 | 0.21 | 33.0 | **38.8** | best |
| 5 | 0.19 | 33.6 | 39.3 | clean |
| 6 | 0.25 | 48.2 | 54.7 | rollout wobble |
| 7 | 0.43 | 33.9 | 40.7 | clean |
| 8 | **156.2** | 34.4 | 199.4 | reward spike (see §4) |
| 9 | 0.20 | 42.0 | 47.9 | mild wobble |

Median (1–9) **40.7**; clean band 38.8–40.7; mean 60.9 (spike-dominated).

## 3. Generation & reward health

- OSL ~1750–2100 (textual stop active), truncation ~37–47% (was 81–85% without the stop)
- `check_numbers` (TB): 0.019–0.028 — correct-answer density at Summer's GPU scale (+1.0/correct;
  reward functions cross-validated GPU↔TPU, one sci-notation edge case noted in the perf report)
- Per-item dumps: 40 samples — fmt=0.1 on 25 (62% tag compliance), fmt=0.0 on 15, 0 correct
  (consistent with ~2–3% accuracy at temp 0.8, step-0-policy)
- Stopped completions end `…<answer>X</answer>\n<end_of_turn><eos>` — exactly one marker (the
  49-marker self-dialog is gone)

## 4. The reward spike (step 8, 156 s) — root cause, from logs

Timeline (Cloud Logging, 04:35–04:38Z):
1. A burst of `WARNING:math_verify.grader: Timeout is disabled as timeout_seconds is None` —
   **per-item sympy timeouts are OFF in this lineage** (tianyu's stack never carried our
   `patch_verify_timeout`).
2. `REWARD-T fn=math_verify_func s=124.9 n=2048` — the synchronous math-verify pass ate 125 s of
   the 156 s: pathological LaTeX expressions running unbounded sympy, serialized 32-items/proc.
3. The step hit the sync path because the async scorer degraded that step
   (`ASYNC-REWARD … errors=1`); on every other step the async path delivered in 0.2–0.4 s.

**Mechanism = conjunction of two conditions:** (a) the async-reward path occasionally falls back to
synchronous scoring, and (b) the synchronous path has no per-item sympy timeout. Reproducible
(same step-8 spike in g4r32 and g4r32b). **Fix:** our `patch_verify_timeout`
(`VERIFY_ITEM_TIMEOUT=5`, patches `math_verify_pool.py` parse/verify per-item timeouts —
validated per-item back in g4r14b) — now folded into the master manifests for all future runs;
expected worst-case spike ≤ ~15–30 s.

## 5. Artifacts

| What | Where |
|---|---|
| Phasetime JSONs | `gs://ubench-logs/g4r32b/phasetime/step_*.json` |
| LENDIST (OSL/truncation) | `gs://ubench-logs/g4r32b/lendist/step_*.json` |
| TensorBoard | `gs://ubench-logs/g4r32b/tensorboard/` |
| Compiled reward log | `gs://ubench-logs/g4r32b/rewards/g4r32b-rewards.txt` |
| E2E manifest | repo `gemma4/g4r32b-manifest/{patches,launcher,jobset}.yaml` |
