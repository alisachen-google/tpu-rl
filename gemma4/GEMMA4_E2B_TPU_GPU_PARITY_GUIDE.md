# gemma4-E2B TPU ↔ GPU Accuracy Parity Guide (OpenMathInstruct-2 GRPO)

**Status:** TPU side is runnable now. **GPU reference numbers are not yet available** —
every "GPU reference" bound below is marked **`TBD`** and must be filled from a verl
gemma4-E2B run before a gate that depends on it can PASS/FAIL. We run and report the TPU
side first; GPU bounds are backfilled when the GB200 reference lands.

Modeled on the Qwen3-0.6B GSM8K parity guide
(`syangx39/verl …/Tianyu_TPU_Parity_Guide_gsm8k_8k.md`), adapted to **our gemma4-E2B
recipe** (OpenMathInstruct-2, `check_numbers`+`match_format` reward, fixed `raw2`
checkpoint, DAPO clip-higher, β=0). Companion docs: `GEMMA4_REWARD_BUG_REPORT.md`,
`GPU_TPU_REWARD_ALIGNMENT.md`, `g4-e2b-head-all-in-one.yaml` (the production recipe).

---

## 0. Goal

Reproduce — on TPU v7x (4×4×4, 64 chips) — a GRPO training/accuracy curve for gemma4-E2B
using our recipe, and (once a GPU reference exists) show it lands within parity bounds of a
verl/GB200 run on the same recipe. The accuracy metric is **`check_numbers` correctness**
(extracted final answer == gold), the same scorer the GPU side uses.

Key differences from the Qwen3-0.6B guide:
| | Qwen3-0.6B guide | **gemma4-E2B (this guide)** |
|---|---|---|
| Dataset | GSM8K (boxed) | **nvidia/OpenMathInstruct-2** (train_1M); eval on a held-out split / GSM8K test |
| Reward | `boxed_math` (last `\boxed{}`) | **`check_numbers` (+1.0 exact-number) + `match_format_exactly` (+0.1)** |
| Group size | 16 generations | **8 generations** |
| Response cap | 8192 | **4096** (`max_target 8192 − max_prefill 4096`) |
| Clip | ε 0.2 (symmetric) | **DAPO clip-higher: ε_low 0.2 / ε_high 0.28** |
| KL / β | 0 | **0** (same) |
| Stop | `[151645,151643]` | **native id 106 (`<turn|>`)** |
| GPU reference | available (acc 0.748→0.806) | **TBD — not yet produced** |

---

## 1. Recipe (Tier-1 requirements) — our production config

From `g4-e2b-head-all-in-one.yaml` (latest upstream image, pure-head):

| Component | Value |
|---|---|
| Model | `gemma4-e2b`, tokenizer `google/gemma-4-E2B-it`, vocab 262144, hidden 1536 |
| **Checkpoint** | **`gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/0/items`** (the fixed one; `raw` is embedding-corrupted — see `GEMMA4_REWARD_BUG_REPORT.md`). THE parity lever. |
| Dataset | `nvidia/OpenMathInstruct-2`, `train_1M` split |
| Batch | 256 prompts × 8 generations = **2048 completions/step** |
| GRPO | β=0 (no KL, no ref model), μ=1 (one update/rollout), ε_low 0.2 / ε_high 0.28 (DAPO) |
| Advantage | A = (R − group_mean) / (group_std(ddof=1) + 1e-6); group = 8 samples/prompt |
| Optimizer | AdamW, lr 1e-6, grad_clip 1.0 |
| Train sampling | T=0.8, top_k 50, top_p 0.95 |
| Lengths | max_target 8192, max_prefill 4096 → **response cap 4096** |
| Mesh | TP1 × DP128, train_micro_batch_size 1 (262k-vocab logits; tmbs>1 OOMs) |
| Reward | `check_numbers` (+1.0 if extracted answer == gold) + `match_format_exactly` (+0.1) |
| Stop | native id 106; `stop_strings=None`; fixed-depth CD windows (`max_decode_steps=512`) |

---

## Gate 1 — Prompt rendering (CPU, ~minutes)

**Purpose:** the model sees the data template our recipe renders, and the gemma4 tokenizer
maps turn tokens to the trained ids (105 `<|turn>`, 106 `<turn|>`).

**Method:** tokenize a fixture set of OpenMathInstruct-2 prompts with `google/gemma-4-E2B-it`
and the recipe's `data_template_path`; verify ids 105/106 appear where expected.

**Note (gemma4-specific):** parity does **not** depend on the canonical chat frame — on the
fixed checkpoint, textual (`g4r38`=0.54) and canonical (`g4r44`=0.54) prompts score
identically (the corrupted-embedding era's "prompt divergence" was a red herring; the
checkpoint was the cause). So this gate is a sanity check, not a parity-critical gate.

**Pass:** 0 tokenizer id mismatches vs fixtures; 105/106 present as single special ids.

---

## Gate 2 — Scorer validation (CPU, ~minutes)

**Purpose:** TPU and GPU compute the **same** reward.

**Method:** run `check_numbers` + `match_format_exactly` on a fixture set of
(response, gold) pairs and diff against expected scores. Cross-check the TPU scorer against
the GPU verl scorer (`maxtext_math_reward.compute_score`, +1.0/+0.1 weights) — worker-level
parity was previously verified (`GPU_TPU_REWARD_ALIGNMENT.md`).

**Reward semantics (must match GPU exactly):**
- `check_numbers`: extract the final numeric answer from `<answer>…</answer>`; **1.0** iff it
  equals gold (numeric compare), else 0.0. Sentinel `-1000000` = unparseable.
- `match_format_exactly`: **+0.1** iff the closed `<answer>…</answer>` block matches the
  strict format regex, else 0.0.
- No length penalty, no partial credit.

**Pass:** 0 scorer mismatches vs expected on the fixture set.

---

## Gate 3 — Step-0 evaluation (TPU, ~10 min)

**Purpose:** frozen `raw2` weights already produce sane accuracy (confirms the checkpoint
loaded correctly, not corrupted).

**Method:** greedy generate (n=1, T=0, cap 4096) on the eval split; score `check_numbers`.

**Pass (TPU):** step-0 greedy accuracy is **non-trivial and stable** (healthy `raw2`: the
training reward sits ~0.58; greedy step-0 acc expected in a comparable band — record it).
**GPU reference bound: `TBD`.** (On the *broken* `raw` ckpt this was ~0.02 — a hard
regression signal if it ever reappears.)

---

## Gate 4 — Trainer vs sampler numerics (TPU, ~30 min)

**Purpose:** the rollout sampler logprobs and the trainer logprobs agree (TIS importance
weights are well-formed; no silent distribution shift).

**Method:** on a fixed batch, dump sampler `logp` and trainer `logp` for N sequences; compare
mean |Δlogp|, p95, max. Validate the TIS weight form:
`w = min(exp(clamp(logπ_trainer.detach() − logπ_sampler, −20, 20)), 3.0)` (no gradient).

**Pass:** mean |Δlogp|, p95, max within bands — **GPU reference bands: `TBD`**. logp computed
in **fp32** (bf16 logp differs ~0.02 nats and will fail this gate).

---

## Gate 5 — Single-step gradient replay (optional, TPU, ~1 h)

**Purpose:** given an identical step-1 batch, TPU gradients match GPU gradients in direction.

**Method:** inject a fixed step-1 batch, recompute loss/grads, compare θ-update cosine.
**Pass:** gradient cosine ≥ 0.95 (per-group) / ≥ 0.96 overall — **vs GPU replay fixture:
`TBD`** (needs the GPU reference's `grad_step1` artifact).

---

## Gate 6 — Full training runs + accuracy curve

**Purpose:** the end-to-end accuracy curve lands within parity.

**Method:** run N steps (start 8-step smoke → then the full curve, e.g. 250 steps) with
greedy eval dumps at steps 0, 20, 40, …. Three seeds for variance.

**Metric:** greedy eval `check_numbers` accuracy vs step; secondary: format rate, OSL,
truncation, entropy.
**Pass:** TPU accuracy curve within **`TBD` pp** of the GPU reference per seed; entropy
stable; no reward collapse. **Until the GPU reference exists, report the TPU curve as the
first-party result** (no PASS/FAIL, just the measured curve + the perf numbers).

---

## TPU-side execution (runnable now)

**Perf/smoke run (8 steps):** `g4-e2b-head-all-in-one.yaml` with `num_batches=8`,
`num_test_batches=0`. This is the gate before the accuracy run (confirms the recipe runs
clean + measures step time; see `GEMMA4_HEADROOM_ANALYSIS.md`).

**Accuracy run (after the 8-step smoke passes):** same manifest, with these deltas:
- `num_batches=250` (or run to a target), `checkpoint_period` as desired
- **enable greedy eval**: `num_test_batches=<n>` + an eval split + eval sampling T=0
  (eval cadence every ~20 steps) — produces `val_dump/<step>.jsonl`
- keep `enable_tunix_perf_metrics=True` for the step-time curve
- reward-dump JSONL (GPU-compatible schema) for row-level diffing against a future GPU run

Queue/capacity: run on **`ubench-regression-tests`** (dedicated 256 chips, non-preemptible
at night) once access lands, or a booked `r-xxxxx` peak slice — **not** `default` (borrow),
which gets preempted mid-run (observed: ubench reclaimed our borrowed chips before step 0).

---

## Delivery format

**Per eval step** `val_dump/<step>.jsonl`: `input, output, gt, score, check_numbers (acc),
match_format (fmt), qid, step`.
**Per training step** `rollout_dump/<step>.jsonl`: `call, fmt, ans, n_chars, gt, tail, guess`
(the GPU-compatible schema from `patch_reward_dump.py`; diffable row-for-row against verl
dumps).
**TensorBoard scalars:** `check_numbers` (correctness), `match_format`, `response_length/mean`,
truncation %, `actor/entropy_loss`, `actor/grad_norm`, `perf/train/global_step_time`.
**GCS:** `gs://ubench-logs/<run_name>/{val_dump,reward_dumps,phasetime,lendist}/`.

---

## Pitfalls (gemma4-specific)

- **Checkpoint:** must be `raw2`. `raw` has a corrupted embedding (uniform row norms, rotated
  turn tokens) → reward ~0.02 and no natural stop. If step-0 acc looks dead, suspect the ckpt.
- **106-stop:** with `stop_strings=None` the model must stop natively on id 106 (fixed-depth
  window + host EOS-trim). Watch truncation: healthy ~3%; if ~85%, the stop isn't firing.
- **262k vocab:** tmbs must be 1; if raising tmbs, set `num_vocab_tiling>1` or the trainer OOMs.
- **Cap mismatch vs GPU:** TPU response cap is 4096 vs a likely GPU 8192 — truncated
  completions auto-score 0, so match caps (or note the gap) for a clean accuracy comparison.
- **Greedy non-determinism:** compare eval-acc *distributions/means*, not individual outputs.
- **Reward ≠ eval acc:** training reward (`check_numbers` over 8 sampled gens at T=0.8) and
  greedy eval acc (n=1, T=0) are related but not identical — report both.
