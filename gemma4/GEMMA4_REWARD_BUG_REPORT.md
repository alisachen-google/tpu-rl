# The gemma4-E2B Reward Bug — Full Report

*2026-09-02 → 09-03 · gemma4-E2B GRPO on TPU v7x (4×4×4) vs GPU verl (GB200) ·
companion docs: `GPU_TPU_REWARD_ALIGNMENT.md` (evidence detail), `GEMMA4_LADDER_REPORT.md` (perf)*

**TL;DR:** TPU reward correctness sat at 2–3% while the GPU reference scored ~50% on the
same task, model, prompt, and reward code. The cause was not RL, prompts, or rewards — it
was a **corrupted embedding table in the shared TPU checkpoint**, present in every TPU run
of the campaign (reference lineage included). We reconverted the checkpoint from the
verified HF safetensors, validated it three independent ways, and TPU correctness is now
**0.49–0.58 — at (slightly above) GPU parity**.

---

## 1. How we found it — and how TPU reward differed from GPU

The trigger was a side-by-side of reward dumps. The GPU parity run
(`rl05_parity_gemma4`, verl on GB200) and our TPU champion logged per-item reward rows in
comparable formats, and the difference was not subtle:

| Metric (full population, TensorBoard) | GPU | TPU (then) | Gap |
|---|---|---|---|
| Correctness (`check_numbers`-equivalent) | **0.44–0.55** | **0.02–0.03** | ~20× |
| Format compliance | 98.5% | 62% | — |
| Truncation at cap | 0.1–0.5% (@8192) | 36–85% (@4096) | — |
| Termination | natural stop-token | never — forced by our patch | — |

What made it a *bug hunt* rather than a tuning question: **every controllable knob was
verified identical** before blaming anything —

- **Reward function:** same code family (`maxtext_math_reward.compute_score` ↔ our
  `utils_rl`), same weights (+1.0 correct / +0.1 format), worker-level cross-check passed.
- **Sampling:** temp 0.8, top_k 50, top_p 0.95 on both (their hydra overrides vs our flags).
- **GRPO:** batch 256, n=8, lr 1e-6, clip 0.2/0.28 — identical.
- **Prompt content:** byte-identical inner string, read from their parquet directly.
- **Response length distribution:** similar means (~1750–2050 tokens).

Same task, same policy weights (supposedly), same judge — yet TPU answers were *wrong*
(`guess=40` vs gold 10) even when well-formatted. That is a generation-quality defect,
not a scoring defect.

## 2. Root cause

Two theories were tested in order; the first died, the second was proven byte-level.

**Theory A — prompt contract (killed).** The GPU pipeline applies gemma-4's canonical
chat template at data load, wrapping prompts in the *trained* turn tokens `<|turn>`(105)
/ `<turn|>`(106); our TPU prompt used only textual gemma-3-style markers. We built a
byte-exact GPU prompt replica (run g4r41) with an in-run tokenizer check proving ids
105/106 reached the model — **and behavior did not change** (0/33 correct). Prompt ruled out.

**Theory B — the weights (proven).** A direct byte comparison of the embedding table —
HF `model.language_model.embed_tokens.weight` (what GPU loads) vs our TensorStore
`params.params.token_embedder.embedding` (what TPU loads):

| Token | HF row norm | TPU row norm | cos(TPU, HF) |
|---|---|---|---|
| `<eos>` (1) | 1.143 | 39.189 | **+0.61** |
| `<|turn>` (105) | 1.334 | 39.179 | **+0.74** |
| `<turn|>` (106) | 1.175 | 39.189 | **+0.70** |
| 17 ordinary ids | 1.03–1.46 | 39.18–39.20 | median **+0.81** |

- **Every row's norm ≈ √1536 = 39.19** — the table had been per-row normalized. No trained
  embedding has uniform row norms; token magnitude information was destroyed.
- **Directions were rotated** (median cos 0.81), *worst on the control/turn tokens* — the
  model was effectively deaf to its own stop/turn signals and degraded on everything else,
  while remaining fluent (the rest of the network matched HF at cos 1.0000 with only
  quantization-round-trip-sized deltas).
- Provenance: the `pathways-compat` checkpoint predates this campaign and was not produced
  by the (later) official gemma4-e2b converter; **every TPU run — ours and the reference
  g4-003 lineage — loaded this one damaged file.** The old "canonical markers are
  untrained" belief was this bug being observed and misattributed.

## 3. The fix

1. **Verified source:** mirrored the GPU run's `model.safetensors`
   (HF `google/gemma-4-E2B-it`, 10,246,621,918 bytes — size-matched to HF's published
   metadata, rows byte-compared) to `gs://ubench-logs/alisachen-tpu7x/hf/gemma-4-E2B-it/`.
2. **Official converter:** ran the MaxText fork's `checkpoint_conversion/to_maxtext.py`
   (native `gemma4-e2b` mapping; its embedding hook copies rows verbatim and multiplies by
   the single scalar √hidden — the documented store convention, undone at runtime by the
   tunix→tpu_inference weight-sync push) in a CPU pod on the pinned image, then re-saved
   as raw TensorStore: **`gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/0/items`**.
3. **Adoption without image churn:** a `DL_CKPT_PATH` env knob in the launcher; all ladder
   runs now point at `raw2`. No image rebuild; the checkpoint is the only variable.

**Validated three independent ways before any run adopted it:**

| Tier | Test | Result |
|---|---|---|
| Bytes | row audit vs HF (built into the conversion job) | every row **cos +0.999999**, ratio exactly √1536, norms varied (47–52) |
| Function | MaxText's official `forward_pass_logit_checker` vs HF transformers, same tokens | max per-token **KL 5.3e-3** (threshold 0.015) |
| Behavior | real GRPO rollouts on TPU | section 4 |

## 4. Current reward — and the parity evidence

**Full-population TensorBoard (`rewards/train/check_numbers`, 2048 completions/step):**

| Run (fixed ckpt) | Steps | `check_numbers` per step |
|---|---|---|
| g4r38 (textual prompt, L2 config) | 8 | 0.541 / 0.535 / 0.534 / 0.584 / 0.554 / 0.491 / 0.554 / 0.507 |
| g4r44 (canonical prompt, L1 config) | 3 | 0.560 / 0.533 / 0.542 |
| *(old checkpoint, all runs)* | — | *0.00–0.03* |

**GPU reference:** `critic/score/mean` 0.537–0.647 minus its ~0.0985 format component =
**pure correctness ≈ 0.44–0.55**. → **TPU (0.49–0.58) is at, slightly above, GPU parity** —
despite our smaller response cap (4096 vs 8192) still costing 7–11% truncated completions.

**Behavioral signatures, before → after (LENDIST, full population):**

| | Broken ckpt | Fixed ckpt |
|---|---|---|
| Truncation @4096 | 36–85% | **7–11%** |
| Textual `<end_of_turn>` self-dialog | 63–65% of completions | **0.0%** (1/2048 once) |
| Natural stop-token emission | never | **every completion**: `…<answer>X</answer><turn|>` |
| Dump-level correctness | 0/33 | 61/97 (63%) |

**Artifacts anyone can open:**
- TPU reward JSONL (GPU-compatible schema, diffable row-for-row against verl dumps):
  `gs://ubench-logs/{g4r44,g4r38,g4r20b}/reward_dumps/rewards.jsonl`
- TPU TensorBoard: `gs://ubench-logs/alisachen-tpu7x/runs/<run>/tensorboard/`
- GPU dumps/TB: `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/{reward_dumps,\.home/tensorboard_log}/`
- Fixed checkpoint + audit log: `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/`

**Perf note:** the fix is perf-neutral at fixed config (g4r44 clean 135.8 s ≈ g4r41's
133–141 s) — mechanics unchanged, quality 0.02 → 0.54 — while healthy natural stopping
plus the L2 host-work levers showed clean steps at ~54 s (window 64). The ladder continues
on the fixed checkpoint.

**Lesson institutionalized:** checkpoint conversions are now gated — the conversion job
refuses to hand over a checkpoint whose rows don't match the source at cos ≥ 0.999, and
the KL check is the acceptance test before adoption. "Loads without error" was never
evidence of correctness.
