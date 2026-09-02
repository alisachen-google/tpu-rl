# GPU ↔ TPU reward alignment check — 2026-09-02

Compared: GPU verl gemma4 run `rl05_parity_gemma4` (`run0902_0342` dumps + TB
`events…1788320551`, 20 steps) vs our TPU champion g4r32b (REWARD-DUMP + TB).
GPU sources: `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/` (reward_dumps,
`.hydra/config.yaml` + `overrides.yaml`, tensorboard_log, val.parquet, model dir).

## Verdict

**The reward *machinery* is aligned. The reward *outcomes* are not — TPU generation
quality is ~20× below GPU with every controllable knob matched.** This is a
generation-correctness gap on the TPU side, not a scoring gap.

## What is aligned (verified today, artifact-level)

| Dimension | GPU (verl) | TPU (ours) | Match |
|---|---|---|---|
| Reward fn | `maxtext_math_reward.py compute_score` (+1.0 correct, +0.1 format) | rl.yml defaults, same weights (worker-level parity previously cross-checked) | ✅ |
| Dump schema | `fmt/ans/gt/n_chars/tail` | `fmt/score/gold/n_chars/tail` | ✅ |
| Sampling | temp 0.8, top_k 50, top_p 0.95 | identical | ✅ |
| GRPO | batch 256, n=8, lr 1e-6, clip 0.2/0.28 | identical | ✅ |
| Prompt | `<start_of_turn>user\n{SYSTEM}\n\n{Q}<end_of_turn>\n<start_of_turn>model` (byte-read from their val.parquet) | identical string (`gemma_data.py`) | ✅ |
| Checkpoint | `gemma-4-E2B-it` HF safetensors | same, converted HF→MaxText→raw TensorStore | ⚠️ same source, different bytes |
| Response cap | 8192 | 4096 | ✗ (secondary) |
| Mean response length | 1767–2067 tok (TB) | ~1750–2100 tok (LENDIST, with textual stop) | ≈ |

## What is NOT aligned (the finding)

| Metric | GPU (TB, unbiased, steps 1–20) | TPU g4r32b | Gap |
|---|---|---|---|
| Mean score (`critic/score/mean` vs `check_numbers`+fmt) | **0.54–0.65** (≈50–60% correct) | **0.02–0.03** (2–3% correct) | ~20× |
| Format compliance | 98.5% (dump) | 62% (dump) | — |
| Truncation at cap | 0.1–0.5% @8192 | ~40% @4096 (85% without textual stop) | — |
| Natural termination | model emits a stop id; cap almost never hit | model emits textual marker then self-dialogs; stop must be forced | — |

Same weights, same prompt, same sampling → GPU answers are usually right; TPU answers
are usually wrong (`guess=40` vs gold 10, `guess=19` vs gold 60000, …) even when
well-formatted and normally terminated.

## New facts that kill an old theory

The gemma-4 HF tokenizer's actual special turn tokens are **`<|turn>`=105 and
`<turn|>`=106** — `<start_of_turn>`/`<end_of_turn>` are *not* special tokens in this
tokenizer on **either** platform (0 matches in `added_tokens`). The GPU run feeds the
same multi-BPE-piece textual markers we do and still scores ~55%. So "textual markers
tokenize as 7 pieces" does **not** explain the TPU quality collapse, and the earlier
"canonical markers are untrained" story needs re-examination: `eos_token_id=[1,106,50]`
in generation_config is what terminates GPU responses.

## RESOLVED (same day): full pipeline walk — the divergent step found

Every step from dataset to reward, compared with artifacts:

| # | Pipeline step | GPU (verl) | TPU (ours) | Verdict |
|---|---|---|---|---|
| 1 | Dataset | OpenMathInstruct-2 (preprocessed parquet, 50k val rows checked) | same dataset, HF stream | ✅ |
| 2 | System prompt + question | byte-identical text in every parquet row | identical (`gemma_data.py`) | ✅ |
| 3 | Gold answer | `ground_truth = ["ans","processed"]` | `process_answer` → same shape (`gold='["10","10"]'`) | ✅ |
| 4 | Inner prompt string | parquet `prompt[0].content` = `<start_of_turn>user\n{SYS}\n\n{Q}<end_of_turn>\n<start_of_turn>model` | same string (ours adds trailing `\n`) | ✅ |
| 5 | **Final prompt rendering** | **verl applies the model's CANONICAL gemma-4 chat template at load** (their run script says so explicitly; template renders `<bos><|turn>user\n{content}<turn|>\n<|turn>model\n`) → the model sees its **trained turn frame, ids 105/106**, wrapped around the textual content | **no template** — the inner textual string IS the whole prompt; no trained frame anywhere | ❌ **THE DIVERGENT STEP** |
| 6 | Prompt tokenization | `<|turn>`/`<turn|>` parse as single special ids 105/106 (verified in tokenizer.json added_tokens + vocab) | textual markers → plain multi-BPE pieces | ❌ (consequence of 5) |
| 7 | Stop criterion | model closes its turn with `<turn|>`=106 ∈ eos ids [1,106,50] → natural stop, 0.1–0.5% truncation | model never emits 106 (no canonical frame in prompt) → textual marker + self-dialog; our patch forces the stop | ❌ (consequence of 5) |
| 8 | Response cap | 8192 | 4096 | ✗ minor |
| 9 | Detok → extraction → scoring | `maxtext_math_reward.compute_score` | `utils_rl` (worker-level parity previously verified) | ✅ |

**Root cause: the TPU prompt lacks the canonical `<|turn>`(105)/`<turn|>`(106) trained
turn frame that the GPU pipeline adds via chat template.** The GPU model answers inside
its trained contract (≈55% correct, natural termination); the TPU model is prompted
outside it (2–3%, no termination).

This also **overturns the inherited belief** that "canonical markers are untrained in
this checkpoint": no TPU run ever tested the true canonical frame — the TPU "canonical"
attempts used the gemma-3 chat-template *file* (textual markers), and the GPU run proves
ids 105/106 are trained and load-bearing. (Note the GPU prompt is double-wrapped —
canonical frame outside, textual noise inside — and works anyway; parity should
replicate it exactly before simplifying.)

## ROOT CAUSE FOUND (2026-09-02, byte-level weights audit): corrupted embedding table

The g4r41 canonical-prompt probe came back **negative** (banner confirmed ids 105/106 reach
the model as single trained tokens, yet behavior was unchanged: truncation 36%, textual
`<end_of_turn>` in 64% of completions, 0 correct in dumps). That killed the prompt-contract
theory and pointed at the weights. A direct byte-level comparison of
`model.language_model.embed_tokens.weight` (HF safetensors, the file the GPU run loads)
vs `params.params.token_embedder.embedding` (our raw TensorStore) settles it:

| Token id | HF row norm | our row norm | cos(ours, HF) |
|---|---|---|---|
| 1 `<eos>` | 1.143 | 39.189 | **+0.61** |
| 50 `<|tool_response>` | 1.297 | 39.196 | **+0.63** |
| **105 `<|turn>`** | 1.334 | 39.179 | **+0.74** |
| **106 `<turn|>`** | 1.175 | 39.189 | **+0.70** |
| ordinary ids (17 sampled) | 1.03–1.46 | 39.18–39.20 | +0.69…+0.96 (median 0.81) |

- **Every row of our embedding has norm ≈ √1536 = 39.19** — per-row RMS-normalized.
  Real trained embeddings never have uniform norms; the HF table's norms vary naturally.
- Directions also drift (median cos 0.81), **worst on the control/turn tokens** —
  matching every observed symptom: canonical frame inert, no stop-token emission,
  fluent-but-wrong math (2–3% vs 55%).
- **The rest of the network is fine:** the final norm tensor matches HF at cos 1.0000 with
  only tiny rounding deltas (8.625→8.5625 — a quantization round-trip signature),
  suggesting the source of our `pathways-compat` checkpoint was a quantized/QAT variant
  or a conversion that dropped the embedding's per-row dequantization scales.

**Implication:** every TPU run of this campaign — ours AND the reference g4-003 lineage,
which shares this checkpoint — ran with a damaged embedding. The GPU/TPU reward gap is a
checkpoint defect, not a framework, prompt, sampling, or reward-stack difference.

**Fix:** reconvert from the verified bf16 HF safetensors
(`gs://xiaotongyang-bucket/.../models/gemma-4-E2B-it/model.safetensors`, whose bytes this
audit validated) through the MaxText conversion path → new raw TensorStore → rerun the
ladder's quality-sensitive rungs. Expected: TPU `check_numbers` ≈ GPU's ~0.55, natural
termination via ids [1,106,50], truncation → ~0 — and likely a large perf gain for free
(shorter, naturally-stopping completions shrink rollout time).

## FIX EXECUTED (2026-09-02 ~21:5xZ): reconversion complete, audit PASS

Job `g4ckpt-reconv` (CPU pod, pinned image + overlay of the fork's
`checkpoint_conversion` module with native gemma4-e2b support): HF safetensors →
`to_maxtext` (541 arrays, 4.63B params) → raw TensorStore at
**`gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/0/items`**. Built-in audit
against the HF bytes:

| id | HF norm | new norm | ratio | cos vs HF |
|---|---|---|---|---|
| 105 `<|turn>` | 1.3342 | 52.289 | 39.192 | **+0.999999** |
| 106 `<turn|>` | 1.1754 | 46.065 | 39.191 | **+0.999999** |
| 100 / 887 / 5000 / 236820 | 1.21–1.32 | 47.4–51.8 (varied!) | 39.19 | **+0.999999** |

Directions identical to HF; uniform ×√1536 scale = the documented MaxText store
convention (undone at runtime by the tunix→tpu_inference push); row norms now *varied*,
as trained embeddings must be. Contrast with the old `gemma4-e2b-raw`: uniform norms
39.19 (per-row normalized) and cos 0.61–0.96. **`gemma4-e2b-raw2` is the verified-good
checkpoint.** Validation run: g4r43 (2×2×1 smoke, canonical prompt, `DL_CKPT_PATH`
knob) launched on it.

**Formal seal (same day): MaxText's official conversion acceptance test PASSED.**
`tests/utils/forward_pass_logit_checker` (CPU job `g4ckpt-logitcheck`): MaxText forward
pass on `gemma4-e2b-raw2` vs HF transformers on the same tokens — max per-token KL
divergence **1.2e-3 / 4.0e-3 / 5.3e-3**, all under the 0.015 threshold. This validates
the entire network (attention, norms, MLPs), not just the embedding. Remaining:
behavioral smoke on TPU (g4r43/g4r44, pending capacity).

**BEHAVIORAL CONFIRMATION (2026-09-03 ~00:1xZ, g4r43 step 0 on TPU):** first rollout on
`gemma4-e2b-raw2` — **8/17 dumped items correct (47%**, GPU call-0 band is 52–70%; old
checkpoint: 0/33), median completion 1.7k chars, and the model **naturally closes its
turn with the canonical `<turn|>` (id 106)**:
`tail='</reasoning>\n<answer>108</answer><turn|>'` — the trained stop behavior no TPU
run had ever produced. The GPU↔TPU reward gap is closed: found (byte audit) → fixed
(reconversion) → formally validated (KL 5.3e-3) → behaviorally confirmed on hardware.

## The probe run: g4r41 (queued)

`g4r41` = g4r20b + `gemma_data_canonical.py`: byte-exact replica of the GPU prompt
(`<|turn>user\n{inner}<turn|>\n<|turn>model\n`, bos via tokenizer), stop ids [106,50],
plus a first-call self-check that prints the token ids of `<|turn>`/`<turn|>` (must be
single ids 105/106 for the probe to be valid). Expected if root cause confirmed:
`check_numbers` 0.02 → ~0.5, truncation → ~0, natural stops via id 106, OSL ≈ GPU's
~1900 mean. If output is still broken under the canonical frame, suspicion moves to the
checkpoint conversion of those token rows (then: teacher-forced logprob parity probe).

## Consequence if fixed

Not just correctness: GPU's naturally-terminating distribution at ~55% accuracy is the
same ~1900-token mean but with near-zero truncation — an RL signal ~20× denser. Any fix
here compounds with every perf rung on the ladder.
