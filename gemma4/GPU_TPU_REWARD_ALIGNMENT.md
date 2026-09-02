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
