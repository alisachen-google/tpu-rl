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

## Hypotheses for the TPU-side collapse (ranked)

1. **Checkpoint conversion defect** (HF→MaxText OCDBT→raw TensorStore, 541 arrays):
   a systematic transform error (scaling, transpose, norm placement) that leaves the
   model fluent but reliably wrong — the classic signature.
2. **TPU inference numerics** in the rollout path: RPA attention kernel config,
   logits dtype, CD fused-loop KV/position handling.
3. **Token-level prompt delta**: bos handling / trailing `\n` after the model marker
   (cheap to check, unlikely to explain 20×).

## Proposed diagnostics (in cost order)

- **D0 (one-line patch):** dump the first request's prompt token ids on TPU; compare
  against HF `tokenizer(prompt)` — settles bos/marker encoding in minutes.
- **D1 (one short run):** greedy 20-prompt probe through our rollout stack; compare
  answers to the GPU dump's for the same questions. Right answers ⇒ blame sampling
  path; wrong answers ⇒ blame weights/numerics.
- **D2 (targeted):** teacher-force one fixed GPU completion through the TPU model and
  compare per-token logprobs vs GPU — localizes a conversion/numerics defect precisely.

## Consequence if fixed

Not just correctness: GPU's naturally-terminating distribution at ~55% accuracy is the
same ~1900-token mean but with near-zero truncation — an RL signal ~20× denser. Any fix
here compounds with every perf rung on the ladder.
