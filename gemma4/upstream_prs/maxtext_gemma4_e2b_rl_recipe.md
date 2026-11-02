# Description

Adds a GRPO reinforcement-learning recipe for **gemma4-e2b** on **TPU v7x (Ironwood)**, next to the existing gemma4-e4b and gemma4-26b recipes:

- `src/maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh`: Cluster Toolkit / Pathways launch script for a `tpu7x` 4x4x4 slice (64 chips).
- `docs/tutorials/posttraining/rl_gemma4_e2b.md`: tutorial (setup, checkpoint conversion, what each setting does, reference performance), plus entries in `docs/tutorials/post_training_index.md`.

The recipe trains `google/gemma-4-E2B-it` on OpenMathInstruct-2: 256 prompts × 8 samples per update, prompts up to 4096 tokens, responses up to 8192 tokens. It uses existing config knobs only:

- trainer and vLLM rollout share the slice (`rollout_data_parallelism=128`, `rollout_tensor_parallelism=1`); native trainer packing (`max_seq_token_per_tpu=12288`)
- continue-decode with `max_decode_steps=1024` and `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`, which runs each decode window on device to full depth; about 30 % faster per update than the default on this workload (74.7 s → 53 s per update in our A/B)
- `DP_SCHED_ROUTING=round_robin` and `APPROX_TOPK=1` for the 128-rank vLLM scheduler
- the bundled `openmathinstruct2_rl.json` data template and Gemma chat template, as in the e4b recipe

**Depends on #6081** (KV-sharing scatter fix). Without it, gemma4-e2b rollouts through `MaxTextForCausalLM` on vLLM-TPU degenerate into repeated tokens; the tutorial says so.

FIXES: b/123456

# Tests

- `bash -n` passes; the expanded `MAXTEXT_COMMAND` matches the validated run's arguments.
- End to end on TPU v7x 4x4x4 (Pathways `jax_0.11.0`, MaxText image of 2026-09-22 plus #6081, `google/gemma-4-E2B-it` checkpoint, 8 GRPO updates, 2026-11-02): coherent rollouts, exit 0.

| update | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| seconds | 54.4 | 54.5 | 56.5 | 56.2 | 67.4 | 58.2 |

Median ≈ 56 s per update (2048 completions, responses up to 8192 tokens). For reference, the same workload with verl on 64 GPUs runs at 31.7–35.0 s per step.

- Accuracy of the same model and stack: 3-seed GSM8K 92.39 % on TPU vs 92.42 % on GPU (verl).

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed, including adding the new page to the toctree.
