# Add gemma4-e2b GRPO RL recipe for TPU v7x

# Description

Adds a reinforcement-learning recipe for **gemma4-e2b** next to the existing gemma4-e4b and gemma4-26b ones:

- `src/maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh`: launch script (Cluster Toolkit, Pathways) for a TPU v7x 4x4x4 slice (64 chips).
- `docs/tutorials/posttraining/rl_gemma4_e2b.md`: tutorial, plus entries in `post_training_index.md`.

The recipe trains `google/gemma-4-E2B-it` with GRPO on OpenMathInstruct-2, 256 prompts × 8 samples per update, responses up to 8192 tokens, using only existing config knobs:

- native trainer packing (`max_seq_token_per_tpu=12288`)
- continue-decode with `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`, which is about 30 % faster per update than the default on this workload
- round-robin DP routing over 128 vLLM data-parallel ranks
- the bundled `openmathinstruct2_rl.json` data template

**Depends on** the KV-sharing scatter fix (`fix/vllm-kv-shared-alias-scatter`). Without that fix, gemma4-e2b rollouts through the MaxText vLLM adapter are gibberish.

# Tests

Validated live on TPU v7x 4x4x4 with the 2026-09-22 MaxText RL image plus the KV fix, Pathways `jax_0.11.0` and 8 updates. Coherent rollouts, exit 0.

| update | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| seconds | 54.4 | 54.5 | 56.5 | 56.2 | 67.4 | 58.2 |

Median: about 56 s per update. A 3-seed GSM8K accuracy check of the same model and stack against a verl GPU reference was within 0.03 pp: TPU 92.39 %, GPU 92.42 %.

`bash -n` passes on the script, and its expanded command matches the validated run's arguments.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed, including adding new documentation pages to the relevant Table of Contents.
