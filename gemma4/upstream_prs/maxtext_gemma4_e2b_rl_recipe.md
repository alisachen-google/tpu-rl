# Description

## 1. Summary

Adds a GRPO reinforcement-learning recipe for gemma4-e2b on TPU v7x (Ironwood), alongside the existing gemma4-e4b and gemma4-26b recipes:

- `src/maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh`: Cluster Toolkit / Pathways launch script for a `tpu7x` 4x4x4 slice (64 chips).
- `docs/tutorials/posttraining/rl_gemma4_e2b.md` + two index entries: setup, checkpoint conversion, what each setting does, reference performance.

Training: `google/gemma-4-E2B-it` on OpenMathInstruct-2, 256 prompts × 8 samples per update, prompts ≤ 4096 and responses ≤ 8192 tokens. Configuration only, no code changes. **Depends on #6081.**

## 2. Why this PR

- MaxText has RL recipes for gemma4-e4b (v6e) and gemma4-26b but none for gemma4-e2b, and none for TPU v7x. Users who tried had to discover the working settings themselves.
- Three settings matter and are not obvious:
  - `max_seq_token_per_tpu=12288` (native trainer packing). Without it an unpacked 64-row micro-batch does not shard over 128 devices (`query.shape[0]=64 … devices_in_data_fsdp=128`).
  - continue-decode with `max_decode_steps=1024` and `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`: each 1024-step decode window runs on device to full depth instead of returning to the host whenever a sequence ends. ~30 % faster per update, identical rollouts.
  - `DP_SCHED_ROUTING=round_robin`: the default scheduler probes all 128 vLLM data-parallel ranks per request.
- Without #6081, gemma4-e2b rollouts through `MaxTextForCausalLM` are repeated tokens; the tutorial says so.
- The script passes `hf_access_token=${HF_TOKEN}` because the `-it` tokenizer is a gated repository; its own chat template provides the turn markers, so no `chat_template_path` is needed.

## 3. Live validation: before vs after

All runs: TPU v7x 4x4x4, Pathways `jax_0.11.0`, MaxText image of 2026-09-22 plus #6081, OpenMathInstruct-2, 2048 completions per update, responses ≤ 8192 tokens, same seed.

| configuration | seconds per GRPO update (median, updates 2+) | rollouts |
|---|---:|---|
| **before**: same settings without continue-decode / round-robin (default scheduling) | 74.7–76.8 | reference |
| **after**: this script's settings (run `g4e2bup`, official `-it` conversion, 8 updates, exit 0) | **56.4** (54.4 / 54.5 / 56.5 / 56.2 / 67.4 / 58.2) | coherent, identical to the tuned internal recipe |
| same workload, verl on 64 GPUs | 31.7–35.0 | |

Accuracy of this model and stack: 3-seed GSM8K 92.39 % on TPU vs 92.42 % with verl on GPU. First update includes compilation; update time varies with completion length (one straggler batch per run at 65–78 s).

FIXES: b/123456

# Tests

- `bash -n` passes; the expanded `MAXTEXT_COMMAND` equals the validated run's arguments (every setting exists in `base.yml` / `rl.yml`; settings the run set explicitly that the script omits are at their defaults).
- End-to-end run above (`g4e2bup`, 2026-11-02).
- Repo pre-commit hooks (codespell, mdformat) pass.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed, including adding the new page to the toctree.
