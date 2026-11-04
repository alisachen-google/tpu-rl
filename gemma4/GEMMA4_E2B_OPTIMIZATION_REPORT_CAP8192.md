# gemma4-E2B GRPO on TPU v7x: end-to-end performance from response cap 2560 to cap 8192

**Scope.** gemma4-E2B GRPO on the GSM8K parity recipe (128 prompts × 16 samples = 2048 completions per update,
token-level truncated IS, scorer v2), one tpu7x 4x4x4 slice (64 chips = 128 devices), MaxText `train_rl` + tunix +
vLLM-TPU under Pathways, TP1 × DP128. Step time = median `run_global_step` over training updates (no eval or
checkpoint updates). GPU reference: verl + vLLM on 64 GPUs, same recipe, cap 8192: **31.7 / 33.1 / 35.0 s** per
update (3 seeds).

**Result.** The recipe that is like for like with the GPU reference (cap 8192, nothing truncated) runs at
**46.4 s per update** (`g4lp8k`, updates 2–19; 49.4 s over the whole 250-update run as completions lengthen) on
the latest image at p-state 6, with accuracy parity PASS
([`GEMMA4_E2B_CAP8192_PARITY_REPORT.md`](GEMMA4_E2B_CAP8192_PARITY_REPORT.md)). The capped operating points are
faster (28.9–36.7 s) but are a different training problem (§2).

## 1. How the step time moved

| stage | image | cap | DVFS | step time | what changed | evidence |
|---|---|---|---|---:|---|---|
| 1. 35 s parity recipe | pinned `f7c957d0…` | 2560 | p7 (seed 44: p6) | 34.6 / 36.2 / 36.7 s | 17 parity fixes (GPU-matching IS, scorer, template, eval); first full GPU-curve match | [`GEMMA4_E2B_GPU_TPU_PARITY_REPORT.md`](GEMMA4_E2B_GPU_TPU_PARITY_REPORT.md), `parity/logs/seed4{2,3,4}_*.log` |
| 2. 24 s configuration | pinned | 2048 | p6 | 32.1 s | perf levers: async reward scoring, round-robin DP routing, approx-max-k sampling, EOS-check interval 32, cap 2048 (−7 % vs stage 1 on GSM8K; −29 % on OpenMathInstruct where completions are longer) | [`GEMMA4_E2B_24S_CONFIG_PARITY_REPORT.md`](GEMMA4_E2B_24S_CONFIG_PARITY_REPORT.md) |
| 3. 24 s configuration, latest image | `2026-09-22-head` | 2048 | p6 | **28.9 s** | newer jax 0.11.0 / libtpu 0.0.44 / vLLM 0.28.1rc1; 13 of 36 perf patches now native (round-robin routing, KV kept across sync, 128-token blocks, async detok/logprobs, …); KV-sharing fix and explicit `local_sa_*` needed to run at all | [`GEMMA4_E2B_LATEST_IMAGE_STATUS.md`](GEMMA4_E2B_LATEST_IMAGE_STATUS.md), `parity/logs/latest_image/g4l24f_*` |
| 4. cap 8192, first check | latest | 8192 | p6 | 49.0 s | `max_target_length=8704`, `kv_cache_buffer=8704`, splash blocks 512; truncation 0 % | `parity/logs/latest_image/g4l24fc8k_*` |
| **5. cap 8192, parity run** | latest | 8192 | p6 | **46.4 s** (49.4 s whole run) | same as 4, 250 updates, 3 admissions; accuracy parity PASS | `parity/logs/latest_image/g4lp8k_*`, [`parity/configs/g4lp8k_cap8192_parity_full_config.md`](parity/configs/g4lp8k_cap8192_parity_full_config.md) |
| GPU reference | verl, 64 GPUs | 8192 | — | 31.7 / 33.1 / 35.0 s | | `parity/logs/gpu_reference_verl_config.txt` |

p-state 7 was used in stages 1–2 before the team setting was fixed at 6; on the latest image p-state 7 gives no
measurable gain (29.7 vs 28.9 s), so all later numbers are p-state 6.

### Where the time goes (phase medians, seconds)

| phase | 1. cap 2560 pinned (`g4pt1`) | 3. cap 2048 latest (`g4l24f`) | 4. cap 8192 latest (`g4l24fc8k`) | 5. cap 8192 parity (`g4lp8k`) |
|---|---:|---:|---:|---:|
| `engine_generate` (2048 completions, 128 DP ranks) | 29.5 | 19.0 | 31.6 | 27.1 |
| reward scoring (async, off the critical path) | 0.2 | 0.1 | 0.1 | 0.1 |
| advantage + data prep | 2.0 | 2.2 | 3.2 | 3.1 |
| trainer (IS recompute + 16 micro-batches of 8) + weight sync | 2.4 | 7.7 | 14.2 | 13.2 |
| **`run_global_step`** | **33.9** | **28.9** | **49.0** | **46.4** |
| mean completion length (tokens) | 432 | 486 | 489 | 443 → 626 over training |

Reading the table:

- **Stage 1 → 3 (−5 s):** generation fell from 29.5 to 19.0 s. The 24 s levers remove host work from the decode
  critical path (reward scoring overlaps the decode tail, round-robin routing avoids probing all 128 ranks, deeper
  continue-decode windows cut head round-trips), and the latest image adds faster kernels. The trainer grew from
  2.4 to 7.7 s on the new image (new attention op, `local_sa_*` path), which the generation gain more than covers.
- **Stage 3 → 5 (+17.5 s): the cost of matching the GPU cap, not a regression of the stack.** Two effects:
  1. *Rollout makespan is the longest completion.* At cap 2048 the slowest sequence on each rank stops at 2048
     tokens; at cap 8192 it runs to wherever the model stops (the GPU's longest completions are 1.5k–4.8k tokens
     on most updates). Generation goes from 19.0 to 27.1 s although the *mean* completion is unchanged (~490
     tokens). Updates whose batch holds one very long completion stand out (69 s and 113 s at update 2) and are
     deterministic for the seed; the GPU reference has the same property.
  2. *The trainer pads to the cap.* IS recompute and micro-batches run on sequences padded to 8704 tokens for
     ~500 real tokens: 7.7 → 13.2 s. This is the largest recoverable item.
- **Stage 4 → 5 (−2.6 s):** same configuration; the difference is the batch mix (update 2 of the first check was
  the 113 s outlier) and the longer sample of updates.

## 2. Why the GPU comparison has to use cap 8192

The GPU reference was trained with `max_response_length=8192`, `max_prompt_length=512` (launcher echo in
`parity/logs/gpu_reference_verl_config.txt`); the cap bit on one update of 250. At cap 2048/2560 the TPU truncates
0.05–2.2 % of rollouts on every update, each truncated completion scores 0 (cut before `\boxed{}`), and the GPU's
long completions cannot exist. The capped checks matched the GPU curve because the mean rollout is ~500 tokens and
the truncated tail is rare, but that is agreement despite a recipe difference. At cap 8192, truncation is 0 % on
both sides and the rollout-length trajectory matches the GPU's (443 → 626 vs 480 → 610 tokens), so **46.4 s is the
TPU number that pairs with the GPU's 31.7–35.0 s**.

## 3. What it took to run cap 8192 on the latest image

| # | change | why | status |
|---|---|---|---|
| 1 | latest image for every library | newer kernels; 13 of 36 perf patches are upstream and became config knobs (`DP_SCHED_ROUTING=round_robin`, `free_kv_cache_during_weight_sync=false`, `vllm_block_size=128`) | banked |
| 2 | **KV-sharing write-back fix** | the MaxText vLLM adapter scattered the 20 sharing layers' stale cache entries back over the donors' updates: gemma4-E2B rollouts were gibberish (0 % accuracy) | banked; now MaxText PR [#6081](https://github.com/AI-Hypercomputer/maxtext/pull/6081) (0.00 → 80.82 % at step 0) |
| 3 | explicit `local_sa_*` / `eval_local_sa_*` splash settings (512, `HEAD_DIM_MINOR`) | the new attention op reads them directly and the RL config left them `None` → crash at the first training step | config |
| 4 | **cap 8192**: `max_target_length=8704`, `kv_cache_buffer=8704`, splash blocks 512 | match the GPU; vLLM HBM fraction 0.3 still holds the KV cache for 256 sequences × 8704 tokens | config |
| 5 | reward-regex guard | one degenerate completion stalled reward scoring for minutes (793 s / 32 items in an earlier run) | now MaxText PR [#6082](https://github.com/AI-Hypercomputer/maxtext/pull/6082) |
| 6 | checkpoint every 20 updates, resumable checkpoints, `tools/chain-seeds4.sh`, JobSets with an ownerReference | 120-minute admission limit, node evictions, kyverno policy stripping ownerless JobSets | ops |

Not used: p-state 7 (team setting is 6); OSL-tail levers (they change the training problem).

## 4. The same move on the OpenMathInstruct-2 recipe (256 prompts × 8 samples, prefill 4096)

| cap | image | step time | note |
|---|---|---:|---|
| 2048 | pinned, p7 / p6 | 23.5 / 25.3 s | the "24 s recipe"; completions ~1255 tokens mean, 0.05–2.2 % truncated |
| 8192 | latest, p6, full patch set | 54.7 s | first cap-8192 runs (`g4u8k`, `g4u8kr`) |
| 8192 | latest, p6, **no patches except the KV fix** | **53.4 s** | after the patch audit: native `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`, `max_seq_token_per_tpu=12288` packing; rollouts identical to the patched run ([`GEMMA4_E2B_PATCH_AUDIT_LATEST_IMAGE.md`](GEMMA4_E2B_PATCH_AUDIT_LATEST_IMAGE.md)) |
| 8192 | latest, p6, upstream recipe (MaxText PR [#6083](https://github.com/AI-Hypercomputer/maxtext/pull/6083)) | 56.4 s | official `-it` conversion, MaxText data templates, 8 updates |

Here the cap costs ~2× because this recipe's completions are long (mean ~1255 tokens at cap 2048, so a large tail
was being truncated); without continue-decode-to-depth the cap-8192 recipe runs at 74.7–76.8 s.

## 5. Final state and headroom

| item | GSM8K parity recipe | OpenMathInstruct recipe |
|---|---|---|
| step time at cap 8192 | **46.4 s** (49.4 s whole run) | **53.4 s** |
| manifest | `parity/manifests/g4lp8k-latest-image-cap8192-seed42-parity-250.yaml` | `parity/manifests/g4e2bcap8k-FINAL-latest-image-openmathinstruct-cap8192-20step.yaml` |
| full configuration | `parity/configs/g4lp8k_cap8192_parity_full_config.md` | section 3 of the patch-audit report |
| truncation | 0 % | < 0.1 % |

Headroom at cap 8192, in order of size, none of which changes the accuracy curve: (1) trainer padding to the cap
on the GSM8K recipe (IS recompute and micro-batches at 8704 tokens for ~500 real tokens, ~13 s): packing or length
bucketing the trainer, as the OpenMathInstruct recipe already does with `max_seq_token_per_tpu`; (2) the rollout
tail (one longest completion per rank), which only training-objective changes shorten, so it stays for a GPU
comparison; (3) the one-off compile at update 0 and the deterministic recompile at update 6.
