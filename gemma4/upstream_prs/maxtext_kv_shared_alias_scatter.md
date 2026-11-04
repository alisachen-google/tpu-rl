# Description

## 1. Summary

One-line fix in `scatter_layer_kv_caches` (`src/maxtext/integration/vllm/hybrid_cache_utils.py`): skip layer entries that still are the gathered input object, so a KV-sharing layer no longer writes a stale cache back over its donor layer's update. Two unit tests added. This makes gemma4-e2b / gemma4-e4b rollouts through `MaxTextForCausalLM` on vLLM-TPU correct.

## 2. Why this PR

- gemma4-e2b has 35 layers; layers 15–34 reuse the K/V caches of layers 13 (sliding) and 14 (global), so tpu-inference allocates 15 physical caches. gemma4-e4b has 18 such layers.
- The adapter gathers a layer-ordered view of the physical caches, runs the decoder, then scatters every layer's entry back. Only donor layers replace their entry; a sharing layer's entry is still the pre-forward object. The old scatter wrote all 35 entries back, highest layer last, so the sharing layers' stale entries overwrote the donors' new K/V on every forward pass.
- Result: the shared layers attended over empty caches and decoding degenerated into repeated tokens. GRPO on gemma4 was unusable on TPU through this path.
- Fix: `if layer_kv_caches[lyr] is kv_caches[idx]: continue` in the scatter loop. Among entries the decoder did replace, the highest layer still wins, so models without KV sharing are unchanged. The identity test runs on the Python list at trace time, zero runtime cost.

## 3. Live validation: before vs after

TPU v7x 4x4x4 (64 chips, Pathways), `train_rl` GRPO, gemma4-e2b-it, GSM8K, image built from main of 2026-09-22 with only this change applied (run `g4kvpr`, 2026-11-02):

| | GSM8K accuracy, step 0, 1319 questions, greedy | sample completion |
|---|---|---|
| **before** (main) | **0.00 %** (no answer contains `\boxed{}`) | `Here is the trucks trucks trucks trucks …` |
| **after** (this PR) | **80.82 %** | `Here is the step-by-step calculation: **Step 1: …` |
| GPU reference, same model (verl) | 80.31 ± 0.24 % | |

After 2 GRPO updates: 81.35 %. Eval dumps: `gs://ubench-logs/g4kvpr/evals/eval_000.jsonl`, `eval_002.jsonl`. With the equivalent fix, a 250-update run reached 91.81 % vs 92.42 ± 0.42 % on GPU, and a 3-seed GSM8K parity check gave 92.39 % (TPU) vs 92.42 % (GPU).

FIXES: b/123456

# Tests

- `tests/post_training/unit/vllm_hybrid_cache_test.py`: `test_unchanged_sharing_layers_do_not_clobber_the_donor_update` and `test_gemma4_e2b_kv_sharing_round_trip_keeps_every_donor_update` (35 layers / 15 caches). Both fail on main, pass with the fix; the 15 existing tests still pass.

```
PYTHONPATH=src:. python3 -m pytest -q tests/post_training/unit/vllm_hybrid_cache_test.py
17 passed
```

- Repo pre-commit hooks (pylint, pyink, codespell) pass.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed (none needed: internal helper behaviour).
