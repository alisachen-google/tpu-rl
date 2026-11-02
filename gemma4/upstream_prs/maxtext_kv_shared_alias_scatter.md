# Description

`scatter_layer_kv_caches` in `src/maxtext/integration/vllm/hybrid_cache_utils.py` writes every decoder layer's KV-cache entry back to its physical cache. With **KV sharing** (gemma4-e2b: layers 15–34 reuse the caches of layers 13/14; gemma4-e4b: 18 sharing layers) only the donor layers update their cache. The sharing layers' entries are still the gathered pre-forward objects, so writing them back overwrote the donors' new K/V on every forward pass. The shared layers then attended over empty caches and vLLM-TPU decoding degenerated into repeated tokens, making GRPO rollouts through `MaxTextForCausalLM` unusable.

**Fix (one check in the scatter loop):** skip an entry if it is still the gathered input object (`layer_kv_caches[lyr] is kv_caches[idx]`) — that layer did not update the cache, so there is nothing to write back. Among entries the decoder did replace, the highest-numbered layer still wins, so models without KV sharing are unchanged. The identity test runs on the traced Python list and costs nothing at runtime.

The adapter is the only place that knows several layers alias one physical cache, so the fix belongs here rather than in the gemma4 decoder, and it protects any future KV-sharing model.

FIXES: b/123456

# Tests

**Unit (CPU)**, `tests/post_training/unit/vllm_hybrid_cache_test.py`, two new tests, both fail without the fix:
- `test_unchanged_sharing_layers_do_not_clobber_the_donor_update`
- `test_gemma4_e2b_kv_sharing_round_trip_keeps_every_donor_update` (35 layers / 15 physical caches)

```
PYTHONPATH=src:. python3 -m pytest -q tests/post_training/unit/vllm_hybrid_cache_test.py -k LayerOrderedKvCachesTest
9 passed
```
Repo pre-commit hooks (pylint, pyink, codespell) pass.

**End to end**, TPU v7x 4x4x4 (64 chips, Pathways), `train_rl` GRPO, gemma4-e2b-it, GSM8K, image `maxtext-rl-ironwood:2026-09-22-head` with only this change applied:

| | GSM8K accuracy at step 0 (1319 questions, greedy) | sample output |
|---|---|---|
| without the fix | 0.00 % (no answer has `\boxed{}`) | `Here is the trucks trucks trucks trucks …` |
| **with this change** | **80.82 %** | `Here is the step-by-step calculation: **Step 1: …` |
| same model on GPU (verl) | 80.31 ± 0.24 % | |

After 2 GRPO updates: 81.35 %. Workload `g4kvpr` (2026-11-02); eval dumps `gs://ubench-logs/g4kvpr/evals/eval_000.jsonl`, `eval_002.jsonl`. A 250-update run with the equivalent fix reached 91.81 % (GPU reference 92.42 ± 0.42 %).

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed (none needed: internal helper behaviour).
