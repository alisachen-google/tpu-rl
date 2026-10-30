"""KV-ALIAS-DEDUPE: fix gemma4-E2B/E4B gibberish rollouts on the 2026-09-22 image.

Root cause (maxtext/integration/vllm/hybrid_cache_utils.py, new image):
`resolve_layer_kv_cache_indices` expands tpu-inference's `layer_name_to_kvcache_index`
into one entry PER LAYER, so KV-shared layers (gemma4-e2b: layers 15..34 -> donors 13/14)
alias their donor's physical cache. MaxText's gemma4_small decoder indexes `kv_caches` by
compact SLOT (one per physical cache; nnx_decoders._apply_gemma4_small_layers /
gemma4_small.kv_cache_slot_map) and writes updates only at slots 0..14. The adapter's
`scatter_layer_kv_caches` then copies the untouched alias entries 15..34 -- the stale
pre-forward donor caches -- back over the donors' updates ("highest-numbered layer wins"),
so layers 13 and 14 never persist any K/V and the 20 layers that share them attend over
zeros. The old adapter passed the physical list straight through (no alias expansion).

Fix: present each physical cache exactly once, in first-use layer order (== MaxText slot
order). Models without KV sharing have no aliases, so this is a no-op for them.

Fail-loud (exit 1 on anchor miss), idempotent, compile-checked.
"""
import glob
import os
import sys

MARK = "_KV_ALIAS_DEDUPE"
CANDIDATES = [
    "/deps/src/maxtext/integration/vllm/hybrid_cache_utils.py",
    "/app/src/maxtext/integration/vllm/hybrid_cache_utils.py",
    "/usr/local/lib/python3.12/site-packages/maxtext/integration/vllm/hybrid_cache_utils.py",
]

OLD = """  physical_indices = [by_layer[lyr] for lyr in range(num_layers)]
  out_of_range = [idx for idx in physical_indices if not 0 <= idx < num_kv_caches]
"""
NEW = """  physical_indices = [by_layer[lyr] for lyr in range(num_layers)]
  # _KV_ALIAS_DEDUPE: several layers may alias one physical cache (KV-shared
  # layers redirect to their donor's slot, e.g. gemma4-e2b layers 15..34 ->
  # 13/14). MaxText decoders index ``kv_caches`` by *slot*, one entry per
  # physical cache in first-use layer order (gemma4_small.kv_cache_slot_map),
  # so hand them each physical cache exactly once. Expanding the aliases
  # instead made scatter_layer_kv_caches write the stale pre-forward donor
  # cache back over the donor's update, so KV-shared layers never saw any
  # K/V and decoding degenerated into repeated tokens.
  physical_indices = list(dict.fromkeys(physical_indices))
  out_of_range = [idx for idx in physical_indices if not 0 <= idx < num_kv_caches]
"""


def main() -> int:
  paths = [p for p in CANDIDATES if os.path.exists(p)]
  override = os.environ.get("KV_ALIAS_DEDUPE_TARGET")
  if override:
    paths = [override]
  if not paths:
    print("KV-ALIAS-DEDUPE: FAILED hybrid_cache_utils.py not found", flush=True)
    return 1
  rc = 0
  for fp in paths:
    src = open(fp).read()
    if MARK in src:
      print(f"KV-ALIAS-DEDUPE: already patched {fp}", flush=True)
      continue
    n = src.count(OLD)
    if n != 1:
      print(f"KV-ALIAS-DEDUPE: FAILED anchor count={n} in {fp}", flush=True)
      rc = 1
      continue
    new_src = src.replace(OLD, NEW)
    compile(new_src, fp, "exec")
    with open(fp, "w") as f:
      f.write(new_src)
    print(f"KV-ALIAS-DEDUPE: APPLIED {fp}", flush=True)
  return rc


if __name__ == "__main__":
  sys.exit(main())
