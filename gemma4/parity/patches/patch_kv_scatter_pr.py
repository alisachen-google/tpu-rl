"""KV-SCATTER-PR: apply the MaxText PR change (scatter_layer_kv_caches skips unchanged alias entries) for validation."""
import sys
P="/deps/src/maxtext/integration/vllm/hybrid_cache_utils.py"
OLD='  updated = list(kv_caches)\n  for lyr, idx in enumerate(physical_indices):\n    updated[idx] = layer_kv_caches[lyr]\n  return updated'
NEW="  updated = list(kv_caches)\n  for lyr, idx in enumerate(physical_indices):\n    if layer_kv_caches[lyr] is kv_caches[idx]:\n      # Unchanged view of the physical cache: nothing to write back, and doing\n      # so could clobber another (donor) layer's update of the same cache.\n      continue\n    updated[idx] = layer_kv_caches[lyr]\n  return updated"
s=open(P).read()
if "Unchanged view of the physical cache" in s:
    print("KV-SCATTER-PR: already applied"); sys.exit(0)
if s.count(OLD)!=1:
    print("KV-SCATTER-PR: FAILED anchor count", s.count(OLD)); sys.exit(1)
s=s.replace(OLD,NEW); compile(s,P,"exec"); open(P,"w").write(s); print("KV-SCATTER-PR: APPLIED")
