"""TP-aware qkv/gate_up fusion for the gemma4 native-route weight mapping.

WHY. tpu_inference's JaxQKVParallelLinear / JaxMergedColumnParallelLinear do
NOT store the fused kernel as a plain [Q|K|V] (or [gate|up]) concatenation at
TP>1. UnquantizedLinearMethod._apply_fused un-interleaves the fused matmul
output via slice_sharded_tensor_for_concatenation(out, output_sizes,
n_shards), which reshapes the output dim as (n_shards, -1) — so the STORED
GLOBAL kernel must be interleaved by shard: shard i holds
[q_slice_i | k_slice_i | v_slice_i]  (checkpoint loaders build exactly this
via reorder_concatenated_tensor_for_sharding; n_shards = the mesh's TP degree
per QuantLinearConfig). Our weight mapping's _fuse_for_native builds the
plain concatenation — correct ONLY at TP=1 (how it was validated, t6gn8).
Pushing it at TP>1 would silently scramble attention/MLP projections.

WHAT. Rebinds _fuse_for_native in maxtext's weight_mapping/gemma4.py module
file with a version that applies the shard-interleave to both fused kernels.
The shard count comes from env GEMMA4_QKV_N_SHARDS (set it to
rollout_tensor_parallelism in the launcher; default 1 = old behavior).
Appending to the module file rebinds the module-global, and
GEMMA4_VLLM_MAPPING.preprocess_src_state_fn() resolves it at call time.

Interleave (per fused kernel, output axis last):
    parts_p = proj_p.reshape(in, n, size_p // n)
    fused   = concat([parts_0, parts_1, ...], axis=-1).reshape(in, total)
At n=1 this is exactly the plain concatenation (backward compatible).

FAIL-OPEN. Any anchor mismatch prints FAILED and exits 0.
Prints exactly one of:
    G4-NATIVE-TP PATCH: APPLIED
    G4-NATIVE-TP PATCH: FAILED <reason>
"""

import ast
import os
import sys

CAND = [
    "/deps/src/maxtext/src/maxtext/integration/tunix/weight_mapping/gemma4.py",
    "/deps/src/maxtext/integration/tunix/weight_mapping/gemma4.py",
]


def fail(msg):
    print(f"G4-NATIVE-TP PATCH: FAILED {msg}")
    sys.exit(0)


GM = next((p for p in CAND if os.path.exists(p)), None)
if GM is None:
    import glob
    hits = glob.glob("/deps/**/weight_mapping/gemma4.py", recursive=True)
    GM = hits[0] if hits else None
if GM is None:
    fail("weight_mapping/gemma4.py not found under /deps")

src = open(GM, encoding="utf-8").read()
if "_G4_NATIVE_TP" in src:
    print("G4-NATIVE-TP PATCH: already patched")
    sys.exit(0)
for needle in ("def _fuse_for_native", "preprocess_src_state_fn"):
    if needle not in src:
        fail(f"anchor '{needle}' missing in {GM}")

APPEND = '''

# --- _G4_NATIVE_TP: shard-interleaved fusion for TP>1 ------------------------
# Appended by patch_g4_native_tp.py (see its docstring). Rebinds the
# module-global _fuse_for_native; the mapping factory resolves it at call time.


def _g4tp_interleave(parts, n):
  """[proj_p (in, size_p)] -> fused (in, total) with shard i holding
  [proj0_slice_i | proj1_slice_i | ...] along the output axis."""
  import jax.numpy as jnp
  if n <= 1:
    return jnp.concatenate(parts, axis=-1)
  resh = []
  for p in parts:
    size = p.shape[-1]
    assert size % n == 0, f"output size {size} not divisible by n_shards {n}"
    resh.append(p.reshape(p.shape[0], n, size // n))
  return jnp.concatenate(resh, axis=-1).reshape(parts[0].shape[0], -1)


def _fuse_for_native(state):  # noqa: F811  (deliberate rebind)
  """TP-aware fusion: mlp wi_0/wi_1 -> gate_up and attention q/k/v -> qkv,
  stored in tpu_inference's shard-interleaved fused layout (n_shards from
  env GEMMA4_QKV_N_SHARDS = rollout tensor parallelism)."""
  import os as _os
  import jax.numpy as jnp
  from flax import nnx

  n_shards = int(_os.environ.get("GEMMA4_QKV_N_SHARDS", "1"))
  print(f"_G4_NATIVE_TP: fusing with n_shards={n_shards}", flush=True)

  flat = dict(state.flat_state())
  kv_heads = 1
  for path, var in flat.items():
    if len(path) >= 2 and str(path[-2]) == "key" and str(path[-1]) == "kernel":
      kv_heads = var.value.shape[-2]
      break

  out = []
  for path, var in flat.items():
    parts = tuple(str(p) for p in path)
    if len(parts) >= 3 and parts[-3] == "mlp" and parts[-2] in ("wi_0", "wi_1"):
      if parts[-2] == "wi_0":
        wi_1 = flat[(*path[:-2], "wi_1", path[-1])]
        fused = _g4tp_interleave([var.value, wi_1.value], n_shards)
        out.append(((*path[:-2], "gate_up", path[-1]), var.replace(value=fused)))
      continue
    if len(parts) >= 3 and parts[-3] == "self_attention" and parts[-2] in ("query", "key", "value"):
      if parts[-2] == "query":
        q = var.value
        hidden, _, head_dim = q.shape
        q2 = q.reshape(hidden, -1)
        kv_shape = (hidden, kv_heads * head_dim)
        k_path = (*path[:-2], "key", path[-1])
        v_path = (*path[:-2], "value", path[-1])
        k2 = flat[k_path].value.reshape(hidden, -1) if k_path in flat else jnp.zeros(kv_shape, q.dtype)
        v2 = flat[v_path].value.reshape(hidden, -1) if v_path in flat else jnp.zeros(kv_shape, q.dtype)
        fused = _g4tp_interleave([q2, k2, v2], n_shards)
        out.append(((*path[:-2], "qkv", path[-1]), var.replace(value=fused)))
      continue
    out.append((path, var))
  return nnx.State.from_flat_path(out)
# --- end _G4_NATIVE_TP -------------------------------------------------------
'''

new = src + APPEND
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(GM, "w", encoding="utf-8").write(new)

v = open(GM, encoding="utf-8").read()
problems = []
if v.count("def _fuse_for_native") != 2:
    problems.append("expected original + rebound _fuse_for_native")
if "_g4tp_interleave" not in v:
    problems.append("interleave helper missing")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print(f"G4-NATIVE-TP PATCH: APPLIED ({GM})")
