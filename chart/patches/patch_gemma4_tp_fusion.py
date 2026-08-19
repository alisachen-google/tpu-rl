"""Make the gemma4 native weight-mapping fusion TP-aware (fix for tp>1 gibberish).

ROOT CAUSE (2026-08-19, runs g4n1s/g4n2s/g4nx1s). maxtext weight_mapping/gemma4.py
`_fuse_for_native` builds the fused `gate_up` and `qkv` source kernels in plain
[gate|up] / [Q|K|V] feature order. tpu_inference's JaxMergedColumnParallelLinear /
JaxQKVParallelLinear store the fused kernel INTERLEAVED PER TP SHARD: shard i holds
[proj0_slice_i, proj1_slice_i, ...] (see UnquantizedMergedLinearMethod + utils
`reorder_concatenated_tensor_for_sharding`; the forward un-interleaves with
`slice_sharded_tensor_for_concatenation`). Plain concat == interleaved only when tp=1
(t6gn8 PASS at tp=1/dp=4); at rollout_tensor_parallelism=4 every attention/MLP
projection is scrambled -> 97% of completions hit the cap with random-token text.
The mapping's docstring even states it does not model TP>1.

FIX. Replace `_fuse_for_native` with a version that interleaves along the fused
output dim for n_shards = GEMMA4_NATIVE_TP (env, set by the launcher to the
rollout tensor-parallel degree; default 1 == previous behaviour). Identical to
tpu_inference's reorder: split -> reshape (.., n, size_i/n) -> concat -> flatten.

FAIL-OPEN. Anchor mismatch prints FAILED, file untouched. Prints exactly one of:
    GEMMA4-TP-FUSION PATCH: APPLIED (tp=N)
    GEMMA4-TP-FUSION PATCH: already patched
    GEMMA4-TP-FUSION PATCH: FAILED <reason>
"""
import importlib.util
import os
import re
import sys

NEW_FN = r'''
def _interleave_for_shards(pieces, n_shards):
  """[p0|p1|...] -> per-shard interleaved [p0_s0 p1_s0 ... | p0_s1 p1_s1 ... | ...].

  Mirrors tpu_inference `reorder_concatenated_tensor_for_sharding` on the last dim,
  which is the storage layout of JaxMergedColumnParallelLinear / JaxQKVParallelLinear.
  """
  import jax.numpy as jnp

  if n_shards <= 1:
    return jnp.concatenate(pieces, axis=-1)
  lead = pieces[0].shape[:-1]
  per = []
  for p in pieces:
    if p.shape[-1] % n_shards != 0:
      raise ValueError(f"fused piece size {p.shape[-1]} not divisible by tp={n_shards}")
    per.append(p.reshape(*lead, n_shards, p.shape[-1] // n_shards))
  return jnp.concatenate(per, axis=-1).reshape(*lead, -1)


def _native_tp():
  """Rollout tensor-parallel degree of the native sampler (env GEMMA4_NATIVE_TP, default 1)."""
  import os as _os

  try:
    return max(1, int(_os.environ.get("GEMMA4_NATIVE_TP", "1")))
  except ValueError:
    return 1


def _fuse_for_native(state):
  """Fuses mlp wi_0/wi_1 -> gate_up and attention q/k/v -> qkv for the native model.

  The fused kernels are 2D (hidden, sum(out_sizes)) laid out INTERLEAVED PER TP
  SHARD (shard i = [gate_i|up_i] / [q_i|k_i|v_i]) for tp = GEMMA4_NATIVE_TP, which
  is how tpu_inference's merged column-parallel linears store their kernel
  (plain [Q|K|V] order is the tp=1 special case). KV-shared layers have no
  MaxText k/v weights; their sections are zero-filled (computed but never
  consumed — the layer reads the share target's KV cache).
  """
  import jax.numpy as jnp
  from flax import nnx

  tp = _native_tp()
  flat = dict(state.flat_state())
  # kv head count is uniform across layers that have k/v (gemma4: 1).
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
        fused = _interleave_for_shards([var.value, wi_1.value], tp)
        out.append(((*path[:-2], "gate_up", path[-1]), var.replace(value=fused)))
      continue  # drop wi_0/wi_1 from the pushed state
    if len(parts) >= 3 and parts[-3] == "self_attention" and parts[-2] in ("query", "key", "value"):
      if parts[-2] == "query":
        q = var.value  # (hidden, heads, head_dim)
        hidden, _, head_dim = q.shape
        q2 = q.reshape(hidden, -1)
        kv_shape = (hidden, kv_heads * head_dim)
        k_path = (*path[:-2], "key", path[-1])
        v_path = (*path[:-2], "value", path[-1])
        k2 = flat[k_path].value.reshape(hidden, -1) if k_path in flat else jnp.zeros(kv_shape, q.dtype)
        v2 = flat[v_path].value.reshape(hidden, -1) if v_path in flat else jnp.zeros(kv_shape, q.dtype)
        fused = _interleave_for_shards([q2, k2, v2], tp)
        out.append(((*path[:-2], "qkv", path[-1]), var.replace(value=fused)))
      continue  # drop query/key/value from the pushed state
    out.append((path, var))
  return nnx.State.from_flat_path(out)


'''

MARK = "_interleave_for_shards"


def fail(msg):
  print(f"GEMMA4-TP-FUSION PATCH: FAILED {msg}")
  sys.exit(0)


def main():
  spec = importlib.util.find_spec("maxtext")
  if spec is None or not spec.submodule_search_locations:
    fail("cannot locate maxtext")
  path = os.path.join(list(spec.submodule_search_locations)[0], "integration", "tunix", "weight_mapping", "gemma4.py")
  if not os.path.exists(path):
    fail(f"{path} missing")
  src = open(path).read()
  if MARK in src:
    print("GEMMA4-TP-FUSION PATCH: already patched")
    return
  m = re.search(r"\ndef _fuse_for_native\(state\):.*?(?=\ndef build_gemma4_vllm_mapping)", src, flags=re.S)
  if not m:
    fail("anchor `_fuse_for_native` .. `build_gemma4_vllm_mapping` not found")
  new = src[: m.start()] + "\n" + NEW_FN.lstrip("\n") + src[m.end() + 1 :]
  try:
    compile(new, path, "exec")
  except SyntaxError as e:
    fail(f"syntax error after patch: {e}")
  open(path, "w").write(new)
  if MARK not in open(path).read() or "def _fuse_for_native" not in open(path).read():
    fail("re-verify failed")
  print(f"GEMMA4-TP-FUSION PATCH: APPLIED (tp={os.environ.get('GEMMA4_NATIVE_TP', '1')}) -> {path}")


if __name__ == "__main__":
  main()
