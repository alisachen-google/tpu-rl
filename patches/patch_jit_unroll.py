"""Jitted per-layer unroll in weight sync (qwen3 item 7, their -4.1s).

WHY. tunix generate/utils.py _unroll_scanned_layers slices a scanned-layer
param one layer at a time in a Python loop (`src_val.value[tuple(idx)]`),
each slice a separate device op dispatched through the Pathways proxy --
~290 remote round-trips per weight sync (tianyu SYNC-T3: 4.6s -> 0.5s once
jitted). Confirmed present in our stack (transfer_state_with_mappings ->
_unroll_scanned_layers, utils.py:806/464).

WHAT. When JIT_UNROLL=1, replace the per-layer Python slicing with ONE
jitted slice-all per param (cached by (ndim, layer_axis, num_layers) so it
compiles once per shape). Output layers are identical to the loop -- pure
dispatch collapse, no numeric change. Default (env unset) keeps the exact
original loop.

FAIL-OPEN. Prints exactly one of:
    JIT-UNROLL PATCH: APPLIED
    JIT-UNROLL PATCH: FAILED <reason>
"""
import ast
import os
import sys

U = "/usr/local/lib/python3.12/site-packages/tunix/generate/utils.py"


def fail(msg):
    print(f"JIT-UNROLL PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(U):
    fail(f"{U} not found")
src = open(U, encoding="utf-8").read()
if "_JIT_UNROLL_CACHE" in src:
    print("JIT-UNROLL PATCH: already patched")
    sys.exit(0)

ANCHOR = (
    "      num_layers = src_val.value.shape[layer_axis]\n"
    "      for i in range(num_layers):\n"
    "        idx = [slice(None)] * src_val.value.ndim\n"
    "        idx[layer_axis] = i\n"
    "        layer_val = src_val.value[tuple(idx)]\n"
    "        layer_key = tgt_path[i]\n"
    "        unscanned_flat[(src_key, layer_key)] = (layer_val, tgt_param[i])\n"
)
if src.count(ANCHOR) != 1:
    fail(f"unroll-loop anchor count {src.count(ANCHOR)}")

REPL = (
    "      num_layers = src_val.value.shape[layer_axis]\n"
    "      if os.environ.get('JIT_UNROLL', '0') == '1':\n"
    "        _ju_key = (src_val.value.ndim, layer_axis, num_layers)\n"
    "        _ju_fn = _JIT_UNROLL_CACHE.get(_ju_key)\n"
    "        if _ju_fn is None:\n"
    "          import jax as _ju_jax\n"
    "          import jax.numpy as _ju_jnp\n"
    "          def _ju_make(_ax, _n):\n"
    "            def _f(x):\n"
    "              return tuple(_ju_jnp.take(x, _i, axis=_ax) for _i in range(_n))\n"
    "            return _ju_jax.jit(_f)\n"
    "          _ju_fn = _ju_make(layer_axis, num_layers)\n"
    "          _JIT_UNROLL_CACHE[_ju_key] = _ju_fn\n"
    "        _ju_layers = _ju_fn(src_val.value)\n"
    "        for i in range(num_layers):\n"
    "          unscanned_flat[(src_key, tgt_path[i])] = (_ju_layers[i], tgt_param[i])\n"
    "      else:\n"
    "        for i in range(num_layers):\n"
    "          idx = [slice(None)] * src_val.value.ndim\n"
    "          idx[layer_axis] = i\n"
    "          layer_val = src_val.value[tuple(idx)]\n"
    "          layer_key = tgt_path[i]\n"
    "          unscanned_flat[(src_key, layer_key)] = (layer_val, tgt_param[i])\n"
)
src = src.replace(ANCHOR, REPL, 1)
# module-level cache + os import
if "_JIT_UNROLL_CACHE = {}" not in src:
    src = "import os\n_JIT_UNROLL_CACHE = {}\n" + src

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(U, "w", encoding="utf-8").write(src)
v = open(U, encoding="utf-8").read()
if "_JIT_UNROLL_CACHE" not in v or v.count("for i in range(num_layers):") < 2:
    fail("re-verify failed")
print(f"JIT-UNROLL PATCH: APPLIED (JIT_UNROLL={os.environ.get('JIT_UNROLL','0(default)')})")
