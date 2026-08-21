"""Forward RPA_V3_*_BLOCK_SIZES on the MaxText serving path.

THE BUG. PR #3102 added the RPA v3 block-size env override, but wired it into
ONLY ONE of the two entry points to the kernel:

  layers/jax/attention/attention.py       _rpa_block_size_kwargs() -> d_block_sizes  [WIRED]
  layers/common/attention_interface.py    sharded_ragged_paged_attention(...)        [NOT WIRED]

MaxText's forward_serve_vllm imports sharded_ragged_paged_attention from
attention_interface, so OUR path is the unwired one. d_block_sizes therefore
arrives at the kernel as None and kernel.py `_prepare_block_sizes` falls back to
get_default_block_sizes(), which for tpu7 DECODE sets BOTH bkv_sz and bkv_csz to
min(min_bkv_sz_to_peak, max_kv).

MEASURED, not assumed: the decode_only trace from tianyu-rl-05-profile contains
exactly one kernel configuration --
    RPAd-p_64-bq_1_1-bkv_16384_16384      138.4 ms = 74.7% of leaf decode time
-- i.e. the untouched default, despite RPA_V3_DECODE_BLOCK_SIZES=1,16384,1,4096
being present in the process environment. The compute block is 16384, not 4096.

THE FIX. Mirror attention.py:268 at the attention_interface call site: forward the
three env-derived tuples as kwargs when they are non-empty. Guarded by `not
use_hd64` because the hd64 kernel has a different signature (it derives its own
sizes via get_tuned_block_sizes) -- same guard the surrounding code already uses
for update_kv_cache / use_causal_mask.

HOW TO VERIFY IT WORKED. Re-read the decode_only xplane and look at the kernel
name: it must become bkv_16384_4096. If it still says bkv_16384_16384 the patch
did not take effect and any timing comparison is void.

Prints exactly one of:
    RPA-BLOCKS PATCH: APPLIED
    RPA-BLOCKS PATCH: FAILED <reason>
"""

import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tpu_inference"
AI = os.path.join(ROOT, "layers/common/attention_interface.py")


def fail(msg):
    print(f"RPA-BLOCKS PATCH: FAILED {msg}")
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


if not os.path.exists(AI):
    fail(f"{AI} not found")
src = open(AI, encoding="utf-8").read()

if "_RPA_V3_BLOCK_SIZES_FORWARDED" in src:
    print("RPA-BLOCKS PATCH: already patched")
    sys.exit(0)

# `envs` must already be imported in this module (it reads USE_BATCHED_RPA_KERNEL).
if "envs.USE_BATCHED_RPA_KERNEL" not in src:
    fail("envs is not referenced in attention_interface.py -- module layout changed")

ANCHOR = """        if not use_hd64:
            kwargs["update_kv_cache"] = update_kv_cache
            kwargs["use_causal_mask"] = use_causal_mask
        return func(*args, **kwargs)
"""
if src.count(ANCHOR) != 1:
    fail(f"anchor found {src.count(ANCHOR)} times, expected 1")

INJECT = """        if not use_hd64:
            kwargs["update_kv_cache"] = update_kv_cache
            kwargs["use_causal_mask"] = use_causal_mask
            # _RPA_V3_BLOCK_SIZES_FORWARDED -- mirrors _rpa_block_size_kwargs()
            # in layers/jax/attention/attention.py, which PR #3102 wired into the
            # OTHER entry point. Without this the env override is silently
            # ignored on the MaxText serving path and the kernel uses
            # get_default_block_sizes().
            for _kw, _val in (
                    ("d_block_sizes", envs.RPA_V3_DECODE_BLOCK_SIZES),
                    ("p_block_sizes", envs.RPA_V3_PREFILL_BLOCK_SIZES),
                    ("m_block_sizes", envs.RPA_V3_MIXED_BLOCK_SIZES),
            ):
                if _val:
                    kwargs[_kw] = tuple(_val)
        return func(*args, **kwargs)
"""

src = src.replace(ANCHOR, INJECT, 1)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(AI, "w", encoding="utf-8").write(src)

v = open(AI, encoding="utf-8").read()
problems = []
if "_RPA_V3_BLOCK_SIZES_FORWARDED" not in v:
    problems.append("marker missing")
if "envs.RPA_V3_DECODE_BLOCK_SIZES" not in v:
    problems.append("decode override missing")
if v.count("return func(*args, **kwargs)") != src.count("return func(*args, **kwargs)"):
    problems.append("call site disturbed")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("RPA-BLOCKS PATCH: APPLIED")
