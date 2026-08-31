"""RLConfig shim for splash/flash trainer attention (qwen3 item 8).

WHY. rl.yml defaults attention='dot_product': the trainer materializes a
dense [B, S, S] score matrix on 8192-token packed rows. Flash/splash
attention needs the sa_* block fields, which RLConfig does not declare --
setting them on the CLI is rejected. tianyu's sweep on qwen3 adopted
blocks q=1024 / kv=2048 (their nulls: 2048/4096 blocks, fused bwd, base2).

WHAT. Appends class-attribute defaults for every sa_* / local_sa_* field to
RLConfig in maxtext configs/types.py (same mechanism as the launcher's MROPE
shim: pydantic instance lookup misses fall back to the class attribute).
Block sizes come from SA_BLOCK_Q / SA_BLOCK_KV at import time (default
1024/2048); local_* get the same concrete values (gemma's sliding-window
layers read them directly). Activate by ALSO passing attention='flash' on
the CLI -- without it this shim is inert.

FAIL-OPEN. Prints exactly one of:
    FLASH-SA PATCH: APPLIED
    FLASH-SA PATCH: FAILED <reason>
"""
import os
import sys

CANDS = ["/deps/src/maxtext/src/maxtext/configs/types.py",
         "/deps/src/maxtext/configs/types.py"]

SHIM = '''
# --- _FLASH_SA_SHIM: RLConfig splash-attention block defaults ---
if 'RLConfig' in globals():
  import os as _sa_os
  _sa_q = int(_sa_os.environ.get("SA_BLOCK_Q", "1024"))
  _sa_kv = int(_sa_os.environ.get("SA_BLOCK_KV", "2048"))
  for _sa_f, _sa_v in [
      ("sa_block_q", _sa_q), ("sa_block_kv", _sa_kv),
      ("sa_block_kv_compute", _sa_kv), ("sa_block_q_dkv", _sa_q),
      ("sa_block_kv_dkv", _sa_kv), ("sa_block_kv_dkv_compute", _sa_kv),
      ("sa_block_q_dq", _sa_q), ("sa_block_kv_dq", _sa_kv),
      ("sa_use_fused_bwd_kernel", False),
      ("sa_q_layout", "HEAD_DIM_MINOR"), ("sa_k_layout", "HEAD_DIM_MINOR"),
      ("sa_v_layout", "HEAD_DIM_MINOR"),
      ("use_splash_scheduler", False), ("ring_scan_unroll", 1),
      ("sa_fuse_reciprocal", True), ("sa_use_base2_exp", True),
      ("local_sa_block_q", _sa_q), ("local_sa_block_kv", _sa_kv),
      ("local_sa_block_kv_compute", _sa_kv), ("local_sa_block_q_dkv", _sa_q),
      ("local_sa_block_kv_dkv", _sa_kv),
      ("local_sa_block_kv_dkv_compute", _sa_kv),
      ("local_sa_block_q_dq", _sa_q), ("local_sa_block_kv_dq", _sa_kv),
      ("local_sa_use_fused_bwd_kernel", False),
      ("local_sa_q_layout", "HEAD_DIM_MINOR"),
      ("local_sa_k_layout", "HEAD_DIM_MINOR"),
      ("local_sa_v_layout", "HEAD_DIM_MINOR"),
      ("local_use_splash_scheduler", False), ("local_ring_scan_unroll", 1),
      ("local_sa_fuse_reciprocal", True), ("local_sa_use_base2_exp", True),
      ("use_ragged_attention", False), ("ragged_block_size", 256),
  ]:
    if not hasattr(RLConfig, _sa_f):
      setattr(RLConfig, _sa_f, _sa_v)
'''


def fail(msg):
    print(f"FLASH-SA PATCH: FAILED {msg}")
    sys.exit(0)


done = False
for p in CANDS:
    if not os.path.exists(p):
        continue
    src = open(p, encoding="utf-8").read()
    if "_FLASH_SA_SHIM" in src:
        done = True
        continue
    if "class RLConfig" not in src:
        fail(f"RLConfig not found in {p}")
    open(p, "a", encoding="utf-8").write(SHIM)
    done = True
if not done:
    fail("no types.py candidate found")
print(f"FLASH-SA PATCH: APPLIED (SA_BLOCK_Q={os.environ.get('SA_BLOCK_Q', '1024(default)')}"
      f" SA_BLOCK_KV={os.environ.get('SA_BLOCK_KV', '2048(default)')})")
