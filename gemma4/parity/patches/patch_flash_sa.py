"""RUNG 8 support: let MaxText flash/splash attention run under the RL config. The RL pydantic config
(extra=forbid) has no sa_* / use_splash_scheduler fields, and AttentionOp reads them as attributes
(run 72: AttributeError sa_block_q; 72b: CLI rejects the keys). Rewrite those reads in
/deps/src/maxtext/layers/attention_op.py to getattr(config, name, default). Defaults = MaxText base:
blocks 512, fused bwd off, layouts HEAD_DIM_MINOR, splash scheduler off, fuse_reciprocal True, base2 False.
Prints FLASH-SA PATCH: APPLIED n=<replacements> | FAILED."""
import re, sys
P = "/deps/src/maxtext/layers/attention_op.py"
s = open(P).read()
if "_FLASH_SA_DEFAULTS" in s:
    print("FLASH-SA PATCH: already applied", flush=True); sys.exit(0)
def default_for(name):
    n = name.replace("local_", "")
    if n in ("sa_block_q", "sa_block_q_dkv", "sa_block_q_dq"): return "1024"
    # _TIS_PARITY: 1024 (not 2048) so the block divides BOTH the packed training
    # forward (max_seq_token_per_tpu=8192) AND the UNPACKED get_actor_per_token_logps
    # recompute that token-TIS adds at max_target_length (e.g. 3072). 2048 does not
    # divide 3072 -> splash _process_mask ValueError. Tiling only; no numeric effect.
    if n.startswith("sa_block_kv"): return "1024"
    if n in ("sa_use_fused_bwd_kernel", "use_splash_scheduler", "sa_use_base2_exp"): return "False"
    if n == "sa_fuse_reciprocal": return "True"
    if n.endswith("_layout"): return "'HEAD_DIM_MINOR'"
    return "None"
pat = re.compile(r"self\.config\.((?:local_)?(?:sa_[a-z0-9_]+|use_splash_scheduler))\b")
n = [0]
def repl(m):
    n[0] += 1
    return "getattr(self.config, %r, %s)" % (m.group(1), default_for(m.group(1)))
s2 = pat.sub(repl, s)
if n[0] == 0:
    print("FLASH-SA PATCH: FAILED no sa_* reads found", flush=True); sys.exit(0)
s2 = "# _FLASH_SA_DEFAULTS: sa_* reads tolerate configs without those fields\n" + s2
compile(s2, P, "exec"); open(P, "w").write(s2)
print("FLASH-SA PATCH: APPLIED n=%d" % n[0], flush=True)