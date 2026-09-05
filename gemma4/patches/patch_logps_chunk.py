"""Enable tunix's sequence-chunked per-token-logps (the vocab-logits memory fix).

WHY. The train step's memory whale is the logits pair: TMBS x 8192 packed tokens x 262,144 vocab
in bf16 = 34.4G forward + ~34.4G backward at TMBS=8 (~85% of the measured 80.28G jit__train_step
ask, g4r12r1 OOM). tunix ALREADY implements the fix -- compute_per_token_logps(chunk_size=N)
computes log-probs in sequence chunks of N tokens, capping the materialized logits at
[B, N, vocab] -- and grpo_learner passes training_config.compute_logps_chunk_size into the
gradient-bearing policy loss (with_loss_fn). But MaxText's train_rl never populates the field,
so it is always 0 (off).

WHAT. Rewrite the RLTrainingConfig dataclass default in tunix/rl/rl_cluster.py to read env
LOGPS_CHUNK_SIZE at import time (0/unset keeps the exact original behavior). chunk 1024 =>
logits peak ~4.3G instead of 34.4G at TMBS=8.

FAIL-OPEN. Prints exactly one of:
    LOGPS-CHUNK PATCH: APPLIED
    LOGPS-CHUNK PATCH: FAILED <reason>
"""
import sys
P = "/usr/local/lib/python3.12/site-packages/tunix/rl/rl_cluster.py"
s = open(P).read()
if "_LOGPS_CHUNK" in s:
    print("LOGPS-CHUNK PATCH: already applied", flush=True); sys.exit(0)
a = "  compute_logps_chunk_size: int = 0\n"
if s.count(a) != 1:
    print("LOGPS-CHUNK PATCH: FAILED anchor %d" % s.count(a), flush=True); sys.exit(0)
b = ("  compute_logps_chunk_size: int = int(__import__('os').environ.get('LOGPS_CHUNK_SIZE', '0'))  # _LOGPS_CHUNK\n")
s = s.replace(a, b)
compile(s, P, "exec"); open(P, "w").write(s)
v = open(P).read()
if "_LOGPS_CHUNK" not in v:
    print("LOGPS-CHUNK PATCH: FAILED re-verify", flush=True); sys.exit(0)
print("LOGPS-CHUNK PATCH: APPLIED (LOGPS_CHUNK_SIZE=%s)" % __import__('os').environ.get('LOGPS_CHUNK_SIZE', '0(default)'), flush=True)
