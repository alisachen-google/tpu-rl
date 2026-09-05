"""Drop the per-iteration cross-DP EOS reduction in the CD loop (qwen3 item 4).

WHY. Every decode iteration computes any_hit_eos = jnp.any(active & is_eos)
over the dp-sharded batch -- a global scalar all-reduce per step -- and
carries it in the while_loop state. With CONTINUE_DECODE_EOS_CHECK_INTERVAL=0
(our benchmark setting) cond_fn is already `i < max_decode_steps` and NEVER
reads the flag, but XLA does not dead-code loop-carried state: the g4r7b
xplane still shows u32[] all-reduces + s32 permute chatter inside the window
(~half the 14.9 ms device window is these small collectives).

WHAT. When CD_NO_EOS_REDUCE=1 (checked at trace time), any_hit_eos becomes a
static False -- the reduction and its carried state vanish from the compiled
loop. active_mask updates (per-row, local) are untouched, so cd_masking /
cd_report semantics are identical. Guard note: with the flag off the early
exit could not trigger anyway at interval=0; do NOT set CD_NO_EOS_REDUCE=1
together with a nonzero eos-check interval (the env default keeps stock
behavior).

FAIL-OPEN. Prints exactly one of:
    CD-NOEOS PATCH: APPLIED
    CD-NOEOS PATCH: FAILED <reason>
"""
import ast
import os
import sys

DL = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/decode_loop.py"


def fail(msg):
    print(f"CD-NOEOS PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(DL):
    fail(f"{DL} not found")
src = open(DL, encoding="utf-8").read()
if "CD_NO_EOS_REDUCE" in src:
    print("CD-NOEOS PATCH: already patched")
    sys.exit(0)

ANCHOR = "    any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))\n"
if src.count(ANCHOR) != 1:
    fail(f"any_hit_eos anchor count {src.count(ANCHOR)}")
REP = ("    if os.environ.get(\"CD_NO_EOS_REDUCE\", \"0\") == \"1\":\n"
       "        any_hit_eos = jnp.array(False)\n"
       "    else:\n"
       "        any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))\n")
src = src.replace(ANCHOR, REP, 1)
if "import os" not in src:
    src = src.replace("import ", "import os\nimport ", 1)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(DL, "w", encoding="utf-8").write(src)
if "CD_NO_EOS_REDUCE" not in open(DL, encoding="utf-8").read():
    fail("re-verify failed")
print("CD-NOEOS PATCH: APPLIED (CD_NO_EOS_REDUCE="
      f"{os.environ.get('CD_NO_EOS_REDUCE', '0(default)')})")
