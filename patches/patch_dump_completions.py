"""Dump the first N prompt/completion text pairs per step (validity evidence).

Piggybacks on the same grpo_learner anchor as patch_lendist. Prints up to 3
(prompt-tail, completion) pairs per step as COMPLETION-DUMP lines. Fail-open.
Prints exactly one of:
    DUMP-COMPLETIONS PATCH: APPLIED
    DUMP-COMPLETIONS PATCH: FAILED <reason>
"""
import ast, os, sys

ROOT = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages/tunix")
GL = os.path.join(ROOT, "rl/grpo/grpo_learner.py")

def fail(msg):
    print(f"DUMP-COMPLETIONS PATCH: FAILED {msg}")
    sys.exit(0)

if not os.path.exists(GL):
    fail(f"{GL} not found")
src = open(GL, encoding="utf-8").read()
if "_DUMP_COMPLETIONS" in src:
    print("DUMP-COMPLETIONS PATCH: already patched")
    sys.exit(0)

ANCHOR = "    # Log completion lengths.\n    agg_completion_mask = completion_mask.sum(axis=-1)\n"
if src.count(ANCHOR) != 1:
    fail(f"anchor count {src.count(ANCHOR)}")

INJECT = '''    # --- _DUMP_COMPLETIONS: print a few decoded pairs for validity checks ----
    try:
      _dc_step = int(self.rl_cluster.global_steps)
      for _i in range(min(3, len(rollout_output.text))):
        _p = str(training_input["prompts"][_i])[-200:].replace("\\n", "\\\\n")
        _c = str(rollout_output.text[_i])[:400].replace("\\n", "\\\\n")
        print(f"COMPLETION-DUMP step={_dc_step} i={_i} PROMPT_TAIL={_p!r} COMPLETION={_c!r}", flush=True)
    except Exception as _e:
      print("COMPLETION-DUMP ERROR: %r" % (_e,), flush=True)
    # --- end _DUMP_COMPLETIONS ------------------------------------------------

'''
src = src.replace(ANCHOR, INJECT + ANCHOR, 1)
try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"does not parse: {e}")
open(GL, "w", encoding="utf-8").write(src)
v = open(GL, encoding="utf-8").read()
if "_DUMP_COMPLETIONS" not in v:
    fail("marker missing")
print("DUMP-COMPLETIONS PATCH: APPLIED")
