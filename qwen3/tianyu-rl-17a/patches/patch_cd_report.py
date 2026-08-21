"""Report each sequence's real newest token, not the window's last row.

THE BUG THIS FIXES (diagnosed from run 10, which was VOID because of it).

tpu_runner.py tells the host what each sequence's newest token is by reading ONE row
of the fused window's token buffer -- the window's last:

    last_step_idx = clip(final_state.step_counter - 1, 0, static_max_decode_steps - 1)
    next_tokens_in_tpu = generated_tokens[last_step_idx, arange(batch)]

That is only correct because the window STOPS IMMEDIATELY after any sequence emits EOS.
`any_hit_eos` is therefore not a performance heuristic -- it is load-bearing for
correctness, and the single-row protocol is why the break has to be global rather than
per-sequence.

Once the window keeps running (patch_cd_masking.py), a sequence that finishes mid-window
is masked for the remaining iterations and `step_record_tokens` writes padding:

    rows:  [t0 t1 t2 t3 t4 EOS PAD PAD ... PAD]
                            ^              ^
                       correct        what the host reads

So the host sees a padding token, concludes the sequence is still generating, never
retires it, and RE-SEEDS the next window from a padding token. The model happily
continues from that seed, producing fluent text that never terminates. Measured on
run 10: OSL 3750 -> 5700, truncation 25% -> 58%, reward 0.146 -> 0.053.

Termination becomes a lottery -- a sequence only ends if its EOS happens to land on the
window's final row.

THE FIX. Take each sequence's LAST NON-PADDING row instead of the window's last row.
The buffer contents are untouched; only the derivation of "newest token" changes.

    real   = (generated_tokens != pad_token_id) & (row < step_counter)
    last   = max over rows of (row where real else 0)
    tokens = generated_tokens[last, arange(batch)]

THIS ALSO EXPLAINS A RESULT WE FILED WRONGLY. `CONTINUE_DECODE_EOS_CHECK_INTERVAL=8`
was measured 3x SLOWER (02-cdint8: ~2157 s vs 719.2 s) and recorded in FINDINGS as
"holding the loop prevents the scheduler retiring finished sequences, so useful work per
iteration collapses". That is the SAME BUG in milder form: at interval 8, a sequence
emitting EOS at iteration 2 is masked through iteration 7, so row 7 is padding and the
host misses the EOS exactly as above. The lever was never inherently bad -- it hit this
reporting bug. Fixing the report should revive it AND max_decode_steps together.

ORDERING. Independent of patch_cd_masking.py (different file), but pointless without it.

FAIL-OPEN. Any anchor mismatch prints FAILED and exits 0.
Prints exactly one of:
    CD-REPORT PATCH: APPLIED
    CD-REPORT PATCH: FAILED <reason>
"""

import ast
import os
import re
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tpu_inference"
TR = os.path.join(ROOT, "runner/tpu_runner.py")


def fail(msg):
    print(f"CD-REPORT PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(TR):
    fail(f"{TR} not found")
src = open(TR, encoding="utf-8").read()

if "_CD_REPORT_LAST_REAL_TOKEN" in src:
    print("CD-REPORT PATCH: already patched")
    sys.exit(0)

for name in ("self.pad_token_id", "self.static_max_decode_steps",
             "def _execute_continue_decode"):
    if name not in src:
        fail(f"expected '{name}' not present in tpu_runner.py")

# Whitespace-tolerant: match the two statements as one block.
PAT = re.compile(
    r"^([ \t]*)last_step_idx\s*=\s*jnp\.clip\(\s*final_state\.step_counter\s*-\s*1,.*?\)\s*\n"
    r"[ \t]*next_tokens_in_tpu\s*=\s*generated_tokens\[\s*\n?"
    r"[ \t]*last_step_idx,\s*\n?"
    r"[ \t]*jnp\.arange\(generated_tokens\.shape\[1\]\)\]\s*\n",
    re.M | re.S)
m = PAT.search(src)
if not m:
    fail("could not match the last_step_idx / next_tokens_in_tpu block")
if len(PAT.findall(src)) != 1:
    fail(f"block matched {len(PAT.findall(src))} times, expected 1")
ind = m.group(1)

NEW = (
    f"{ind}# _CD_REPORT_LAST_REAL_TOKEN -- see patch_cd_report.py.\n"
    f"{ind}# Reading ONE row (the window's last) is only correct when the window\n"
    f"{ind}# stops right after an EOS. With a continuous-batching window a finished\n"
    f"{ind}# sequence has PADDING in that row, so the host misses its EOS, never\n"
    f"{ind}# retires it, and re-seeds it from a padding token. Take each sequence's\n"
    f"{ind}# last NON-PADDING row instead. Buffer contents are untouched.\n"
    f"{ind}_cd_nsteps = jnp.clip(final_state.step_counter, 1,\n"
    f"{ind}                      self.static_max_decode_steps)\n"
    f"{ind}_cd_rows = jnp.arange(generated_tokens.shape[0])[:, None]\n"
    f"{ind}_cd_real = jnp.logical_and(\n"
    f"{ind}    generated_tokens != self.pad_token_id,\n"
    f"{ind}    _cd_rows < _cd_nsteps)\n"
    f"{ind}_cd_last = jnp.max(jnp.where(_cd_real, _cd_rows, 0), axis=0)\n"
    f"{ind}next_tokens_in_tpu = generated_tokens[\n"
    f"{ind}    _cd_last,\n"
    f"{ind}    jnp.arange(generated_tokens.shape[1])]\n")

src = src[:m.start()] + NEW + src[m.end():]

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(TR, "w", encoding="utf-8").write(src)

v = open(TR, encoding="utf-8").read()
problems = []
if "_CD_REPORT_LAST_REAL_TOKEN" not in v:
    problems.append("marker missing")
if "last_step_idx" in v:
    problems.append("old last_step_idx still referenced")
if v.count("next_tokens_in_tpu = generated_tokens[") != 1:
    problems.append("next_tokens_in_tpu assignment count wrong")
if "_cd_last" not in v:
    problems.append("new gather missing")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("CD-REPORT PATCH: APPLIED")
