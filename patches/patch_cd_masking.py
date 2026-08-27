"""Continuous-batching decode window v2: no EOS early-exit AND inert masked slots.

SUPERSEDES patch_cd_masking.py (contains both edits; do not apply v1).

WHY v2 EXISTS. Runs 10-10f established, with per-request runtime evidence:
- the deep window works mechanically (full depth, EOS recorded/extracted,
  every EOS-ending request retires; zombies=0, empty=0, inv=0);
- yet sequences stop terminating in proportion to window depth
  (capped-at-8192: 24.0% stock / 35.7% depth-32 / 54.0-54.5% depth-128,
  3 replications each at step 0 with byte-identical prompts);
- the failing population is per-sequence random, not rank- or
  movement-clustered, and zero vLLM preemptions occurred.

The one remaining device-side channel that scales with masked iterations:
after a sequence samples EOS, `_update_loop_state` re-feeds it
`padding_token_id` at a FROZEN position. Every subsequent iteration the model
recomputes that slot and WRITES PAD-DERIVED K/V AT THE FROZEN POSITION --
overwriting the KV of the sequence's last real token, once per masked
iteration. Stock never executes a single masked step (interval=1 exits the
window first), so this write path is unexercised in production.

THE FIX (edit 2): re-feed the slot's CURRENT input token instead of PAD.
Same token, same frozen position, same context => the recomputed K/V is
IDENTICAL to what is already stored: the write becomes idempotent and the
masked slot provably inert. One line in `_run_one_step`:

    next_input_ids = jnp.where(new_active_mask, next_input_ids, ct)

(active rows keep next_tokens; masked rows repeat their last input forever).
`step_record_tokens` still records PAD for masked rows, so host extraction is
unchanged. DP-padding rows already re-feed their initial token 0 at a frozen
position -- also idempotent.

Edit 1 (unchanged from v1): cond_fn exits on window exhausted OR all done,
instead of breaking the fused loop on any EOS:

    return jnp.logical_and(i < max_decode_steps, jnp.any(active_mask))

If truncation at depth 128 returns to ~24% with this patch, the PAD-KV
overwrite was the poison and the full rollout win (~45 s/step measured in
run 10c: ~1.3 s per 128-iteration window) unlocks. If it stays ~54%, the
fault is inside the attention kernel's handling of mixed frozen/advancing
metadata, and this escalates upstream with the full evidence chain.

FAIL-OPEN. Any anchor mismatch prints FAILED and exits 0.
Prints exactly one of:
    CD-MASK2 PATCH: APPLIED
    CD-MASK2 PATCH: FAILED <reason>
"""

import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tpu_inference"
DL = os.path.join(ROOT, "runner/decode_loop.py")


def fail(msg):
    print(f"CD-MASK2 PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(DL):
    fail(f"{DL} not found")
src = open(DL, encoding="utf-8").read()

if "_CD_MASK2" in src:
    print("CD-MASK2 PATCH: already patched")
    sys.exit(0)
if "_CD_MASK_NO_EARLY_EXIT" in src:
    fail("v1 masking already applied; image should be fresh per pod")

# ---- Edit 1: cond_fn (structural: def -> next def at same indent) ----------
lines = src.splitlines(keepends=True)
start = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("def cond_fn(carry):"):
        if start is not None:
            fail("multiple cond_fn definitions")
        start = i
if start is None:
    fail("cond_fn not found")
indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
end = None
for j in range(start + 1, len(lines)):
    s = lines[j]
    if s.strip() and not s.startswith(indent + " ") and s.startswith(indent) \
            and s.lstrip().startswith("def "):
        end = j
        break
if end is None:
    fail("could not find the def following cond_fn")

NEW_COND = (
    f"{indent}def cond_fn(carry):\n"
    f"{indent}    # _CD_MASK2 -- continuous-batching semantics: run the window to\n"
    f"{indent}    # its static depth (or until every sequence is done) instead of\n"
    f"{indent}    # breaking the fused loop on any EOS.\n"
    f"{indent}    _cd = _unpack(carry)\n"
    f"{indent}    _cd_i = _cd[0]\n"
    f"{indent}    _cd_active = _cd[2]\n"
    f"{indent}    return jnp.logical_and(_cd_i < max_decode_steps,\n"
    f"{indent}                           jnp.any(_cd_active))\n"
    f"\n")
src = "".join(lines[:start]) + NEW_COND + "".join(lines[end:])

# ---- Edit 2: idempotent masked-slot re-feed in _run_one_step ---------------
ANCHOR = """        lp_ids_step = None
        lp_val_step = None
        lp_ranks_step = None
        if has_logprobs:"""
if src.count(ANCHOR) != 1:
    fail(f"_run_one_step anchor count={src.count(ANCHOR)}")
INSERT = """        # _CD_MASK2 -- inert masked slots. v1 re-fed padding_token_id at a
        # frozen position, so each masked iteration OVERWROTE the slot's last
        # real KV with PAD-derived KV. Re-feeding the slot's current input
        # token instead makes the recomputed K/V identical to what is stored:
        # the write is idempotent and the masked slot provably inert.
        next_input_ids = jnp.where(new_active_mask, next_input_ids, ct)

"""
src = src.replace(ANCHOR, INSERT + ANCHOR)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(DL, "w", encoding="utf-8").write(src)

v = open(DL, encoding="utf-8").read()
problems = []
if v.count("_CD_MASK2") < 2:
    problems.append("markers missing")
if "should_check_eos" in v.split("def body_fn")[0].split("def cond_fn")[1]:
    problems.append("old cond_fn body still present")
if "jnp.where(new_active_mask, next_input_ids, ct)" not in v:
    problems.append("re-feed line missing")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse failed: {e}")
if problems:
    fail("; ".join(problems))
print("CD-MASK2 PATCH: APPLIED")
