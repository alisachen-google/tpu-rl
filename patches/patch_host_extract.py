"""Vectorize EOS trimming in _process_continue_decode_outputs (qwen3 item 5).

WHY. The sync CD output path calls _extract_valid_tokens_host per request --
one np.isin per sequence, ~9.7k calls / ~70 ms per window on the g4r7b host
trace (tianyu measured 130 ms at window 128). All of it is one matrix op.

WHAT. After `eos_arr = np.atleast_1d(eos_token_id)`, compute the first-EOS
cut index for every row at once; replace the per-request call with a slice.
Semantics identical: cut at first EOS inclusive, else keep all rows.
The async-path call in _modify_prev_results is left untouched (unused here).

FAIL-OPEN. Prints exactly one of:
    HOST-EXTRACT PATCH: APPLIED
    HOST-EXTRACT PATCH: FAILED <reason>
"""
import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tpu_inference"
TR = os.path.join(ROOT, "runner/tpu_runner.py")


def fail(msg):
    print(f"HOST-EXTRACT PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(TR):
    fail(f"{TR} not found")
src = open(TR, encoding="utf-8").read()
if "_he_first" in src:
    print("HOST-EXTRACT PATCH: already patched")
    sys.exit(0)

A1 = "    eos_arr = np.atleast_1d(eos_token_id)\n"
if src.count(A1) != 1:
    fail(f"eos_arr anchor count {src.count(A1)}")
INS = (A1 +
       "    # _he_: one vectorized first-EOS pass replaces per-request np.isin\n"
       "    _he_isin = np.isin(generated_tokens_cpu, eos_arr)\n"
       "    _he_first = np.where(_he_isin.any(axis=1),\n"
       "                         _he_isin.argmax(axis=1) + 1,\n"
       "                         generated_tokens_cpu.shape[1])\n")

A2 = "        valid_tokens = _extract_valid_tokens_host(tokens, eos_token_id)\n"
if src.count(A2) != 1:
    fail(f"per-request call anchor count {src.count(A2)}")
REP = "        valid_tokens = tokens[:int(_he_first[req_idx])].tolist()\n"

src = src.replace(A1, INS, 1).replace(A2, REP, 1)
try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(TR, "w", encoding="utf-8").write(src)
if "_he_first" not in open(TR, encoding="utf-8").read():
    fail("re-verify failed")
print("HOST-EXTRACT PATCH: APPLIED")
