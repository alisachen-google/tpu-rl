"""Batched detokenize in tunix VllmSampler (qwen3 item 11).

WHY. detokenize() calls self.tokenizer.decode() once per completion -- 2048
sequential HF decodes, ~4 s/step on tianyu's GEN-T timers.

WHAT. Collect every output's token_ids and decode them in ONE batch_decode
call before the loop; the loop then indexes the precomputed list. Falls back
to per-item decode at runtime if the tokenizer lacks batch_decode.

FAIL-OPEN. Prints exactly one of:
    FAST-DETOK PATCH: APPLIED
    FAST-DETOK PATCH: FAILED <reason>
"""
import ast
import os
import sys

VS = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"


def fail(msg):
    print(f"FAST-DETOK PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(VS):
    fail(f"{VS} not found")
src = open(VS, encoding="utf-8").read()
if "_fd_texts" in src:
    print("FAST-DETOK PATCH: already patched")
    sys.exit(0)

A1 = """    for input_string, multi_sampling_output in zip(
        input_strings, request_outputs
    ):
"""
if src.count(A1) != 1:
    fail(f"loop anchor count {src.count(A1)}")
INS = """    # _fd_: decode every completion in one batched call
    _fd_all = [
        list(_o.token_ids)
        for _mo in request_outputs
        for _o in _mo.outputs
    ]
    try:
      _fd_texts = self.tokenizer.batch_decode(_fd_all)
    except (AttributeError, TypeError):
      _fd_texts = [self.tokenizer.decode(_t) for _t in _fd_all]
    _fd_i = 0
""" + A1

A2 = """        decoded_outputs[idx].append(
            self.tokenizer.decode(single_output.token_ids)
        )
"""
if src.count(A2) != 1:
    fail(f"decode-call anchor count {src.count(A2)}")
REP = """        decoded_outputs[idx].append(_fd_texts[_fd_i])
        _fd_i += 1
"""

src = src.replace(A1, INS, 1).replace(A2, REP, 1)
try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(VS, "w", encoding="utf-8").write(src)
if "_fd_texts" not in open(VS, encoding="utf-8").read():
    fail("re-verify failed")
print("FAST-DETOK PATCH: APPLIED")
