"""TEXTUAL-STOP: retire CD requests when the textual '<end_of_turn>' marker appears.

WHY. The gemma-3-STYLE prompt contract has no single-token terminator: the model ends its real
answer (~1000 tok) with the TEXTUAL 7-BPE-piece '<end_of_turn>' = ids
[236820,643,236779,1340,236779,887,236813], then self-dialogs to the 4096 cap (measured: 49
occurrences in one tianyu g4-003 response; 85% cap-truncation; osl_mean ~3550). GPU/verl stops on
stop-STRINGS so their OSL is ~1000+. Our CD device loop stops only on token IDS -> ~70% of decode
is wasted post-answer tokens.

WHAT. In tpu_runner._process_continue_decode_outputs (inside patch_host_extract's vectorized
block, which MUST be applied first), scan each request's window tokens for the id-subsequence;
on match, truncate valid_tokens at the END of the marker and append eos so the standard retire
path fires. Overshoot <= one CD window (a boundary-straddled marker is caught at the model's next
'<end_of_turn>' emission -- it repeats in self-dialog). Env: TEXTUAL_STOP=1 enables
(default 0 = inert); TEXTUAL_STOP_IDS overrides the pattern (json list).

NOTE. The retained text includes the marker itself (GPU stop-strings exclude it) -- a knowable,
tiny contract difference; reward extraction reads <answer> tags earlier in the text, unaffected.

FAIL-OPEN. Prints TEXTUAL-STOP PATCH: APPLIED | FAILED <reason>."""
import sys
P = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py"
s = open(P).read()
if "_TEXTUAL_STOP" in s:
    print("TEXTUAL-STOP PATCH: already applied", flush=True); sys.exit(0)
if "_he_first" not in s:
    print("TEXTUAL-STOP PATCH: FAILED requires patch_host_extract first", flush=True); sys.exit(0)
a = "        valid_tokens = tokens[:int(_he_first[req_idx])].tolist()\n"
if s.count(a) != 1:
    print("TEXTUAL-STOP PATCH: FAILED loop anchor %d" % s.count(a), flush=True); sys.exit(0)
b = ("        valid_tokens = tokens[:int(_he_first[req_idx])].tolist()\n"
     "        # _TEXTUAL_STOP: cut at the first textual <end_of_turn>; the appended eos +\n"
     "        # shortened length satisfy the original num_eos_hits/retire check below.\n"
     "        if _TS_ON and int(_he_first[req_idx]) >= len(tokens):\n"
     "            _ts_pos = _ts_find(valid_tokens)\n"
     "            if _ts_pos >= 0:\n"
     "                valid_tokens = valid_tokens[:_ts_pos] + [_TS_EOS]\n")
s = s.replace(a, b)
s += r'''
# --- _TEXTUAL_STOP ---------------------------------------------------------------
import os as _ts_os, json as _ts_json
_TS_ON = _ts_os.environ.get("TEXTUAL_STOP", "0") == "1"
try:
    _TS_PAT = [int(x) for x in _ts_json.loads(_ts_os.environ.get(
        "TEXTUAL_STOP_IDS", "[236820,643,236779,1340,236779,887,236813]"))]
except Exception:
    _TS_PAT = [236820, 643, 236779, 1340, 236779, 887, 236813]
_TS_EOS = 1
_TS_N = len(_TS_PAT)
_TS_HEAD = _TS_PAT[0]
def _ts_find(toks):
    """Index just past the END of the first full pattern occurrence, else -1."""
    n = len(toks)
    i = 0
    while i <= n - _TS_N:
        if toks[i] == _TS_HEAD and toks[i:i + _TS_N] == _TS_PAT:
            return i + _TS_N
        i += 1
    return -1
if _TS_ON:
    print("TEXTUAL-STOP: active, pattern=%s" % (_TS_PAT,), flush=True)
'''
compile(s, P, "exec"); open(P, "w").write(s)
print("TEXTUAL-STOP PATCH: APPLIED (TEXTUAL_STOP=%s)" % __import__('os').environ.get("TEXTUAL_STOP", "0(default)"), flush=True)
