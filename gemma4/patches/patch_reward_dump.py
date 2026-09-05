"""Per-item reward diagnostics dump — human-readable lines + GPU-format JSONL.

Two outputs per reward call (first REWARD_DUMP_N items):
1. Log line (unchanged format, for grepping):
     REWARD-DUMP i=K fmt=<match_format_exactly> score=<check_numbers> guess=... gold=... n_chars=... tail=...
2. GPU-parity JSONL row (same schema as verl-side maxtext_math_reward dumps —
   {"call", "fmt", "ans", "n_chars", "n_tokens", "gt", "tail"} — plus our extra "guess"),
   appended to /tmp/reward_dumps/rewards.jsonl and best-effort uploaded after each call to
     gs://ubench-logs/<JOB_IDENTIFIER>/reward_dumps/rewards.jsonl
   (override bucket dir with REWARD_DUMP_GCS; disable upload with REWARD_DUMP_GCS=off).
   "call" counts check_numbers invocations (0-based), mirroring the GPU dump's call counter.

Gated by env REWARD_DUMP_N (default 0 = silent; DL_RWDUMP knob). Applied AFTER
patch_g4_format_guard so it wraps the final bindings. FAIL-OPEN.
Prints REWARD-DUMP PATCH: APPLIED | FAILED <reason>."""
import sys
P = "/deps/src/maxtext/trainers/post_train/rl/utils_rl.py"
s = open(P).read()
if "_REWARD_DUMP" in s:
    print("REWARD-DUMP PATCH: already applied", flush=True); sys.exit(0)
for name in ("def check_numbers(", "def match_format_exactly("):
    if name not in s:
        print("REWARD-DUMP PATCH: FAILED anchor %s" % name, flush=True); sys.exit(0)
s += r'''
# --- _REWARD_DUMP ----------------------------------------------------------------
import os as _rd_os, json as _rd_json, subprocess as _rd_sp
_RD_N = int(_rd_os.environ.get("REWARD_DUMP_N", "0"))
_RD_LOCAL = "/tmp/reward_dumps/rewards.jsonl"
_RD_GCS = _rd_os.environ.get(
    "REWARD_DUMP_GCS",
    "gs://ubench-logs/%s/reward_dumps" % _rd_os.environ.get("JOB_IDENTIFIER", "unknown-run"))
_RD_STATE = {"call": 0}
_rd_orig_cn = check_numbers
def check_numbers(prompts, completions, answer, tmvp_config, **kargs):  # noqa: F811
    r = _rd_orig_cn(prompts, completions, answer, tmvp_config, **kargs)
    if _RD_N > 0:
        try:
            _call = _RD_STATE["call"]; _RD_STATE["call"] += 1
            rows = []
            for i in range(min(_RD_N, len(completions))):
                c = completions[i]
                try:
                    guess = extract_answer(c, tmvp_config)
                except Exception as _e:
                    guess = "<extract-error:%r>" % (_e,)
                # compute fmt inline (pool workers hide the wrapped fn's scores)
                try:
                    _rs = "" if getattr(tmvp_config, "reasoning_start_token_in_prompt", False) else tmvp_config.reasoning_start_token
                    _ok = False
                    _i0 = c.find(_rs) if _rs else 0
                    if _i0 >= 0:
                        _j = c.find(tmvp_config.reasoning_end_token, _i0 + len(_rs))
                        if _j >= 0:
                            _k = c.find(tmvp_config.solution_start_token, _j)
                            if _k >= 0 and c.find(tmvp_config.solution_end_token, _k + 1) >= 0:
                                _ok = True
                    _fmt_v = round(float(tmvp_config.reward_exact_format_match), 4) if _ok else 0.0
                except Exception:
                    _fmt_v = "?"
                _score = round(float(list(r)[i]), 4)
                print("REWARD-DUMP i=%d fmt=%s score=%s guess=%r gold=%r n_chars=%d tail=%r" % (
                    i, _fmt_v, _score, str(guess)[:60], str(answer[i])[:60],
                    len(c), c[-40:]), flush=True)
                rows.append({"call": _call, "fmt": _fmt_v, "ans": _score,
                             "n_chars": len(c), "n_tokens": None,
                             "gt": str(answer[i]), "tail": c[-160:],
                             "guess": str(guess)[:120]})
            _rd_os.makedirs("/tmp/reward_dumps", exist_ok=True)
            with open(_RD_LOCAL, "a") as _f:
                for row in rows:
                    _f.write(_rd_json.dumps(row) + "\n")
            if _RD_GCS != "off":
                _rd_sp.Popen(["gcloud", "storage", "cp", _RD_LOCAL, _RD_GCS + "/rewards.jsonl"],
                             stdout=_rd_sp.DEVNULL, stderr=_rd_sp.DEVNULL)
        except Exception as _e:
            print("REWARD-DUMP ERROR: %r" % (_e,), flush=True)
    return r
'''
compile(s, P, "exec"); open(P, "w").write(s)
print("REWARD-DUMP PATCH: APPLIED (REWARD_DUMP_N=%s, jsonl->%s)" % (
    __import__('os').environ.get("REWARD_DUMP_N", "0(default)"),
    __import__('os').environ.get("REWARD_DUMP_GCS", "gs://ubench-logs/<run>/reward_dumps")), flush=True)
