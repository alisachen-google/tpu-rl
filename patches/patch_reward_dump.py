"""Per-item reward diagnostics dump (REWARD-DUMP lines).

Appends wrappers in utils_rl so each step prints, for the first REWARD_DUMP_N items,
one line pairing everything a reward audit needs:
    REWARD-DUMP step-local i=K fmt=<match_format_exactly score> score=<check_numbers score>
        guess=<extracted answer> gold=<target answers> n_chars=<len> tail=<last 40 chars>
Gated by env REWARD_DUMP_N (default 0 = silent; DL_RWDUMP knob). Applied AFTER
patch_fast_extract / patch_g4_format_guard so it wraps the final bindings.
FAIL-OPEN. Prints REWARD-DUMP PATCH: APPLIED | FAILED <reason>."""
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
import os as _rd_os
_RD_N = int(_rd_os.environ.get("REWARD_DUMP_N", "0"))
_RD_FMT = {"scores": None}
_rd_orig_mfe = match_format_exactly
def match_format_exactly(prompts, completions, tmvp_config, **kargs):  # noqa: F811
    r = _rd_orig_mfe(prompts, completions, tmvp_config, **kargs)
    try:
        _RD_FMT["scores"] = list(r)
    except Exception:
        _RD_FMT["scores"] = None
    return r
_rd_orig_cn = check_numbers
def check_numbers(prompts, completions, answer, tmvp_config, **kargs):  # noqa: F811
    r = _rd_orig_cn(prompts, completions, answer, tmvp_config, **kargs)
    if _RD_N > 0:
        try:
            fmt = _RD_FMT["scores"] or []
            for i in range(min(_RD_N, len(completions))):
                c = completions[i]
                try:
                    guess = extract_answer(c, tmvp_config)
                except Exception as _e:
                    guess = "<extract-error:%r>" % (_e,)
                print("REWARD-DUMP i=%d fmt=%s score=%s guess=%r gold=%r n_chars=%d tail=%r" % (
                    i, (round(fmt[i], 4) if i < len(fmt) else "?"),
                    round(float(list(r)[i]), 4), str(guess)[:60], str(answer[i])[:60],
                    len(c), c[-40:]), flush=True)
        except Exception as _e:
            print("REWARD-DUMP ERROR: %r" % (_e,), flush=True)
    return r
'''
compile(s, P, "exec"); open(P, "w").write(s)
print("REWARD-DUMP PATCH: APPLIED (REWARD_DUMP_N=%s)" % __import__('os').environ.get("REWARD_DUMP_N", "0(default)"), flush=True)
