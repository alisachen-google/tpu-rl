"""Fast extract_answer + reward-phase timers (ported from tianyu-rl-g4-003).
utils_rl.extract_answer scans every completion char by char and slices content[:op] on every '}'
-- O(len x braces) per completion, x2048 per step. Replacement: locate each '\\boxed{' with
str.find and balance braces only inside it. Mode env FAST_EXTRACT=verify8 (default): fast result
used, every 8th item cross-checked against the original and counted. Also wraps each reward fn
with a REWARD-T timer line. Prints FAST-EXTRACT PATCH: APPLIED | FAILED."""
import sys
P = "/deps/src/maxtext/trainers/post_train/rl/utils_rl.py"
s = open(P).read()
if "_FAST_EXTRACT" in s:
    print("FAST-EXTRACT PATCH: already applied", flush=True); sys.exit(0)
for name in ("def extract_answer(", "def check_numbers(", "def match_format_exactly(", "def match_format_approximately(", "def math_verify_func("):
    if s.count(name) != 1:
        print("FAST-EXTRACT PATCH: FAILED anchor %s %d" % (name, s.count(name)), flush=True); sys.exit(0)
s += r'''
# --- _FAST_EXTRACT ----------------------------------------------------------------
import os as _fe_os, time as _fe_time
_fe_orig_extract_answer = extract_answer
_FE = {"n": 0, "checked": 0, "mismatch": 0, "t_fast": 0.0, "t_orig": 0.0}
_FE_MODE = _fe_os.environ.get("FAST_EXTRACT", "verify8")
def _fe_fast(response, tmvp_config):
    answer_tag_regex = get_answer_fallback_regex(tmvp_config)
    answer_matches = answer_tag_regex.findall(response)
    content = answer_matches[-1] if answer_matches else response
    best = None; best_close = -1; n = len(content)
    pos = content.find("\\boxed{")
    while pos != -1:
        op = pos + 6; depth = 0; i = op
        while i < n:
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i < n and i > best_close:
            best_close = i; best = content[op + 1:i].strip()
        pos = content.find("\\boxed{", pos + 1)
    if best is not None:
        return best
    m = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
    if m:
        return m.group(1).strip()
    fallback_matches = answer_tag_regex.findall(response)
    if fallback_matches:
        return fallback_matches[-1].strip()
    return FALLBACK_ANSWER
def extract_answer(response, tmvp_config):  # noqa: F811  (_FAST_EXTRACT)
    if _FE_MODE == "orig":
        return _fe_orig_extract_answer(response, tmvp_config)
    t0 = _fe_time.perf_counter(); r = _fe_fast(response, tmvp_config); _FE["t_fast"] += _fe_time.perf_counter() - t0
    _FE["n"] += 1
    if _FE_MODE == "verify8" and _FE["n"] % 8 == 0:
        t0 = _fe_time.perf_counter(); o = _fe_orig_extract_answer(response, tmvp_config); _FE["t_orig"] += _fe_time.perf_counter() - t0
        _FE["checked"] += 1
        if o != r:
            _FE["mismatch"] += 1
            if _FE["mismatch"] <= 5:
                print("FAST-EXTRACT MISMATCH fast=%r orig=%r" % (r[:80], o[:80]), flush=True)
            r = o
    if _FE["n"] % 2048 == 0:
        print("FAST-EXTRACT n=%d checked=%d mismatch=%d t_fast=%.3f t_orig(x8 est)=%.3f" % (
            _FE["n"], _FE["checked"], _FE["mismatch"], _FE["t_fast"], _FE["t_orig"] * 8), flush=True)
    return r
def _fe_timed(name, fn):
    def w(*a, **k):
        t0 = _fe_time.perf_counter(); r = fn(*a, **k)
        n = len(a[1]) if len(a) > 1 and hasattr(a[1], "__len__") else (len(k.get("completions", [])) if "completions" in k else len(a[0]) if a and hasattr(a[0], "__len__") else -1)
        print("REWARD-T fn=%s s=%.3f n=%d" % (name, _fe_time.perf_counter() - t0, n), flush=True)
        return r
    w.__name__ = fn.__name__; w.__doc__ = fn.__doc__
    return w
check_numbers = _fe_timed("check_numbers", check_numbers)
match_format_exactly = _fe_timed("match_format_exactly", match_format_exactly)
match_format_approximately = _fe_timed("match_format_approximately", match_format_approximately)
math_verify_func = _fe_timed("math_verify_func", math_verify_func)
'''
compile(s, P, "exec"); open(P, "w").write(s); print("FAST-EXTRACT PATCH: APPLIED", flush=True)
