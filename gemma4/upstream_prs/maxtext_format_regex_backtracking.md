# Description

`get_match_format_regex` compiles `<reasoning_start>.*?<reasoning_end>.*?<solution_start>(.+?)<solution_end>` with `DOTALL`. On degenerate completions that **repeat the tag literals without closing them** — common for small models early in RL, e.g. gemma4-e2b "echo" outputs of tens of thousands of characters — the two lazy segments backtrack over every later occurrence of each tag. A single `search()` on a ~66k-character completion runs for **more than 240 s**. `match_format_exactly` runs this regex on every rollout and `evaluate_rl` on every eval response, so one such output stalls an entire reward or eval step (we hit 793 s on a single GRPO update with gemma4-e2b).

**Fix:** make the two skip segments atomic groups — `(?>.*?<reasoning_end>)` and `(?>.*?<solution_start>)` (Python ≥ 3.11; MaxText requires 3.12). Each stops at the first occurrence and never backtracks to a later one. For this pattern that is semantically equivalent: if the remainder fails to match after the first occurrence, it also fails after any later occurrence, so matches, spans and the captured answer are unchanged — only the hopeless backtracking is removed. The `re.Pattern` API used by callers is unchanged.

| input | before | after |
|---|---:|---:|
| typical completion (~3k chars) | 0.3 ms | < 0.1 ms |
| 1000 unclosed reasoning/solution cycles (~66k chars) | > 240 s (killed) | 0.69 s |
| 300 reasoning blocks + 20k-char unclosed solution (~42k) | slow | 0.16 s |

# Tests

New tests in `tests/post_training/unit/rl_utils_test.py` (`TestMatchFormatRegexBacktracking`):
- `test_matches_the_backtracking_pattern_exactly` — on 5000 random token soups the new pattern returns the same match/no-match, span and `group(1)` as the old pattern (also checked offline on 20,000 strings: 0 mismatches).
- `test_repeated_unclosed_tags_do_not_backtrack_catastrophically` — a 1000-cycle unclosed echo completes in < 10 s (it does not finish in 240 s before the change).

```
python3 -m pytest -q tests/post_training/unit/rl_utils_test.py tests/post_training/unit/evaluate_rl_test.py
48 passed
```
(In our environment `math_verify_pool.py`'s `multiprocessing.pool.Pool` annotation needs `import multiprocessing.pool` to be loaded first for these modules to import; unrelated to this change.)

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable (unit-level change; behaviour-preserving).
- [ ] I have made or will make corresponding changes to the doc if needed (not needed).
