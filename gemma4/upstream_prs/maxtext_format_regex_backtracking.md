# Description

## 1. Summary

`get_match_format_regex` (`src/maxtext/trainers/post_train/rl/utils_rl.py`) now uses atomic groups for its two skip segments: `(?>.*?<reasoning_end>)(?>.*?<solution_start>)`. Same matches, same captured answer, but no catastrophic backtracking on completions that repeat the tag literals without closing them. Two unit tests added.

## 2. Why this PR

- The old pattern `<reasoning_start>.*?<reasoning_end>.*?<solution_start>(.+?)<solution_end>` is applied to every rollout (`match_format_exactly`) and every eval response (`evaluate_rl`).
- Small models early in RL emit "echo" completions: thousands of repeated `<reasoning_start>…<reasoning_end><solution_start>…` openings with no closing tag. The two lazy `.*?` segments then backtrack over every later occurrence of each tag: one `search()` on a 66k-character completion runs for more than 240 s, and one such completion stalls the whole reward or eval step. In an earlier gemma4-e2b run, scoring 32 items took 793 s.
- Atomic groups stop at the first occurrence and never retry a later one. For this pattern that is equivalent: if `(.+?)<solution_end>` fails after the first `<solution_start>` it also fails after any later one, and the same holds for `<reasoning_end>`. Python ≥ 3.11 feature; MaxText requires 3.12. Callers keep using the returned `re.Pattern`.

## 3. Live validation: before vs after

| input | before (main) | after (this PR) |
|---|---:|---:|
| typical completion, ~3k chars | 0.3 ms | < 0.1 ms |
| 1000 unclosed reasoning/solution cycles, ~66k chars | > 240 s (killed) | 0.69 s |
| 300 reasoning blocks + 20k-char unclosed solution, ~42k chars | minutes | 0.16 s |
| gemma4-e2b GRPO update, worst observed reward phase | 793 s (32 items) | ≤ 2.8 s |

After: the fixed regex ran in all gemma4-e2b GRPO runs of our TPU v7x work. Across 57 timed updates (OpenMathInstruct-2, 2048 completions each, responses up to 8192 tokens) the reward phase took 0.08–2.81 s, median 1.9 s. Rewards and GSM8K accuracy are unchanged by construction (identical matches), confirmed by the 3-seed parity result 92.39 % (TPU) vs 92.42 % (GPU).

FIXES: b/123456

# Tests

- `tests/post_training/unit/rl_utils_test.py::TestMatchFormatRegexBacktracking`
  - `test_matches_the_backtracking_pattern_exactly`: 5000 random tag/token soups, same match / span / `group(1)` as the old pattern (also 20,000 strings offline, 0 mismatches). Passes on main too, i.e. it tests equivalence.
  - `test_repeated_unclosed_tags_do_not_backtrack_catastrophically`: 1000-cycle echo finishes in < 10 s. On main it does not finish within a 30 s timeout.

```
python3 -m pytest -q tests/post_training/unit/rl_utils_test.py tests/post_training/unit/evaluate_rl_test.py
48 passed
```

- Repo pre-commit hooks (pylint, pyink, codespell) pass.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed (none needed).
