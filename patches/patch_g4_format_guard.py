"""G4-FORMAT-GUARD (ported from tianyu-rl-g4-003): replace match_format_exactly's backtracking
regex with an equivalent ordered find-chain. rs.*?re.*?ss(.+?)se (DOTALL) matches iff the four
literal tokens appear in order; on gemma echo outputs (~30k chars, repeated tag literals, often
no closing tag) the regex backtracks catastrophically (793 s / 32 items on their g4-002k step 7).
Prints G4-FORMAT-GUARD PATCH: APPLIED | FAILED."""
import sys
P = "/deps/src/maxtext/trainers/post_train/rl/utils_rl.py"
s = open(P).read()
if "_G4_FORMAT_GUARD" in s:
    print("G4-FORMAT-GUARD PATCH: already applied", flush=True); sys.exit(0)
if s.count("def match_format_exactly(") != 1:
    print("G4-FORMAT-GUARD PATCH: FAILED anchor", flush=True); sys.exit(0)
s += r'''
# --- _G4_FORMAT_GUARD (appended; column-0 in file) --------------------------------
_g4fg_orig_mfe = match_format_exactly
def match_format_exactly(prompts, completions, tmvp_config, **kargs):
  rs = "" if getattr(tmvp_config, "reasoning_start_token_in_prompt", False) else tmvp_config.reasoning_start_token
  rend = tmvp_config.reasoning_end_token
  ss = tmvp_config.solution_start_token
  se = tmvp_config.solution_end_token
  scores = []
  for c in completions:
    ok = False
    i = c.find(rs) if rs else 0
    if i >= 0:
      j = c.find(rend, i + len(rs))
      if j >= 0:
        k = c.find(ss, j + len(rend))
        if k >= 0 and c.find(se, k + len(ss) + 1) >= 0:
          ok = True
    scores.append(float(tmvp_config.reward_exact_format_match) if ok else 0.0)
  return scores
print("G4-FORMAT-GUARD: match_format_exactly -> find-chain (no regex)", flush=True)
'''
open(P, "w").write(s)
print("G4-FORMAT-GUARD PATCH: APPLIED", flush=True)
