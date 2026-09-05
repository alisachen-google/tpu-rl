"""Opening-format census over ALL completions per step (corruption gate).

WHY. tianyu's qwen3 campaign found prefill steps padded to the oversized max
token bucket corrupt each sequence's FIRST generated token (their readout:
first_think=n/n). Our g4r7b buckets are powers of 2 to 1,048,576 while the
real prefill budget is 540,672 -- full-budget prefill steps pad ~94%. The
3-sample COMPLETION-DUMP showed 43/60 completions opening with an echo of the
instruction text instead of the template's <reasoning> format. This patch
counts openings over the full batch so a control/fix pair is decisive:
corruption predicts the distribution SHIFTS under VLLM_TPU_BUCKET_PADDING_GAP;
weak policy predicts it does not.

Piggybacks AFTER patch_dump_completions (anchors on its marker; roster order
guarantees it). Fail-open. Prints exactly one of:
    FIRSTTOK PATCH: APPLIED
    FIRSTTOK PATCH: FAILED <reason>
Per-step readout:
    FIRSTTOK step=N n=2048 echo=E reasoning=R other=O
"""
import os
import sys

ROOT = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages/tunix")
GL = os.path.join(ROOT, "rl/grpo/grpo_learner.py")


def fail(msg):
    print(f"FIRSTTOK PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(GL):
    fail(f"{GL} not found")
src = open(GL, encoding="utf-8").read()
if "_FIRSTTOK" in src:
    print("FIRSTTOK PATCH: already patched")
    sys.exit(0)

ANCHOR = "    # --- _DUMP_COMPLETIONS: print a few decoded pairs for validity checks ----\n"
if src.count(ANCHOR) != 1:
    fail(f"dump-completions marker count {src.count(ANCHOR)} (needs patch_dump_completions first)")

INJECT = '''    # --- _FIRSTTOK: opening-format census over all completions ----
    try:
      _ft_txts = list(rollout_output.text)
      _ft_echo = sum(1 for _t in _ft_txts if _t.lstrip().startswith("You are given"))
      _ft_reason = sum(1 for _t in _ft_txts if _t.lstrip().startswith("<reasoning>"))
      print(
          f"FIRSTTOK step={int(self.rl_cluster.global_steps)} n={len(_ft_txts)}"
          f" echo={_ft_echo} reasoning={_ft_reason}"
          f" other={len(_ft_txts) - _ft_echo - _ft_reason}",
          flush=True)
    except Exception as _ft_e:  # never break training
      print(f"FIRSTTOK census error: {_ft_e}", flush=True)
'''

src = src.replace(ANCHOR, INJECT + ANCHOR, 1)
open(GL, "w", encoding="utf-8").write(src)
if "_FIRSTTOK" not in open(GL, encoding="utf-8").read():
    fail("re-verify failed")
print("FIRSTTOK PATCH: APPLIED")
