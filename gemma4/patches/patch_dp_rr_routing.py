"""Round-robin DP request routing (qwen3 item 9).

WHY. DPScheduler._find_best_rank_for_request probes EVERY rank per new
request: with prefix caching a PROBE_COMPUTED_BLOCKS round-trip to all ranks,
then pending/inflight/min-remaining queries to all ranks -- ~100 pipe
messages per request. At 2048 requests x dp64 that serializes on the head
loop (~13 s/step measured at dp32 on qwen3). Our GRPO workload gives every
rank identical work (256 prompt groups x 8 identical-length generations), so
load probing buys nothing.

WHAT. When DP_ROUTING=rr, _route_and_forward_request assigns ranks by a
plain counter and never calls _find_best_rank_for_request. Group locality is
preserved automatically (consecutive requests spread evenly). Default (env
unset) keeps stock behavior.

FAIL-OPEN. Prints exactly one of:
    DP-RR-ROUTING PATCH: APPLIED
    DP-RR-ROUTING PATCH: FAILED <reason>
"""
import ast
import os
import sys

DS = "/usr/local/lib/python3.12/site-packages/tpu_inference/core/sched/dp_scheduler.py"


def fail(msg):
    print(f"DP-RR-ROUTING PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(DS):
    fail(f"{DS} not found")
src = open(DS, encoding="utf-8").read()
if "_rr_next" in src:
    print("DP-RR-ROUTING PATCH: already patched")
    sys.exit(0)

ANCHOR = "        rank = self._find_best_rank_for_request(request)\n"
if src.count(ANCHOR) != 1:
    fail(f"route anchor count {src.count(ANCHOR)}")
REP = ("        if os.environ.get(\"DP_ROUTING\", \"\") == \"rr\":\n"
       "            rank = getattr(self, \"_rr_next\", 0) % self.dp_size\n"
       "            self._rr_next = rank + 1\n"
       "        else:\n"
       "            rank = self._find_best_rank_for_request(request)\n")
src = src.replace(ANCHOR, REP, 1)

if "\nimport os\n" not in src and not src.startswith("import os\n"):
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            lines.insert(i, "import os")
            break
    src = "\n".join(lines)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(DS, "w", encoding="utf-8").write(src)
v = open(DS, encoding="utf-8").read()
if "_rr_next" not in v or "import os" not in v:
    fail("re-verify failed")
print(f"DP-RR-ROUTING PATCH: APPLIED (DP_ROUTING={os.environ.get('DP_ROUTING','<unset>')})")
