"""Enable vLLM engine stats logging (KV-cache usage gauge) -- run 15b.

CORRECTED from run 15, where the patch flipped the setdefault in
_build_engine_args -- dead code on our path: with server_mode=False tunix
constructs the OFFLINE engine, `LLM(**self.args)`, and vLLM's LLM.__init__
defaults disable_log_stats=True when the key is absent (llm.py:224 at pin
0ba2aa35a). Run 15 therefore produced zero stats lines (banner APPLIED,
effect nil).

Fix: setdefault the key into `self.args` at its single construction point,
_vllm_config(). Covers BOTH paths -- LLM(**args) sees the key (vLLM's
"not in kwargs" guard skips), and _build_engine_args copies self.args so
its own setdefault becomes a no-op.

Deliverable: "GPU KV cache usage: X%" lines in pathways-head logs during
rollout, to size the TP=4 experiment. Reading caveats: CD lookahead
allocation books up to 128 steps of blocks per window up front, so the
gauge reads structurally high; throughput numbers are meaningless under
fused CD windows. Startup-line context from run 15 (identical config):
pool = 34,325,568 tokens at hbm_utilization=0.3; max concurrency 2062.83x
for 16,640-token requests, i.e. worst case barely fits at TP8.

Prints exactly one of:
    LOG-STATS PATCH: APPLIED
    LOG-STATS PATCH: already patched
    LOG-STATS PATCH: FAILED <reason>
"""

import ast
import os
import sys

P = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"


def fail(msg):
    print(f"LOG-STATS PATCH: FAILED {msg}")
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


if not os.path.exists(P):
    fail(f"{P} not found")
src = open(P, encoding="utf-8").read()

OLD = "    args = config._processed_engine_kwargs.copy()"
NEW = (
    "    args = config._processed_engine_kwargs.copy()\n"
    '    args.setdefault("disable_log_stats", False)  # run-15b: stats ON (offline LLM path)'
)

if 'args.setdefault("disable_log_stats", False)' in src:
    print("LOG-STATS PATCH: already patched")
    sys.exit(0)
if src.count(OLD) != 1:
    fail(f"anchor found {src.count(OLD)} times, expected 1")

src = src.replace(OLD, NEW, 1)
try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(P, "w", encoding="utf-8").write(src)
print("LOG-STATS PATCH: APPLIED")
