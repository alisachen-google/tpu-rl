"""Wire tunix perf-metrics v2 (Perfetto timeline export) -- run 15.

maxtext's train_rl populates only custom_export_fn (v1 -> the TensorBoard
perf/* metrics we already use). The v2 path (tunix.perf.experimental)
writes a Perfetto timeline of every phase span (rollout / weight_sync /
train, tagged per step) viewable at ui.perfetto.dev. At the pinned tunix
commit (683256db) rl_cluster consumes custom_export_fn_v2, the GRPO
learner is instrumented, and rl_learner calls perf_v2.export() every
iteration -- only the config field is never populated.

We patch tunix's rl_cluster.py (byte-verified against the pin) rather than
maxtext's train_rl.py (whose image content is date-approximated): default
the v2 export fn at the point where cluster_config is in scope. Best-effort
by construction -- any init failure prints PERF-V2: DISABLED and the run
proceeds with v2 off.

Output: ONE file gs://ubench-logs/tianyu-rl-17a/perf_traces/
perfetto_trace_v2_<epoch>.pb, rewritten each step with all steps
accumulated (writes serialized on a 1-thread executor; epath/gcsfs handle
the gs:// path; perfetto+etils+gcsfs verified in the image layer).

Prints exactly one of:
    PERF-V2 PATCH: APPLIED
    PERF-V2 PATCH: already patched
    PERF-V2 PATCH: FAILED <reason>
"""

import ast
import os
import sys

P = "/usr/local/lib/python3.12/site-packages/tunix/rl/rl_cluster.py"


def fail(msg):
    print(f"PERF-V2 PATCH: FAILED {msg}")
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


if not os.path.exists(P):
    fail(f"{P} not found")
src = open(P, encoding="utf-8").read()

if "_PERF_V2_DEFAULT_WIRED" in src:
    print("PERF-V2 PATCH: already patched")
    sys.exit(0)

ANCHOR = "      export_fn_v2 = self.perf_config.custom_export_fn_v2\n"
if src.count(ANCHOR) != 1:
    fail(f"anchor found {src.count(ANCHOR)} times, expected 1")

INJECT = ANCHOR + (
    "\n"
    "      # _PERF_V2_DEFAULT_WIRED (tianyu-rl-17a). maxtext train_rl populates\n"
    "      # only the v1 export fn; default the v2 (Perfetto) exporter here,\n"
    "      # where cluster_config is in scope. Best-effort: on any failure v2\n"
    "      # stays off and the run proceeds.\n"
    "      if export_fn_v2 is None:\n"
    "        try:\n"
    "          from tunix.perf.experimental import export as _pv2_export\n"
    "          export_fn_v2 = _pv2_export.PerfMetricsExport.from_cluster_config(\n"
    "              cluster_config=self.cluster_config,\n"
    "              trace_dir=\"gs://ubench-logs/tianyu-rl-17a/perf_traces\",\n"
    "          ).export_metrics\n"
    "          print(\"PERF-V2: perfetto export enabled -> \"\n"
    "                \"gs://ubench-logs/tianyu-rl-17a/perf_traces\")\n"
    "        except Exception as _pv2_e:  # pylint: disable=broad-except\n"
    "          print(\"PERF-V2: DISABLED (init failed):\", _pv2_e)\n"
    "          export_fn_v2 = None\n"
)

src = src.replace(ANCHOR, INJECT, 1)
try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(P, "w", encoding="utf-8").write(src)
print("PERF-V2 PATCH: APPLIED")
