"""Hotfix for pathwaysutils profiling + tpu_runner req_id_kwargs nanobind limit.

Adapted for image head-612fcd2e-gemma4c-perf. Two of the four original
sub-fixes are already merged upstream in this image and are DETECTED, not
re-applied:
  - profiling.py executable.call(out_avals=..., out_shardings=...): the image
    passes them conditionally (xprofTraceOptions branch) -- already fixed.
  - profiling.py advanced_config int -> {"int64Value": v}: already present.
Still applied here:
  - profiling.py start_trace / start_trace_patch: max_num_hosts default 1 ->
    None, resolved to 16 inside start_trace.
  - tpu_runner.py: cap **req_id_kwargs at 100 entries (nanobind kwargs limit).

Fail-open: prints exactly one of
    VLLM-SAMPLER PATCH: APPLIED
    VLLM-SAMPLER PATCH: already patched
    VLLM-SAMPLER PATCH: FAILED <reason>
"""

import os
import sys

SITE = os.environ.get("PATCH_ROOT", "/usr/local/lib/python3.12/site-packages")
PROF = os.path.join(SITE, "pathwaysutils/profiling.py")
TR = os.path.join(SITE, "tpu_inference/runner/tpu_runner.py")


def fail(msg):
    print(f"VLLM-SAMPLER PATCH: FAILED {msg}")
    sys.exit(0)  # non-fatal: let the run start so the failure is visible


problems = []
applied_any = False
already_all = True

# --- pathwaysutils/profiling.py ---------------------------------------------
if not os.path.exists(PROF):
    fail(f"{PROF} not found")
with open(PROF, "r", encoding="utf-8") as f:
    c4 = f.read()
modified4 = False

# (1) executable.call out_avals/out_shardings -- merged upstream in this image
# (conditional xprofTraceOptions branch). Detect either form.
target4 = "      _, result_future = _profile_state.executable.call()"
if target4 in c4:
    c4 = c4.replace(target4, (
        "      import jax\n"
        "      import jax.numpy as jnp\n"
        "      out_avals = ()\n"
        "      out_shardings = ()\n"
        "      _, result_future = _profile_state.executable.call(\n"
        "          out_avals=out_avals, out_shardings=out_shardings\n"
        "      )"
    ))
    modified4 = True
elif "out_avals=out_avals, out_shardings=out_shardings" not in c4:
    problems.append("profiling.py: executable.call anchor missing and "
                    "out_avals fix not present")

# (2) int -> int64Value -- merged upstream in this image.
target4_int = ("      elif isinstance(v, int):\n"
               '        advanced_config[k] = {"intValue": v}')
if target4_int in c4:
    c4 = c4.replace(target4_int,
                    "      elif isinstance(v, int):\n"
                    '        advanced_config[k] = {"int64Value": v}')
    modified4 = True
elif '{"int64Value": v}' not in c4:
    problems.append("profiling.py: intValue anchor missing and int64Value "
                    "fix not present")

# traceSessionName is enabled now; no replacement needed (as in original).

# (3) start_trace: max_num_hosts default 1 -> None (resolved to 16).
target4_start = (
    "def start_trace(\n"
    "    log_dir: os.PathLike[str] | str,\n"
    "    *,\n"
    "    create_perfetto_link: bool = False,\n"
    "    create_perfetto_trace: bool = False,\n"
    "    profiler_options: jax.profiler.ProfileOptions | None = None,\n"
    "    max_num_hosts: int = 1,\n"
    ") -> None:"
)
replacement4_start = (
    "def start_trace(\n"
    "    log_dir: os.PathLike[str] | str,\n"
    "    *,\n"
    "    create_perfetto_link: bool = False,\n"
    "    create_perfetto_trace: bool = False,\n"
    "    profiler_options: jax.profiler.ProfileOptions | None = None,\n"
    "    max_num_hosts: int | None = None,\n"
    ") -> None:\n"
    "  if max_num_hosts is None:\n"
    "    max_num_hosts = 16"
)
if target4_start in c4:
    c4 = c4.replace(target4_start, replacement4_start)
    modified4 = True
    already_all = False
elif "  if max_num_hosts is None:\n    max_num_hosts = 16" not in c4:
    problems.append("profiling.py: start_trace signature anchor missing and "
                    "max_num_hosts fix not present")

# (4) start_trace_patch: same default change on the monkeypatch wrapper.
target4_monkey = (
    "  def start_trace_patch(\n"
    "      log_dir,\n"
    "      create_perfetto_link: bool = False,\n"
    "      create_perfetto_trace: bool = False,\n"
    "      profiler_options: jax.profiler.ProfileOptions | None = None,\n"
    "      max_num_hosts: int = 1,\n"
    "  ) -> None:"
)
replacement4_monkey = (
    "  def start_trace_patch(\n"
    "      log_dir,\n"
    "      create_perfetto_link: bool = False,\n"
    "      create_perfetto_trace: bool = False,\n"
    "      profiler_options: jax.profiler.ProfileOptions | None = None,\n"
    "      max_num_hosts: int | None = None,\n"
    "  ) -> None:"
)
if target4_monkey in c4:
    c4 = c4.replace(target4_monkey, replacement4_monkey)
    modified4 = True
    already_all = False
elif ("  def start_trace_patch(\n"
      "      log_dir,\n"
      "      create_perfetto_link: bool = False,\n"
      "      create_perfetto_trace: bool = False,\n"
      "      profiler_options: jax.profiler.ProfileOptions | None = None,\n"
      "      max_num_hosts: int | None = None,\n") not in c4:
    problems.append("profiling.py: start_trace_patch anchor missing and "
                    "None-default fix not present")

if problems:
    fail("; ".join(problems))

if modified4:
    with open(PROF, "w", encoding="utf-8") as f:
        f.write(c4)
    applied_any = True

# --- tpu_inference/runner/tpu_runner.py --------------------------------------
if not os.path.exists(TR):
    fail(f"{TR} not found")
with open(TR, "r", encoding="utf-8") as f:
    c5 = f.read()

target5 = ", **req_id_kwargs):"
replacement5 = ", **dict(list(req_id_kwargs.items())[:100])):"
if target5 in c5:
    c5 = c5.replace(target5, replacement5)
    with open(TR, "w", encoding="utf-8") as f:
        f.write(c5)
    applied_any = True
    already_all = False
elif "dict(list(req_id_kwargs.items())" not in c5:
    fail("tpu_runner.py: req_id_kwargs anchor missing and cap not present")

# --- verify ------------------------------------------------------------------
import ast

for path in (PROF, TR):
    try:
        ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as e:
        fail(f"patched {path} does not parse: {e}")

v_prof = open(PROF, encoding="utf-8").read()
v_tr = open(TR, encoding="utf-8").read()
if "  if max_num_hosts is None:\n    max_num_hosts = 16" not in v_prof:
    fail("verify: start_trace max_num_hosts fix missing after write")
if "dict(list(req_id_kwargs.items())[:100])" not in v_tr:
    fail("verify: tpu_runner req_id_kwargs cap missing after write")

if applied_any or not already_all:
    print("VLLM-SAMPLER PATCH: APPLIED")
else:
    print("VLLM-SAMPLER PATCH: already patched")
