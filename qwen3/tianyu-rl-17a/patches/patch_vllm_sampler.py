"""Hotfix to disable prompt_logprobs in tunix vLLM Sampler and bypass vLLM V1 scheduler AssertionError."""

import os

# --- Patch 4: pathwaysutils profiling.py ---
print("Applying hotfix to pathwaysutils/profiling.py...")
paths4 = [
    "/usr/local/lib/python3.12/site-packages/pathwaysutils/profiling.py",
    "/app/pathways-utils/pathwaysutils/profiling.py",
]
for p4 in paths4:
  if os.path.exists(p4):
    print(f"Checking pathwaysutils file: {p4}")
    with open(p4, "r", encoding="utf-8") as f:
      c4 = f.read()

    target4 = "      _, result_future = _profile_state.executable.call()"
    replacement4 = (
        "      import jax\n"
        "      import jax.numpy as jnp\n"
        "      out_avals = ()\n"
        "      out_shardings = ()\n"
        "      _, result_future = _profile_state.executable.call(\n"
        "          out_avals=out_avals, out_shardings=out_shardings\n"
        "      )"
    )

    modified4 = False
    if target4 in c4:
      c4 = c4.replace(target4, replacement4)
      modified4 = True

    target4_int = "      elif isinstance(v, int):\n" '        advanced_config[k] = {"intValue": v}'
    replacement4_int = "      elif isinstance(v, int):\n" '        advanced_config[k] = {"int64Value": v}'
    if target4_int in c4:
      c4 = c4.replace(target4_int, replacement4_int)
      modified4 = True

    # traceSessionName is enabled now, bypassing replacement:
    print("Retaining traceSessionName configuration in pathwaysutils.")

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

    if modified4:
      with open(p4, "w", encoding="utf-8") as f:
        f.write(c4)
      print(f"Hotfix successfully applied to {p4}.")
    else:
      print(f"Warning: target strings not found or already patched in {p4}.")
  else:
    print(f"Warning: {p4} not found.")

# --- Patch 5: tpu_inference tpu_runner.py ---
print("Applying hotfix to tpu_inference/runner/tpu_runner.py...")
p5 = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py"
if os.path.exists(p5):
  with open(p5, "r", encoding="utf-8") as f:
    c5 = f.read()

  # Simpler replacement to limit the size of req_id_kwargs to avoid nanobind limit
  target5 = ", **req_id_kwargs):"
  replacement5 = ", **dict(list(req_id_kwargs.items())[:100])):"

  if target5 in c5:
    c5 = c5.replace(target5, replacement5)
    with open(p5, "w", encoding="utf-8") as f:
      f.write(c5)
    print("Hotfix successfully applied to tpu_inference/runner/tpu_runner.py (limited req_id_kwargs).")
  else:
    if "dict(list(req_id_kwargs.items())" in c5:
      print("Hotfix already applied to tpu_inference/runner/tpu_runner.py.")
    else:
      print("Warning: target string not found in tpu_runner.py.")
else:
  print("Warning: tpu_runner.py path not found.")
