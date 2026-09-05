# Patch & knob catalog for the gemma4-E2B perf ladder

Every optimization in `../GEMMA4_PERF_LADDER_COMPLETE.md` ships as one of:
- a **runtime patch** here (`patch_*.py`) — applied at pod start, self-verifying, fail-open,
  prints `X PATCH: APPLIED|FAILED`. Targets the pinned image's tpu-inference / tunix / maxtext.
- an **env knob** (`DL_*`) read by `launcher_with_knobs.sh` into a MaxText config flag.

The ladder doc's "Patch / knob" column names the exact file/knob for each rung.
`gemma_data.py` (gemma-3-style) and `gemma_data_canonical.py` (canonical `<|turn>` frame)
are the two prompt builders. Applied in the roster order shown in each run's launcher configmap.
