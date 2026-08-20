#!/bin/bash
# Render + submit one gemma4-e2b RL run. Wraps the helm-template | kubectl-apply flow
# (our K8s SA cannot create secrets, so `helm install` is not available).
#
# Usage:
#   tools/launch_run.sh <profile> [extra --set args...]
#
# Profiles:
#   r1n-4x4x4    native route + tianyu stack, 4x4x4, 30 steps        (control benchmark)
#   r1np-4x4x4   as above + maxtext PR#4409 chunked logps, tmbs=2    (PR rung)
#   r1-smoke     native route + stack, 2x2x1, 2 steps                (pre-flight)
#   r1-smoke-pr  PR#4409 rung at 2x2x1, tmbs=2 / chunk 512           (pre-flight)
#
# Prints the run name; artifacts land in gs://ubench-logs/alisachen-tpu7x/runs/<run>/.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${KB:-$HOME/bin/kb}
PROFILE=${1:?usage: launch_run.sh <profile> [--set k=v ...]}; shift || true

case "$PROFILE" in
  r1n-4x4x4)
    VALUES=(-f chart/values.yaml -f chart/values-4x4x4.yaml -f chart/values-dl.yaml -f chart/values-r1n-4x4x4.yaml)
    LAUNCHER=gemma4/gemma4_e2b_r1_native_launcher.sh; PREFIX=g4r1n ;;
  r1np-4x4x4)
    VALUES=(-f chart/values.yaml -f chart/values-4x4x4.yaml -f chart/values-dl.yaml -f chart/values-r1np-4x4x4.yaml)
    LAUNCHER=gemma4/gemma4_e2b_r1_native_pr_launcher.sh; PREFIX=g4r1np ;;
  r1-smoke)
    VALUES=(-f chart/values.yaml -f chart/values-r1-smoke-tpfix.yaml)
    LAUNCHER=gemma4/gemma4_e2b_r1_native_launcher.sh; PREFIX=g4ns ;;
  r1-smoke-pr)
    VALUES=(-f chart/values.yaml -f chart/values-r1-smoke-pr3.yaml)
    LAUNCHER=gemma4/gemma4_e2b_r1_native_pr_launcher.sh; PREFIX=g4nps ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

RUN=$PREFIX-$(date +%m%d%H%M)
# queue/priority: `multislice-queue` + `medium` are the values that exist on
# bodaborg-tpu7x-nap today; `high`/`very-high` were deleted (RUNBOOK Fix #3 churn).
# Ports are shifted off the 2900x defaults so a foreign pathways head on the same
# cpu-np node cannot collide with ours (Fix #12/#22).
helm template "$RUN" chart "${VALUES[@]}" \
  --set queue=multislice-queue --set kueue_priority_class=medium \
  --set pathways.proxyPort=29100 --set pathways.rmPort=29101 --set pathways.workerPort=29105 \
  --set-file workload_launcher="$LAUNCHER" \
  --set-file workload_config=chart/config.yml \
  "$@" > "renders/$RUN.yaml"

"$K" apply -f "renders/$RUN.yaml" >/dev/null
echo "$RUN" > renders/.current_4x4x4
echo "$RUN"
