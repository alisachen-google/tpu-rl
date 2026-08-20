#!/bin/bash
# Follow one run: Kueue admission, pod counts, and the log lines that matter.
# Usage: tools/watch_run.sh [run-name]   (default: last run launched by launch_run.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
K=${KB:-$HOME/bin/kb}
RUN=${1:-$(cat renders/.current_4x4x4)}
last=""; seen=""
while true; do
  S=$("$K" get jobset "$RUN" -o jsonpath='{.spec.suspend}' 2>/dev/null) || true
  [ -z "$S" ] && { echo "[$RUN] jobset gone"; exit 0; }
  ev=$("$K" get events -o jsonpath='{range .items[*]}{.reason}{"|"}{.message}{"\n"}{end}' 2>/dev/null \
        | grep -E "jobset-$RUN" | tail -1 | cut -c1-200)
  [ -n "$ev" ] && [ "$ev" != "$seen" ] && { echo "[$RUN] kueue: $ev"; seen="$ev"; }
  HP=$("$K" get pods --no-headers 2>/dev/null | awk "/$RUN-pathways-head/{print \$3}" | head -1)
  WRUN=$("$K" get pods --no-headers 2>/dev/null | awk "/$RUN-worker/{print \$3}" | grep -c Running)
  RS=$("$K" get jobset "$RUN" -o jsonpath='{.status.restarts}' 2>/dev/null)
  cur="suspended=$S restarts=${RS:-0} head=$HP workers=$WRUN"
  [ "$cur" != "$last" ] && { echo "[$RUN] $cur"; last="$cur"; }
  if [ "$HP" = "Running" ]; then
    POD=$("$K" get pods --no-headers | awk "/$RUN-pathways-head/{print \$1}" | head -1)
    "$K" logs -f "$POD" -c jax-tpu --since=60s 2>&1 | grep -E --line-buffered \
      "PATCH:|TUNING:|LENDIST step|PHASETIME step|math_verify_pool: Processed|Train step [0-9]|RL Training Completed|Job Ended|DEBUG-HOLD|Traceback|RESOURCE_EXHAUSTED|Cache key mismatch" \
      | sed -u "s/^/[$RUN] /" | cut -c1-220
  fi
  sleep 45
done
