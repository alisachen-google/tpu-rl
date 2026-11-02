#!/bin/bash
# early-stop.sh <name> <baseline_median_s> [tolerance=1.15] [min_steps=3]
# Deletes the JobSet early when: (a) the median of timed updates (>=2) exceeds baseline*tolerance after min_steps
# timed updates (perf regression), (b) the log has not grown for 15 min after startup (hang), or (c) a Traceback /
# non-zero EXIT_CODE appears (crash). Logs the decision to run-after.log.
name=$1; base=$2; tol=${3:-1.15}; minn=${4:-3}; SP=$(dirname "$(readlink -f "$0")")
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
K="kubectl --context gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap -n ubench-regression-tests"
L="$SP/$name-raw.log"; log(){ echo "[$(date -u +%H:%M:%S)] early-stop $name: $*" >> "$SP/run-after.log"; }
stop(){ log "STOP: $*"; $K delete jobset "$name" --wait=false >/dev/null 2>&1; exit 0; }
log "armed (baseline ${base}s x${tol}, after ${minn} timed updates; hang 15 min; crash)"
while :; do
  $K get jobset "$name" >/dev/null 2>&1 || { log "jobset gone -> exit"; exit 0; }
  grep -aq "EXIT_CODE=0" "$L" 2>/dev/null && { log "completed normally"; exit 0; }
  grep -aqE "Traceback|EXIT_CODE=[1-9]" "$L" 2>/dev/null && stop "crash: $(grep -aE 'Error|EXIT_CODE=[1-9]' "$L" | tail -1 | cut -c1-120)"
  mapfile -t T < <(grep -a 'PHASETIME step=' "$L" 2>/dev/null | sort -u | grep -oE 'step=[0-9]+ .*run_global_step=[0-9.]+' | awk -F'[= ]' '$2>=2{print $NF}' | sort -n)
  n=${#T[@]}
  if [ "$n" -ge "$minn" ]; then
    med=${T[$(( (n-1)/2 ))]}
    awk -v m="$med" -v b="$base" -v t="$tol" 'BEGIN{exit !(m > b*t)}' && stop "perf regression: median of $n timed updates ${med}s > ${base}s x${tol}"
  fi
  if grep -aq "PHASETIME step=0" "$L" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -L -c %Y "$L") )); [ $age -gt 900 ] && stop "hang: no log output for ${age}s"
  fi
  sleep 30
done
