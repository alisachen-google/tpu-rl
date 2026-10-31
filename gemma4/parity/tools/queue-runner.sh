#!/bin/bash
# queue-runner.sh <name1> <name2> ...: wait until kubectl works (credentials), then run each <name>.yaml in turn
# (apply via apply-jobset.sh, follow + guard, wait for Completed/Failed, delete), one 64-chip job at a time.
SP=$(dirname "$(readlink -f "$0")"); export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap; NS=ubench-regression-tests; K="kubectl --context $CTX -n $NS"
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$SP/run-after.log"; }
log "queue-runner: $* (waiting for working credentials)"
until $K get jobsets >/dev/null 2>&1; do sleep 60; done
log "queue-runner: credentials OK"
for name in "$@"; do
  while $K get jobsets 2>/dev/null | grep -qE "^(g4pt|g4l|g4m|g4u|g4p)[a-z0-9]* "; do sleep 60; done   # one of ours at a time
  log "queue-runner: launching $name"; bash "$SP/apply-jobset.sh" "$SP/$name.yaml" >> "$SP/run-after.log" 2>&1
  (nohup bash "$SP/follow-job.sh" "$name" > /dev/null 2>&1 < /dev/null &); (nohup bash "$SP/guard-job.sh" "$name" "$name.yaml" > /dev/null 2>&1 < /dev/null &)
  t0=$(date +%s)
  while :; do
    js=$($K get jobset "$name" -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null)
    echo "$js" | grep -qE '(^|[|;])(Completed|Failed)=True' && { log "$name terminal: $js"; break; }
    grep -aq "EXIT_CODE=0" "$SP/$name-raw.log" 2>/dev/null && { log "$name exited 0 (log)"; sleep 60; break; }
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && { log "$name exceeded 120 min -> stop waiting"; break; }
    sleep 60
  done
  pkill -f "^bash guard-job.sh $name " 2>/dev/null; pkill -f "^bash follow-job.sh $name" 2>/dev/null
  log "$name steady step times (updates >= 2): $(grep -a 'PHASETIME step=' "$SP/$name-raw.log" | sort -u | grep -oE 'step=[0-9]+ .*run_global_step=[0-9.]+' | awk -F'[= ]' '$2>=2{print $NF}' | sort -n | awk '{a[NR]=$1} END{if(NR) printf "n=%d median=%.2f min=%.2f max=%.2f", NR, a[int((NR+1)/2)], a[1], a[NR]}')"
  $K delete jobset "$name" --wait=true --timeout=180s >/dev/null 2>&1; for i in $(seq 1 40); do $K get jobset "$name" >/dev/null 2>&1 || break; sleep 3; done
done
log "queue-runner done"
