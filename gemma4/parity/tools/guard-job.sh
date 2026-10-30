#!/bin/bash
# guard-job.sh <name> <manifest> : re-apply a JobSet (fresh admission) when its re-admission is stuck
# (head Running but a worker Pending > 10 min: expired ProvisioningRequest booking after node failures), or when
# it is admitted+ready but produced no train step for 30 min. Stops when the JobSet completes or disappears >15 min.
name=$1; yaml=$2; SP=$(dirname "$(readlink -f "$0")"); export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap; NS=ubench-regression-tests; K="kubectl --context $CTX -n $NS"
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$SP/guard-$name.log"; }
log "guarding $name"; pend_since=""; absent=0; reapplies=0
reapply(){ log "re-applying $name (reason: $1)"; $K delete jobset "$name" --wait=true --timeout=180s >/dev/null 2>&1; for i in $(seq 1 40); do $K get jobset "$name" >/dev/null 2>&1 || break; sleep 3; done; bash "$SP/apply-jobset.sh" "$SP/$yaml" >> "$SP/guard-$name.log" 2>&1; reapplies=$((reapplies+1)); pend_since=""; sleep 120; }
while :; do
  js=$($K get jobset "$name" -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then absent=$((absent+1)); [ $absent -gt 7 ] && { log "absent >15 min -> stop"; exit 0; }; sleep 120; continue; fi; absent=0
  echo "$js" | grep -qE '(^|[|;])(Completed|Failed)=True' && { log "terminal: $js -> stop"; exit 0; }
  [ $reapplies -ge 4 ] && { log "4 re-applies done -> stop guarding"; exit 0; }
  head=$($K get pods 2>/dev/null | awk "/${name}-pathways-head/{print \$3}" | head -1)
  npend=$($K get pods 2>/dev/null | awk "/${name}-worker/ && \$3==\"Pending\"" | wc -l)
  if [ "$head" = "Running" ] && [ "$npend" -gt 0 ]; then
    [ -z "$pend_since" ] && pend_since=$(date +%s)
    if [ $(( $(date +%s) - pend_since )) -gt 600 ]; then reapply "worker Pending >10 min with head Running"; continue; fi
  else pend_since=""; fi
  last=$(grep -a "Train step [0-9]* training" "$SP/$name-raw.log" 2>/dev/null | tail -1); t_last=$(stat -L -c %Y "$SP/$name-raw.log" 2>/dev/null || date +%s)
  if [ "$head" = "Running" ] && [ "$npend" -eq 0 ] && [ $(( $(date +%s) - t_last )) -gt 1800 ]; then reapply "no log progress for 30 min"; continue; fi
  sleep 120
done
