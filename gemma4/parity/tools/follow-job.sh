#!/bin/bash
# follow-job.sh <name>: append the head pod's jax-tpu stdout to <name>-raw.log until the JobSet is terminal or gone.
name=$1; SP=$(dirname "$(readlink -f "$0")"); export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap; NS=ubench-regression-tests; K="kubectl --context $CTX -n $NS"
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$SP/follow-$name.log"; }
log "following $name"; : > "$SP/$name-raw.log"; missing=0
while :; do
  js=$($K get jobset "$name" -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then missing=$((missing+1)); [ $missing -gt 20 ] && { log "$name absent for 10 min -> stop"; exit 0; }; sleep 30; continue; fi; missing=0
  echo "$js" | grep -qE '(^|[|;])(Completed|Failed)=True' && { log "$name terminal: $js"; exit 0; }
  pod=$($K get pods 2>/dev/null | awk "/${name}-pathways-head/{print \$1}" | head -1)
  if [ -n "$pod" ] && [ "$($K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; then
    timeout 600 kubectl --context $CTX -n $NS logs -f "$pod" -c jax-tpu --since=15m >> "$SP/$name-raw.log" 2>&1
  else sleep 30; fi
done
