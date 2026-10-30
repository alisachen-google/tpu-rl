#!/bin/bash
# Sequential 3-seed gate chain with RESUME: g4pt1 (seed 42) -> g4pt2 (43) -> g4pt3 (44).
# Rules: ONE 64-chip jobset at a time. The cluster's time-limit-controller DEACTIVATES every
# ubench-regression-tests workload 120 min after PodsReady (per admission); a 250-step run needs ~2.9 h,
# so each seed needs one resume: when the workload is deactivated (spec.active=false), delete the JobSet
# and re-apply the same manifest -> same run_name -> tunix restores the latest checkpoint (every 20 steps,
# patcher edit I) and continues to step 250. A JobSet "Completed" (train_steps reached 250) ends the seed.
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
NS=ubench-regression-tests
SP=/tmp/claude-731364623/-home-alisachen-google-com/a6b0b610-ff1b-40c9-8ca7-d4e8bacbb24a/scratchpad
OUT="$SP/chain-seeds.log"
MAX_RESUMES=${MAX_RESUMES:-4}
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
K(){ kubectl --context $CTX -n $NS "$@"; }

state(){  # prints: absent | completed | failed | deactivated | running | pending | unknown
  local js rc err wl
  err=$(mktemp); js=$(K get jobset "$1" -o jsonpath='{.spec.suspend}|{range .status.conditions[*]}{.type}={.status};{end}' 2>"$err"); rc=$?
  if [ $rc -ne 0 ]; then
    if grep -qiE 'NotFound|not found' "$err"; then rm -f "$err"; echo absent; return; fi
    rm -f "$err"; echo unknown; return
  fi
  rm -f "$err"; [ -z "$js" ] && { echo unknown; return; }
  echo "$js" | grep -qE '(^|[|;])Completed=True' && { echo completed; return; }
  echo "$js" | grep -qE '(^|[|;])(Failed)=True' && { echo failed; return; }
  # workload deactivated by the time-limit-controller (or any Deactivated eviction) -> resume
  wl=$(K get workloads -o json 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('err'); sys.exit()
for w in d['items']:
    if w['metadata']['name'].startswith('jobset-$1-'):
        act=(w.get('spec') or {}).get('active', True)
        ev=[c for c in w['status'].get('conditions',[]) if c['type']=='Evicted' and c['status']=='True']
        print('deactivated' if (act is False or any(c.get('reason')=='Deactivated' for c in ev)) else 'ok'); sys.exit()
print('nowl')" 2>/dev/null)
  [ "$wl" = "deactivated" ] && { echo deactivated; return; }
  [ "${js%%|*}" = "true" ] && { echo pending; return; }
  echo running
}

resume(){  # $1 = name: delete the JobSet and re-apply the same manifest (run_name unchanged -> checkpoint resume)
  local name=$1
  log "$name RESUME: deleting JobSet and re-applying $name.yaml (latest ckpt: $(env -u GOOGLE_APPLICATION_CREDENTIALS gsutil ls gs://ubench-logs/$name/checkpoints/actor/ 2>/dev/null | grep -oE '/[0-9]+/$' | tr -d / | sort -n | tail -1))"
  K delete jobset "$name" --wait=true --timeout=180s >/dev/null 2>&1
  for i in $(seq 1 40); do K get jobset "$name" >/dev/null 2>&1 || break; sleep 3; done
  # a failed async save leaves an uncommitted step dir that orbax would remove sequentially (hours, trainer hangs)
  /mnt/disks/scratch/xpk-venv/bin/python3 "$SP/clean-uncommitted-ckpts.py" "$name" 2>/dev/null | while read -r l; do log "  $l"; done
  bash "$SP/apply-jobset.sh" "$SP/$name.yaml" 2>&1 | tail -2 | while read -r l; do log "  $l"; done
}

stream_until_done(){  # $1 = jobset name; streams head-pod log; resumes on deactivation; returns on completed/failed
  local name=$1 raw="$SP/$1-raw.log" seen="$SP/$1-chain-seen.txt" st pod resumes=0
  : > "$seen"
  while :; do
    st=$(state "$name")
    case "$st" in
      completed|failed) log "$name TERMINAL state=$st"; return;;
      absent) log "$name ABSENT (deleted externally) -- holding, no actions"; sleep 300; continue;;
      unknown) log "$name state=unknown (kubectl/auth failure) -- holding, no actions"; sleep 300; continue;;
      deactivated)
        if [ $resumes -ge $MAX_RESUMES ]; then log "$name deactivated but MAX_RESUMES=$MAX_RESUMES reached -- stopping"; return; fi
        resumes=$((resumes+1)); resume "$name"; sleep 60; continue;;
    esac
    # hung-trainer guard: the head pod is Running but the log has not advanced a train step in 25 min -> resume
    if [ -s "$raw" ]; then
      last_ts=$(grep -a "Train step [0-9]* training loss" "$raw" | tail -1 | grep -oE '^I[0-9]{4} [0-9:]{8}' | awk '{print $2}')
      if [ -n "$last_ts" ]; then
        now_s=$(date -u +%s); last_s=$(date -u -d "$(date -u +%Y-%m-%d) $last_ts" +%s 2>/dev/null || echo $now_s)
        [ $((now_s - last_s)) -lt 0 ] && last_s=$((last_s - 86400))
        if [ $((now_s - last_s)) -gt 1500 ] && [ "$(state "$name")" = "running" ] && [ $resumes -lt $MAX_RESUMES ]; then
          log "$name HUNG: no train step for $(( (now_s - last_s) / 60 )) min -> resume"; resumes=$((resumes+1)); resume "$name"; : > "$raw"; sleep 60; continue
        fi
      fi
    fi
    pod=$(K get pods 2>/dev/null | awk "/${name}-pathways-head/{print \$1}" | head -1)
    if [ -n "$pod" ] && [ "$(K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; then
      timeout 600 kubectl --context $CTX -n $NS logs -f "$pod" -c jax-tpu --since=15m >> "$raw" 2>&1
      grep -aoE 'TIS-PARITY PATCH: (APPLIED.{0,40}|FAILED.{0,80})|EVAL-V2ACC total=[0-9]+ v2_acc=[0-9.]+% mean_reward=[0-9.]+|EVAL-BREAKDOWN step=[0-9]+ k=[0-9]+ total=[0-9]+ acc=[0-9.]+%|Restored params from step: [0-9]+|Found [0-9]+ checkpoint steps|Failed to restore from step.{0,20}|Finished saving checkpoint.{0,80}|RESOURCE_EXHAUSTED.{0,60}|Address already in use' "$raw" 2>/dev/null \
        | while read -r l; do grep -qxF "$l" "$seen" || { echo "$l" >> "$seen"; log "$name | $l"; }; done
    else
      sleep 120
    fi
  done
}

log "chain3 start: g4pt1 state=$(state g4pt1)"
for name in ${NAMES:-$(for i in ${SEEDS:-1 2 3}; do echo g4pt$i; done)}; do
  st=$(state "$name")
  if [ "$st" = "completed" ] || [ "$st" = "failed" ]; then log "$name already $st; skipping"; continue; fi
  if [ "$st" = "absent" ]; then
    log "applying $name.yaml"
    bash "$SP/apply-jobset.sh" "$SP/$name.yaml" 2>&1 | tail -2 | while read -r l; do log "  $l"; done
    : > "$SP/$name-raw.log"
  else
    log "$name present (state=$st); watching"
  fi
  stream_until_done "$name"
done
log "CHAIN3 DONE: $(for n in g4pt1 g4pt2 g4pt3; do printf '%s=%s ' $n "$(state $n)"; done)"
