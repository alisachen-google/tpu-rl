#!/bin/bash
# Sequential 3-seed gate chain: g4pt1 (seed 42, already running) -> g4pt2 (43) -> g4pt3 (44).
# Rule: ONE 64-chip jobset at a time (concurrent jobsets self-preempt). Each next seed is applied
# only when the previous jobset reaches a terminal condition (Completed/Failed) or disappears.
# Streams each run's head-pod log to g4pt{N}-raw.log and emits evals + EVAL-V2ACC + verdicts to
# chain-seeds.log. Idempotent across session restarts: re-running skips seeds already terminal.
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
NS=ubench-regression-tests
SP=${PARITY_DIR:-$(dirname "$0")/..}
OUT="$SP/chain-seeds.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
K(){ kubectl --context $CTX -n $NS "$@"; }

state(){  # prints: absent | completed | failed | running | pending
  local js rc err
  err=$(mktemp); js=$(K get jobset "$1" -o jsonpath='{.spec.suspend}|{range .status.conditions[*]}{.type}={.status};{end}' 2>"$err"); rc=$?
  # auth/API failure (expired ADC, network) must NEVER look like "absent" -> would trigger an apply
  if [ $rc -ne 0 ]; then
    if grep -qiE 'NotFound|not found' "$err"; then rm -f "$err"; echo absent; return; fi
    rm -f "$err"; echo unknown; return
  fi
  rm -f "$err"; [ -z "$js" ] && { echo unknown; return; }
  # exact condition types only: "StartupPolicyCompleted=True" must NOT count as Completed
  echo "$js" | grep -qE '(^|[|;])Completed=True' && { echo completed; return; }
  echo "$js" | grep -qE '(^|[|;])(Failed|Evicted|Deactivated)=True' && { echo failed; return; }
  [ "${js%%|*}" = "true" ] && { echo pending; return; }
  echo running
}

stream_until_terminal(){  # $1=jobset name; streams head-pod log, exits when terminal
  local name=$1 raw="$SP/$1-raw.log" seen="$SP/$1-chain-seen.txt" st pod
  : > "$seen"
  while :; do
    st=$(state "$name")
    case "$st" in
      completed|failed|absent) log "$name TERMINAL state=$st"; break;;
      unknown) log "$name state=unknown (kubectl/auth failure) -- holding, no actions"; sleep 300; continue;;
    esac
    pod=$(K get pods 2>/dev/null | awk "/${name}-pathways-head/{print \$1}" | head -1)
    if [ -n "$pod" ] && [ "$(K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; then
      timeout 600 kubectl --context $CTX -n $NS logs -f "$pod" -c jax-tpu >> "$raw" 2>&1
      grep -aoE 'TIS-PARITY PATCH: (APPLIED.{0,60}|FAILED.{0,80})|EVAL-V2ACC total=[0-9]+ v2_acc=[0-9.]+% mean_reward=[0-9.]+|EVAL-BREAKDOWN k=[0-9]+ total=[0-9]+ acc=[0-9.]+% boxed_wrong=[0-9.]+% nobox=[0-9.]+% mean_chars=[0-9]+|\[eval-(v2acc|dump)\] failed.{0,100}|Pre RL Training:.{0,110}|Intermediate Eval \(step=[0-9]+\):.{0,110}|RESOURCE_EXHAUSTED.{0,60}|PreconditionFailed: 412.{0,40}|Failed to restore from step.{0,20}' "$raw" 2>/dev/null \
        | while read -r l; do grep -qxF "$l" "$seen" || { echo "$l" >> "$seen"; log "$name | $l"; }; done
    else
      sleep 120
    fi
  done
}

log "chain start: g4pt1 state=$(state g4pt1)"
# wait for the running seed-42 to finish
stream_until_terminal g4pt1

for i in 2 3; do
  name="g4pt$i"
  st=$(state "$name")
  if [ "$st" = "completed" ] || [ "$st" = "failed" ]; then log "$name already $st; skipping"; continue; fi
  if [ "$st" = "absent" ]; then
    log "applying $name.yaml (seed $((41+i)))"
    K apply --server-side --force-conflicts -f "$SP/$name.yaml" 2>&1 | tail -1 | while read -r l; do log "  $l"; done
    : > "$SP/$name-raw.log"
  else
    log "$name already present (state=$st); resuming watch"
  fi
  stream_until_terminal "$name"
done
log "CHAIN DONE: $(for n in g4pt1 g4pt2 g4pt3; do printf '%s=%s ' $n "$(state $n)"; done)"
for n in g4pt1 g4pt2 g4pt3; do
  log "$n step-250: $(grep -aoE 'Intermediate Eval \(step=250\):.{0,110}' "$SP/$n-raw.log" 2>/dev/null | head -1) | $(grep -aoE 'EVAL-V2ACC total=[0-9]+ v2_acc=[0-9.]+%' "$SP/$n-raw.log" 2>/dev/null | tail -1)"
done
