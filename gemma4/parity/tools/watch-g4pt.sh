#!/bin/bash
# Monitor the g4pt1 TIS-parity run (standard offline path + token TIS, 5120 cap, seed 42).
# Waits through the pending/admission phase, then streams logs and EXITS (notifying the
# session) at the decision point: step-20 captured, a hard crash/OOM, or Kueue deactivation.
# Emits decisive lines to g4pt1-verdict.log; full stream to g4pt1-raw.log.
#
# What to verify from the verdict log:
#   - "TIS-PARITY PATCH: APPLIED (all)"            -> patch applied cleanly
#   - "sampler_is: weight_mean=.. weight_min=.."   -> TIS ACTIVE if weight_min < 1.0
#     (standard loss aux logs sampler_is/weight_min; == 1.0 exactly means TIS inert)
#   - "Pre RL Training: ... accuracy=.."           -> step-0 (vs GPU 80.31)
#   - "Intermediate Eval (step=20): ... accuracy=.."-> step-20 (THE decider; vs GPU 91.03)
#   - OOM / RESOURCE_EXHAUSTED / Traceback / restart-churn -> fall back to 4096 cap
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
NS=ubench-regression-tests
SP=${PARITY_DIR:-$(dirname "$0")/..}
RAW="$SP/g4pt1-raw.log"; OUT="$SP/g4pt1-verdict.log"; SEEN="$SP/g4pt1-seen.txt"
: > "$OUT"; : > "$SEEN"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
emit_new(){ grep -qxF "$1" "$SEEN" 2>/dev/null || { echo "$1" >> "$SEEN"; log "$1"; }; }

log "watching g4pt1 (TIS parity, cap 5120, seed 42); waiting for admission"
DEADLINE=$((SECONDS+86400))   # up to 24h total (run may pend for capacity)
admitted=0
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  # deactivation / restart-churn guard
  JS=$(kubectl --context $CTX -n $NS get jobset g4pt1 -o jsonpath='{.status.restarts}|{range .status.conditions[*]}{.type}={.status};{end}' 2>/dev/null)
  RESTARTS=${JS%%|*}
  if echo "$JS" | grep -qE 'Evicted=True|Deactivated'; then
    log "VERDICT: g4pt1 DEACTIVATED/evicted (restarts=$RESTARTS) -- investigate (likely OOM at cap 5120 -> fall back 4096)"; exit 0
  fi
  if [ -n "$RESTARTS" ] && [ "$RESTARTS" -ge 2 ] 2>/dev/null; then
    log "VERDICT: g4pt1 restart churn (restarts=$RESTARTS >= 2) -- likely cap OOM; stop + fall back 4096"; exit 0
  fi

  POD=$(kubectl --context $CTX -n $NS get pods 2>/dev/null | awk '/g4pt1-pathways-head/{print $1}' | head -1)
  ST=$(kubectl --context $CTX -n $NS get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ -z "$POD" ] || [ "$ST" != "Running" ]; then
    [ "$admitted" -eq 0 ] && emit_new "status: pending admission (tpu usage check); pod=${POD:-none} phase=${ST:-NA}"
    sleep 60; continue
  fi
  [ "$admitted" -eq 0 ] && { admitted=1; log "ADMITTED: $POD Running; streaming logs"; }

  timeout 500 kubectl --context $CTX -n $NS logs -f "$POD" -c jax-tpu >> "$RAW" 2>&1

  # decisive markers (dedup)
  grep -aoE 'TIS-PARITY PATCH: (APPLIED.{0,40}|FAILED.{0,60})|EVAL-V2ACC total=[0-9]+ v2_acc=[0-9.]+% mean_reward=[0-9.]+|\[convert\] .{0,40}|Pre RL Training:.{0,130}|Intermediate Eval \(step=[0-9]+\):.{0,130}' "$RAW" 2>/dev/null | while read -r l; do emit_new "$l"; done

  # hard errors
  if grep -qaE 'RESOURCE_EXHAUSTED|OutOfMemory|OOM|Killed|Traceback \(most|AssertionError|RuntimeError|TIS-PARITY PATCH: FAILED' "$RAW" 2>/dev/null; then
    grep -aoE 'RESOURCE_EXHAUSTED.{0,70}|OutOfMemory.{0,40}|OOM[^a-zA-Z].{0,20}|Killed|[A-Za-z]*Error:.{0,120}|TIS-PARITY PATCH: FAILED.{0,60}' "$RAW" 2>/dev/null | grep -avE 'cuda|pynvml|vllm._C|rope_scaling' | tail -4 | while read -r e; do emit_new "ERR: $e"; done
    if grep -qaE 'RESOURCE_EXHAUSTED|OutOfMemory|OOM|TIS-PARITY PATCH: FAILED' "$RAW" 2>/dev/null; then
      log "VERDICT: hard error (OOM or patch-fail) -- see ERR lines above; stop + adjust"; exit 0
    fi
  fi

  # emit each intermediate eval as it lands (20/40/.../250); exit only at step-250 or crash.
  # exit at the target eval step (TARGET env, default 40) so each eval wakes the session.
  T=${TARGET:-40}
  if grep -qaE "Intermediate Eval \(step=${T}\):" "$RAW" 2>/dev/null; then
    log "VERDICT: step-${T} reached -- $(grep -aoE "Intermediate Eval \(step=${T}\): corr=[0-9]+, total=[0-9]+, accuracy=[0-9.]+" "$RAW" | head -1); TIS: $(grep -aoE 'TIS-ACTIVE weight_mean=[0-9.]+ weight_min=[0-9.]+' "$RAW" | tail -1). GPU curve: 20=91.03 40~91.7 250=92.42."; exit 0
  fi
done
log "monitor window closed (re-arm to continue)"
