#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$BENCH_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_profile.sh PROFILE [OPTIONS]

Profiles: tp1-dp64, tp2-dp32, tp4-dp32

Options:
  --run-name NAME  Use an explicit JobSet/run name.
  --render-only    Render and validate; do not access the cluster.
  --submit-only    Render, validate, and submit; do not wait.
  --attach         Do not submit; monitor an existing --run-name.
  -h, --help       Show this help.
EOF
}

PROFILE=${1:-}
if [ -z "$PROFILE" ]; then
  usage >&2
  exit 2
fi
shift

case "$PROFILE" in
  tp1-dp64) PREFIX=g4btp1; EXPECTED_TP=1; EXPECTED_DP=64 ;;
  tp2-dp32) PREFIX=g4btp2; EXPECTED_TP=2; EXPECTED_DP=32 ;;
  tp4-dp32) PREFIX=g4btp4d32; EXPECTED_TP=4; EXPECTED_DP=32 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac

RUN_NAME=""
MODE=run
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-name)
      [ "$#" -ge 2 ] || { echo "--run-name requires a value" >&2; exit 2; }
      RUN_NAME=$2
      shift 2
      ;;
    --render-only) MODE=render; shift ;;
    --submit-only) MODE=submit; shift ;;
    --attach) MODE=attach; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  if [ "$MODE" = attach ]; then
    echo "--attach requires --run-name" >&2
    exit 2
  fi
  RUN_NAME="${PREFIX}-$(date -u +%m%d%H%M)"
fi
if [ "${#RUN_NAME}" -gt 63 ] || ! [[ "$RUN_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "Invalid Kubernetes run name: $RUN_NAME" >&2
  exit 2
fi

CONTEXT=${BENCH_CONTEXT:-gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap}
NAMESPACE=${BENCH_NAMESPACE:-ubench-regression-tests}
QUEUE=${BENCH_QUEUE:-multislice-queue}
PRIORITY=${BENCH_PRIORITY:-medium}
RESULT_ROOT=${BENCH_RESULT_ROOT:-$HOME/gemma4-benchmark-results}
RESULT_DIR="$RESULT_ROOT/$RUN_NAME"
PROFILE_VALUES="$BENCH_DIR/profiles/$PROFILE.yaml"
IMAGE='gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:f7c957d0b286e03966fdd6f9b15e7609fae8460a933a9da0b9fbab3e04723c94'

for command_name in helm python3 sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done

(cd "$BENCH_DIR" && sha256sum --check --status patches.sha256) || {
  echo "PATCH-CONTRACT: FAILED top-level patches differ from patches.sha256" >&2
  exit 2
}

mkdir -p "$RESULT_DIR"
STAGE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/gemma4-benchmark.XXXXXX")
cleanup() {
  case "$STAGE_ROOT" in
    */gemma4-benchmark.*) rm -rf -- "$STAGE_ROOT" ;;
    *) echo "Refusing to remove unexpected staging path: $STAGE_ROOT" >&2 ;;
  esac
}
trap cleanup EXIT

cp -a "$REPO_ROOT/chart" "$STAGE_ROOT/chart"
cp -a "$REPO_ROOT/patches/." "$STAGE_ROOT/chart/patches/"

PATCH_COUNT=$(find "$STAGE_ROOT/chart/patches" -maxdepth 1 -type f | wc -l)
if [ "$PATCH_COUNT" -ne 33 ]; then
  echo "PATCH-CONTRACT: FAILED staged patch count=$PATCH_COUNT, expected 33" >&2
  exit 2
fi

RENDER="$RESULT_DIR/jobset.yaml"
helm template "$RUN_NAME" "$STAGE_ROOT/chart" \
  -f "$STAGE_ROOT/chart/values.yaml" \
  -f "$STAGE_ROOT/chart/values-4x4x4.yaml" \
  -f "$PROFILE_VALUES" \
  --namespace "$NAMESPACE" \
  --set "namespace=$NAMESPACE" \
  --set "queue=$QUEUE" \
  --set "kueue_priority_class=$PRIORITY" \
  --set maxRestarts=5 \
  --set workload_patches=true \
  --set patches.enabled=true \
  --set workload.configFile= \
  --set "workload.image=$IMAGE" \
  --set-file "workload_launcher=$BENCH_DIR/launcher.sh" \
  > "$RENDER"

python3 "$BENCH_DIR/validate_render.py" "$RENDER" \
  --profile "$PROFILE" \
  --run "$RUN_NAME" \
  --namespace "$NAMESPACE" \
  --queue "$QUEUE" \
  --priority "$PRIORITY" \
  --image "$IMAGE" \
  --manifest "$BENCH_DIR/patches.sha256"

echo "RENDER path=$RENDER"
echo "RUN name=$RUN_NAME profile=$PROFILE"
echo "ARTIFACT uri=gs://ubench-logs/alisachen-tpu7x/runs/$RUN_NAME/"
if [ "$MODE" = render ]; then
  exit 0
fi

for command_name in kubectl jq gcloud; do
  command -v "$command_name" >/dev/null || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done

refresh_token() {
  if [ "${BENCH_SKIP_AUTH_REFRESH:-0}" = "1" ]; then
    return 0
  fi
  local token
  if ! token=$(gcloud auth application-default print-access-token 2>/dev/null); then
    echo "AUTH: ADC token refresh failed" >&2
    return 1
  fi
  export CLOUDSDK_AUTH_ACCESS_TOKEN="$token"
}

kube() {
  kubectl --context "$CONTEXT" --namespace "$NAMESPACE" "$@"
}

refresh_token
if [ "$MODE" = attach ]; then
  kube get jobset "$RUN_NAME" >/dev/null
  echo "ATTACH JobSet/$RUN_NAME"
else
  if kube get jobset "$RUN_NAME" >/dev/null 2>&1; then
    echo "REFUSING: JobSet/$RUN_NAME already exists; use --attach" >&2
    exit 2
  fi
  kube apply -f "$RENDER"
  echo "SUBMITTED JobSet/$RUN_NAME"
fi

if [ "$MODE" = submit ]; then
  echo "MONITOR command: $BENCH_DIR/run_profile.sh $PROFILE --run-name $RUN_NAME --attach"
  exit 0
fi

missing_polls=0
polls=0
terminal=""
while [ -z "$terminal" ]; do
  if ! refresh_token; then
    sleep 30
    continue
  fi
  jobset_json=$(kube get jobset "$RUN_NAME" -o json 2>/dev/null || true)
  if [ -z "$jobset_json" ]; then
    missing_polls=$((missing_polls + 1))
    if [ "$missing_polls" -ge 5 ]; then
      echo "RESULT: FAILED JobSet disappeared for five polls" >&2
      exit 1
    fi
    sleep 30
    continue
  fi
  missing_polls=0
  terminal=$(printf '%s' "$jobset_json" | jq -r '
    [(.status.terminalState // empty),
     (.status.conditions[]?
     | select(.status == "True" and (.type == "Completed" or .type == "Failed"))
     | .type)] | map(select(length > 0)) | first // empty')
  polls=$((polls + 1))
  if [ -z "$terminal" ] && [ $((polls % 10)) -eq 1 ]; then
    admitted=$(printf '%s' "$jobset_json" | jq -r \
      '.metadata.annotations["kueue.x-k8s.io/workload"] // "created"')
    echo "WAIT JobSet/$RUN_NAME state=$admitted"
  fi
  if [ -z "$terminal" ]; then sleep 30; fi
done
echo "TERMINAL JobSet/$RUN_NAME state=$terminal"

pods_json=""
for _attempt in 1 2 3 4 5; do
  if refresh_token; then
    pods_json=$(kube get pods -l "jobset.sigs.k8s.io/jobset-name=$RUN_NAME" \
      -o json 2>/dev/null || true)
  fi
  [ -n "$pods_json" ] && break
  sleep 5
done
pod=$(printf '%s' "$pods_json" | jq -r \
  '[.items[]? | select(.metadata.name | contains("pathways-head"))]
   | sort_by(.metadata.creationTimestamp) | last | .metadata.name // empty')
HEAD_LOG="$RESULT_DIR/head.log"
if [ -n "$pod" ]; then
  kube logs "$pod" -c jax-tpu --timestamps > "$HEAD_LOG" 2>&1 || true
  echo "LOG path=$HEAD_LOG pod=$pod"
else
  echo "LOG: no pathways-head pod found" >&2
fi

SUMMARY="$RESULT_DIR/summary.json"
summary_ok=0
if [ -s "$HEAD_LOG" ]; then
  if python3 "$BENCH_DIR/summarize_log.py" "$HEAD_LOG" \
    --profile "$PROFILE" --run "$RUN_NAME" --output "$SUMMARY" \
    --expected-tp "$EXPECTED_TP" --expected-dp "$EXPECTED_DP" \
    --expected-steps 20; then
    summary_ok=1
  fi
fi

if [ "$terminal" != Completed ]; then
  echo "RESULT: FAILED JobSet/$RUN_NAME terminal=$terminal" >&2
  exit 1
fi
if [ "$summary_ok" -ne 1 ] || [ ! -s "$SUMMARY" ]; then
  echo "RESULT: FAILED completed run has no phase summary" >&2
  exit 1
fi
echo "RESULT: COMPLETE summary=$SUMMARY"
