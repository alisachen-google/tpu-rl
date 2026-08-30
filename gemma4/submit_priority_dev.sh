#!/bin/bash
# One-shot resubmit of the qwen3-transfer bundle probe into the priority-dev
# owned lane (ClusterQueue priority-dev, 51 idle chips). Run AFTER joining
# tpu-priority-dev-team@twosync.google.com -- the preflight below verifies
# RBAC and refuses to run without it, so it is safe to invoke early.
#
# Usage:  bash submit_priority_dev.sh [RELEASE]     (default release: g4r9pd)
set -euo pipefail

CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
NS=priority-dev
REL="${1:-g4r9pd}"
STAGE=/mnt/disks/scratch/g4r8-stage
IMG=gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing

# Fresh ADC token for GKE (the gke-gcloud-auth-plugin path breaks on invalid_rapt)
export CLOUDSDK_AUTH_ACCESS_TOKEN="$(gcloud auth application-default print-access-token)"

echo "== preflight: RBAC in $NS =="
CJ=$(kubectl --context=$CTX auth can-i create jobsets   -n $NS 2>/dev/null || true)
CC=$(kubectl --context=$CTX auth can-i create configmaps -n $NS 2>/dev/null || true)
echo "  create jobsets=$CJ  create configmaps=$CC"
if [ "$CJ" != "yes" ] || [ "$CC" != "yes" ]; then
  echo "REFUSING: not yet authorized in $NS. Join tpu-priority-dev-team@twosync.google.com first."
  exit 1
fi

echo "== owned-quota check =="
kubectl --context=$CTX get clusterqueue priority-dev \
  -o jsonpath='{range .spec.resourceGroups[*].flavors[*]}{.name}={.resources[?(@.name=="google.com/tpu")].nominalQuota}{"\n"}{end}' 2>/dev/null

echo "== install $REL into $NS =="
cd "$STAGE"
helm --kube-context=$CTX install "$REL" . \
  -f values-base.yaml -f values-4x4x4.yaml -f values-priority-dev.yaml -f g4r9p1-envs.yaml \
  --namespace $NS \
  --set-file workload_launcher="$STAGE/g4r8_launcher.sh" \
  --set workload.image="$IMG" \
  --set workload_patches=true --set patches.enabled=true \
  --set workload.configFile= --set maxRestarts=50 \
  --set kueue_priority_class=medium 2>&1 | grep -E "STATUS|Error" || true

kubectl --context=$CTX get jobset "$REL" -n $NS --no-headers 2>/dev/null || true
echo "Done. Monitor pattern: pods -n $NS matching ${REL}-pathways-head / ${REL}-worker."
