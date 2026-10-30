#!/bin/bash
# apply-jobset.sh <manifest.yaml> [context] [namespace]
# Applies the ConfigMaps of a parity manifest, then the JobSet with an ownerReference to its <name>-launcher
# ConfigMap (works around the kyverno owner-only-exec-stamp-creator mutation that strips ownerless JobSets).
set -euo pipefail
M=$1; CTX=${2:-gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap}; NS=${3:-ubench-regression-tests}
K="kubectl --context $CTX -n $NS"
python3 - "$M" <<'PY' > /tmp/apply-jobset-cms.yaml
import sys,re
raw=open(sys.argv[1]).read(); parts=re.split(r'(?m)^---\s*$', raw)
print("---".join(p for p in parts if 'kind: JobSet' not in p))
PY
$K apply --server-side --force-conflicts -f /tmp/apply-jobset-cms.yaml
NAME=$(python3 -c "
import sys,re,yaml
raw=open('$M').read(); parts=re.split(r'(?m)^---\s*$', raw)
print([yaml.safe_load(p)['metadata']['name'] for p in parts if 'kind: JobSet' in p][0])")
UID_=$($K get configmap "$NAME-launcher" -o jsonpath='{.metadata.uid}')
python3 - "$M" "$NAME" "$UID_" <<'PY' > /tmp/apply-jobset-js.yaml
import sys,re,yaml
raw=open(sys.argv[1]).read(); parts=re.split(r'(?m)^---\s*$', raw)
d=[yaml.safe_load(p) for p in parts if 'kind: JobSet' in p][0]
d['metadata']['ownerReferences']=[{'apiVersion':'v1','kind':'ConfigMap','name':sys.argv[2]+'-launcher','uid':sys.argv[3]}]
print(yaml.safe_dump(d, sort_keys=False, width=10**6))
PY
$K apply --server-side --force-conflicts -f /tmp/apply-jobset-js.yaml
$K get jobset "$NAME" -o json | python3 -c "
import json,sys; j=json.load(sys.stdin); print('replicatedJobs:', [(rj['name'], len(rj['template']['spec'].get('template',{}).get('spec',{}).get('containers',[]))) for rj in j['spec']['replicatedJobs']])"
