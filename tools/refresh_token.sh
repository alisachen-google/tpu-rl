#!/bin/bash
# Pull the vended K8s SA token from GCS into ~/.k8s-token (see tools/token-vendor-pod.yaml).
set -euo pipefail
SRC=${1:-gs://ubench-logs/alisachen-tpu7x/auth/k8s-token}
TMP=$(mktemp)
gsutil -q cp "$SRC" "$TMP"
test -s "$TMP"                      # never overwrite a good token with an empty file
install -m 600 "$TMP" "$HOME/.k8s-token"; rm -f "$TMP"
"$HOME/bin/kb" auth whoami | head -2
