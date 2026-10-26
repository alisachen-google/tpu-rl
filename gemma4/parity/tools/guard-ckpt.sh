#!/bin/bash
# Restore-after-save guard for the RUNNING g4pt1 (checkpointing still on, period 100): resume is
# broken on this stack (orbax v1 "incompatible handler"), so a committed checkpoint makes any later
# restart fatal. Orbax only treats a step as restorable when <step>/commit_success.txt exists, so
# removing that marker (+ _CHECKPOINT_METADATA) >=6 min after it appears (after the post-commit
# metadata thread finished) is enough and instant. gsutil writes 403 on this VM (compute SA scope);
# use the GCS JSON API with a fresh ADC access token every loop. Never touches the -it checkpoint.
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
B=ubench-logs; PFX=g4pt1/checkpoints/actor
OUT=${PARITY_DIR:-$(dirname "$0")/..}/guard-ckpt.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
enc(){ python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$1"; }
declare -A first_seen; END=$((SECONDS+28800))
log "guard start on gs://$B/$PFX (marker-delete mode)"
while [ $SECONDS -lt $END ]; do
  TOK=$(gcloud auth application-default print-access-token 2>/dev/null)
  # list commit markers under the prefix
  MARKERS=$(curl -s -H "Authorization: Bearer $TOK" "https://storage.googleapis.com/storage/v1/b/$B/o?prefix=$(enc "$PFX/")&fields=items(name)&maxResults=1000" \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(i['name'] for i in d.get('items',[]) if i['name'].endswith('/commit_success.txt')))" 2>/dev/null)
  for m in $MARKERS; do
    step=$(basename "$(dirname "$m")")
    if [ -z "${first_seen[$step]}" ]; then first_seen[$step]=$SECONDS; log "step $step committed ($m); marker removal in 6 min"; fi
    if [ $((SECONDS - first_seen[$step])) -ge 360 ]; then
      for o in "$m" "$PFX/$step/_CHECKPOINT_METADATA"; do
        code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $TOK" "https://storage.googleapis.com/storage/v1/b/$B/o/$(enc "$o")")
        log "DELETE $o -> HTTP $code"
      done
      unset first_seen[$step]
    fi
  done
  sleep 60
done; log "guard window closed"
