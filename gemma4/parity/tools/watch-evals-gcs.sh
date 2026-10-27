#!/bin/bash
# Credential-independent eval monitor: polls the per-question dumps the run writes to GCS (readable
# by the VM service account via gsutil), scores each new one locally, prints a breakdown line and
# exits when a new eval (step > ${SEEN_MAX:-0}) lands. Dumps are named eval_<step>.jsonl. Re-arm with SEEN_MAX=<last step>.
RUN=${RUN:-g4pt1}; SEEN_MAX=${SEEN_MAX:-0}; DIR=${PARITY_DIR:-$(dirname "$0")/..}/evals_live
mkdir -p "$DIR"; END=$((SECONDS+14400))
while [ $SECONDS -lt $END ]; do
  for f in $(env -u GOOGLE_APPLICATION_CREDENTIALS gsutil ls gs://ubench-logs/$RUN/evals/ 2>/dev/null | grep -oE 'eval_[0-9]+\.jsonl' | sort -u); do
    k=$((10#$(echo "$f" | grep -oE '[0-9]+')))
    if [ "$k" -gt "$SEEN_MAX" ]; then
      env -u GOOGLE_APPLICATION_CREDENTIALS gsutil -q cp "gs://ubench-logs/$RUN/evals/$f" "$DIR/${RUN}_$f" 2>/dev/null
      python3 -I - "$DIR/${RUN}_$f" "$k" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]; k=int(sys.argv[2]); n=len(rows)
acc=sum(r['acc'] for r in rows)/n*100; nobox=sum(1 for r in rows if r['boxed'] in (None,''))/n*100
bw=sum(1 for r in rows if r['acc']==0 and r['boxed'] not in (None,''))/n*100; mc=sum(r['n_chars'] for r in rows)/n
gpu={0:80.31,1:91.03,2:91.69,3:91.36,4:91.84,5:91.64,6:92.04,7:92.67,8:92.77,9:92.52,10:92.65,11:92.60,12:92.65,13:92.42}
print(f"EVAL step={k} n={n} v2_acc={acc:.2f}% boxed_wrong={bw:.2f}% nobox={nobox:.2f}% mean_chars={mc:.0f} | GPU step{k}={gpu.get(k//20 if k<250 else 13,'?')} delta={acc-gpu.get(k//20 if k<250 else 13,acc):+.2f}pp")
PY
      exit 0
    fi
  done
  sleep 120
done; echo "eval watch window closed (no new eval beyond k=$SEEN_MAX)"
