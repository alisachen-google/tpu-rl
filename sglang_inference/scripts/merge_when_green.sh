#!/bin/bash
# merge_when_green.sh <pr> <expected-head-sha>: rerun NOT-ADMITTED TPU jobs when their run completes, then squash-merge once every latest check is green.
PR=$1; HEAD=$2; G=$HOME/pd_disagg/scripts/ghapi.sh; log(){ echo "$(date -u +%FT%TZ) [merge#$PR] $*"; }
for i in $(seq 1 200); do
  cur=$($G /pulls/$PR | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['head']['sha'], d['merged'], d['state'])")
  set -- $cur; [ "$1" = "$HEAD" ] || { log "head moved to ${1:0:10}; stop"; exit 1; }; [ "$2" = "True" ] && { log "already merged"; exit 0; }
  summary=$($G "/commits/$HEAD/check-runs?per_page=100" | python3 -c "
import json,sys; runs=json.load(sys.stdin)['check_runs']; latest={}
for r in runs: latest.setdefault(r['name'], r)
bad=[(r['name'],r['status'],r['conclusion'],r['details_url']) for r in latest.values() if r['conclusion'] not in ('success','skipped','neutral')]
print('GREEN' if not bad else 'PENDING' if any(b[1]!='completed' for b in bad) else 'FAILED'); [print(*b) for b in bad]")
  state=$(echo "$summary" | head -n1); log "$state"; echo "$summary" | tail -n +2 | sed 's/^/   /'
  if [ "$state" = GREEN ]; then
    $G /pulls/$PR/merge -X PUT -d '{"merge_method":"squash"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d)"; log "merge attempted"; exit 0
  elif [ "$state" = FAILED ]; then
    run=$(echo "$summary" | grep -oE 'runs/[0-9]+' | head -n1 | cut -d/ -f2)
    status=$($G /actions/runs/$run | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
    if [ "$status" = completed ]; then
      if $G /actions/runs/$run/attempts/1/jobs?per_page=100 >/dev/null; then :; fi
      code=$($G /actions/runs/$run/rerun-failed-jobs -X POST -o /dev/null -w '%{http_code}'); log "rerun-failed-jobs on run $run -> HTTP $code"; [ "$code" = 201 ] || { log "rerun refused; stop"; exit 1; }; sleep 120
    else log "run $run still $status; waiting before rerun"; fi
  fi
  sleep 300
done; log "gave up"
