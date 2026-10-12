#!/usr/bin/env bash
# 2P/2D on the tpu7x-16 pair: P1,P2 on host a (chips 0,1 / 2,3), D1,D2 on host b; sglang_router mini-LB on a:8080.
#   pd2.sh start <RUN>  |  pd2.sh stop  |  pd2.sh status
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R=$HOME/pd_disagg/recipes/2p2d
A=10.128.15.217; B=10.128.0.90; LB_PORT=${LB_PORT:-8080}; ME=alisachen_google_com
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@$A"
log(){ echo "$(date -u '+%F %T') $*"; }
on_a(){ $SSH_A "sudo -n -iu $ME bash -lc $(printf %q "$*")"; }
spawn_a(){ $SSH_A "sudo -n -iu $ME bash -lc $(printf %q "cd /tmp && exec >/dev/null 2>&1 </dev/null; ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1} setsid nohup bash $HERE/launch_engine.sh $1 $2 $3 &")"; }
spawn_b(){ mkdir -p "$HOME/pd_disagg/results/runs/$2/$3"; ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1} setsid nohup bash "$HERE/launch_engine.sh" "$1" "$2" "$3" >> "$HOME/pd_disagg/results/runs/$2/launcher_$3.out" 2>&1 < /dev/null & }
wait_url(){ local deadline=$(( $(date +%s) + $2 )) t0=$(date +%s); while (( $(date +%s) < deadline )); do curl -sf -m 5 "$1" >/dev/null 2>&1 && { log "$3 ready after $(( $(date +%s) - t0 )) s"; return 0; }; sleep 15; done; log "$3 NOT ready after $2 s"; return 1; }
case "${1:-}" in
start)
  run=${2:?RUN}; mkdir -p "$HOME/pd_disagg/results/runs/$run"; on_a "mkdir -p /home/$ME/pd_disagg/results/runs/$run"
  log "starting d1,d2 on b"; spawn_b $R/gpt-oss_B_128k_d1.json $run d1; sleep 5; spawn_b $R/gpt-oss_B_128k_d2.json $run d2
  log "starting p1,p2 on a"; spawn_a $R/gpt-oss_B_128k_p1.json $run p1; sleep 5; spawn_a $R/gpt-oss_B_128k_p2.json $run p2
  for e in "p1 $A 30000" "p2 $A 30001" "d1 $B 30000" "d2 $B 30001"; do set -- $e; wait_url "http://$2:$3/health" ${HEALTH_TIMEOUT:-5400} "$1 /health"; done
  log "starting router on a:$LB_PORT"
  on_a "cd /tmp && exec >/dev/null 2>&1 </dev/null; source $HERE/env_singlehost_v7x.sh; setsid nohup \$PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://$A:30000 8998 --prefill http://$A:30001 8999 --decode http://$B:30000 --decode http://$B:30001 --host 0.0.0.0 --port $LB_PORT > /home/$ME/pd_disagg/results/runs/$run/router.log 2>&1 &"
  wait_url "http://$A:$LB_PORT/health" 300 "router /health"
  ;;
stop) log "stopping on a"; on_a "bash $HERE/cleanup.sh"; log "stopping on b"; bash "$HERE/cleanup.sh" ;;
status)
  for e in "p1 $A 30000" "p2 $A 30001" "d1 $B 30000" "d2 $B 30001" "router $A $LB_PORT"; do set -- $e; printf '%-6s %s:%s health=%s\n' "$1" "$2" "$3" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$2:$3/health)"; done
  echo "a holders: $(on_a "sudo -n fuser -v /dev/vfio/* 2>&1 | awk 'NR>1{print \$(NF-3)}' | sort | uniq -c | tr '\n' ' '")"; echo "b holders: $(sudo -n fuser -v /dev/vfio/* 2>&1 | awk 'NR>1{print $(NF-3)}' | sort | uniq -c | tr '\n' ' ')"
  ;;
*) echo "usage: pd2.sh start <RUN> | stop | status" ;;
esac
