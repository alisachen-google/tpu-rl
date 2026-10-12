#!/usr/bin/env bash
# 2P/2D at TP-8, one engine per host. Rank order = PD_HOSTS: p1=a, p2=sixiang, d1=b (this VM), d2=tpu7x-8. Router (mini-LB) on p1:$LB_PORT.
#   pd4.sh start <recipe.json> <RUN>  |  pd4.sh stop  |  pd4.sh status
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ME=alisachen_google_com
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20
export PD_HOSTS=${PD_HOSTS:-$P1,$P2,$D1,$D2}; LB_PORT=${LB_PORT:-8080}; PORT=30000
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u '+%F %T') $*"; }
on(){ local ip=$1; shift; if [[ $ip == "$D1" ]]; then bash -lc "$*"; else $SSH sa_101411256518727260534@$ip "sudo -n -iu $ME bash -lc $(printf %q "$*")"; fi; }
spawn(){ local ip=$1 recipe=$2 rank=$3 run=$4; on "$ip" "mkdir -p /home/$ME/pd_disagg/results/runs/$run; cd /tmp && exec >/dev/null 2>&1 </dev/null; PD_HOSTS=$PD_HOSTS ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1} setsid nohup bash /home/$ME/pd_disagg/scripts/launch_rank.sh /home/$ME/pd_disagg/recipes/$(basename "$recipe") $rank $run &"; }
wait_url(){ local deadline=$(( $(date +%s) + $2 )) t0=$(date +%s); while (( $(date +%s) < deadline )); do curl -sf -m 5 "$1" >/dev/null 2>&1 && { log "$3 ready after $(( $(date +%s) - t0 )) s"; return 0; }; sleep 15; done; log "$3 NOT ready after $2 s"; return 1; }
labels=(p1 p2 d1 d2); IFS=, read -r -a hosts <<< "$PD_HOSTS"
case "${1:-}" in
start)
  recipe=${2:?recipe}; run=${3:?RUN}
  log "starting decodes first (d1=$D1 d2=$D2), then prefills (p1=$P1 p2=$P2); recipe=$(basename "$recipe") run=$run"
  for r in 2 3 0 1; do spawn "${hosts[$r]}" "$recipe" $r "$run"; sleep 3; done
  for r in 2 3 0 1; do wait_url "http://${hosts[$r]}:$PORT/health" ${HEALTH_TIMEOUT:-5400} "${labels[$r]} /health" || exit 1; done
  wait_url "http://$P1:$LB_PORT/health" 600 "router /health"
  ;;
stop) for r in 0 1 2 3; do log "stopping ${labels[$r]} on ${hosts[$r]}"; on "${hosts[$r]}" "bash /home/$ME/pd_disagg/scripts/cleanup.sh" >/dev/null 2>&1; done ;;
status)
  for r in 0 1 2 3; do printf '%-7s %s:%s health=%s\n' "${labels[$r]}" "${hosts[$r]}" "$PORT" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://${hosts[$r]}:$PORT/health)"; done
  printf '%-7s %s:%s health=%s\n' router "$P1" "$LB_PORT" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$P1:$LB_PORT/health)"
  for r in 0 1 2 3; do echo "${labels[$r]} vfio held: $(on "${hosts[$r]}" "n=0; for f in /dev/vfio/[0-9]*; do u=\$(sudo -n fuser \$f 2>/dev/null | wc -w); [ \$u -gt 0 ] && n=\$((n+1)); done; echo \$n" 2>/dev/null | tail -n1)"; done
  ;;
*) echo "usage: pd4.sh start <recipe.json> <RUN> | stop | status" ;;
esac
