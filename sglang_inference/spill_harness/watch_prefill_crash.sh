#!/usr/bin/env bash
# Abort pd_spill's long health wait if the prefill engine on a dies during boot.
RUN=${1:?RUN}; SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@10.128.15.217"
for _ in $(seq 1 90); do
  sleep 20
  if $SSH_A "sudo grep -q -E 'An engine or router exited|Traceback' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log 2>/dev/null"; then
    echo "[watch] $(date -u +%T) prefill crashed:"; $SSH_A "sudo grep -n -E 'Error|exited' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | tail -5 | cut -c1-240"
    for p in $(pgrep -u $(id -un) -f 'phase_dra[m].sh|pd_spil[l].sh start|stores-star[t]'); do pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; done
    exit 1
  fi
  curl -sf -m 5 http://10.128.15.217:30000/health >/dev/null && { echo "[watch] prefill healthy"; exit 0; }
done
