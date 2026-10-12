#!/bin/bash
# F-4 isolation at TP-4: PD pair and golden with --disable-hybrid-swa-memory; 8k greedy; diff PD vs golden (both no-SWA-mem).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d/tp4_noswamem; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
A=10.128.15.217; B=10.128.0.90; ME=alisachen_google_com; LB=http://$B:8081
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@$A"
on_a(){ $SSH_A "sudo -n -iu $ME bash -lc $(printf %q "$*")"; }
log(){ echo "$(date -u +%FT%TZ) [tp4-noswamem] $*"; }
source $S/env_singlehost_v7x.sh >/dev/null 2>&1
log "stopping p1 on a and everything on b"; on_a "bash $S/cleanup.sh" >/dev/null 2>&1; bash $S/cleanup.sh >/dev/null 2>&1; sleep 5
mkdir -p $R/runs/2p2d_noswamem/d1; on_a "mkdir -p /home/$ME/pd_disagg/results/runs/2p2d_noswamem"
log "launch p1(noswamem) on a, d1(noswamem) on b"
$SSH_A "sudo -n -iu $ME bash -lc $(printf %q "cd /tmp && exec >/dev/null 2>&1 </dev/null; ENABLE_COMPILE_CACHE=1 setsid nohup bash $S/launch_engine.sh /home/$ME/pd_disagg/recipes/2p2d/gpt-oss_B_128k_p1_noswamem.json 2p2d_noswamem p1 &")"
(ENABLE_COMPILE_CACHE=1 setsid nohup bash $S/launch_engine.sh $HOME/pd_disagg/recipes/2p2d/gpt-oss_B_128k_d1_noswamem.json 2p2d_noswamem d1 > $R/runs/2p2d_noswamem/launcher_d1.out 2>&1 < /dev/null &)
for e in "p1 $A 30000" "d1 $B 30000"; do set -- $e; for i in $(seq 1 240); do curl -sf -m 5 http://$2:$3/health >/dev/null 2>&1 && { log "$1 up after $((i*15)) s"; break; }; sleep 15; done; done
pkill -u $(id -un) -f 'sglang_router.launch_router.*--port 8081' 2>/dev/null; setsid nohup $PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://$A:30000 8998 --decode http://$B:30000 --host 0.0.0.0 --port 8081 > $OUT/router.log 2>&1 < /dev/null &
for i in $(seq 1 40); do curl -sf -m 5 $LB/health >/dev/null 2>&1 && break; sleep 3; done; log "router $(curl -s -m 5 -o /dev/null -w '%{http_code}' $LB/health)"
log "PD(TP-4, noswamem) 8k generate"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url $LB --out $OUT/pd_8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen_pd.log 2>&1
log "stopping d1 + router; launching golden(TP-4, noswamem) on b"; pkill -u $(id -un) -f 'sglang_router.launch_router.*--port 8081'; bash $S/cleanup.sh >/dev/null 2>&1; sleep 5
mkdir -p $R/runs/2p2d_noswamem_golden/golden; (ENABLE_COMPILE_CACHE=1 setsid nohup bash $S/launch_engine.sh $HOME/pd_disagg/recipes/2p2d/gpt-oss_G_128k_tp4_noswamem.json 2p2d_noswamem_golden golden > $R/runs/2p2d_noswamem_golden/launcher_golden.out 2>&1 < /dev/null &)
for i in $(seq 1 240); do curl -sf -m 5 http://$B:30010/health >/dev/null 2>&1 && { log "golden up after $((i*15)) s"; break; }; sleep 15; done
log "golden(TP-4, noswamem) 8k generate"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url http://$B:30010 --out $OUT/golden_8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen_golden.log 2>&1
log "=== PD(TP-4,noswamem) vs golden(TP-4,noswamem):"; $PY $S/golden_diff.py diff --a $OUT/golden_8192.jsonl --b $OUT/pd_8192.jsonl --report $OUT/diff.json 2>&1 | tail -n 3 | sed 's/^/   /' | cut -c1-200
log "=== golden(TP-4,noswamem) vs golden(TP-4):"; $PY $S/golden_diff.py diff --a $R/2p2d/tp4_golden/8192.jsonl --b $OUT/golden_8192.jsonl --report $OUT/diff_golden_swa.json 2>&1 | tail -n 3 | sed 's/^/   /' | cut -c1-200
bash $S/cleanup.sh >/dev/null 2>&1; on_a "bash $S/cleanup.sh" >/dev/null 2>&1; log "phase tp4-noswamem done (both hosts freed)"
