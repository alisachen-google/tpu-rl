#!/usr/bin/env bash
# 2P/2D TP-8, phases 0-2: bring-up checks, compile audit per engine, routing correctness, TC-1 8k differential vs the TP-8 golden, TC-7 decode-phase abort.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p0; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=${RUN:-2p2d_tp8_bringup}; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080
P="http://$P1:30000,http://$P2:30000"; D="http://$D1:30000,http://$D2:30000"
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p0] $*"; }
rlog(){ # $1 ip $2 label -> prints server.log tail-safe copy path (fetched)
  if [[ $1 == $D1 ]]; then echo $R/runs/$RUN/$2/server.log; else $SSH sa_101411256518727260534@$1 "sudo cat /home/alisachen_google_com/pd_disagg/results/runs/$RUN/$2/server.log" > $OUT/${2}_server.log 2>/dev/null; echo $OUT/${2}_server.log; fi; }
log "=== status"; bash $S/pd4.sh status | sed 's/^/   /'
log "=== compile audit per engine (startup vs serving-time)"
for e in "$P1 p1" "$P2 p2" "$D1 d1" "$D2 d2"; do set -- $e; bash $S/compile_audit.sh "$(rlog $1 $2)" $2 | sed 's/^/   /'; done
log "=== smoke through the router"; curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('   ', repr(d['text']), d['meta_info']['completion_tokens'],'tokens')"
log "=== phase 2: routing probe (24 requests, c=4; every P and D must receive work; repeated prompt gives one distinct output)"
$PY $S/route_probe.py --router $LB --prefills $P --decodes $D --n 24 --concurrency 4 2>&1 | tee $OUT/route_probe.txt | sed 's/^/   /' | cut -c1-240
log "=== phase 1: TC-1 differential, 8k prompts, greedy 1024 tokens, c=1, PD(2P/2D) vs TP-8 monolithic c=1 golden (gpt-oss_fp8_rep2_c1; the c=8 golden differs by batching numerics)"
$PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url $LB --out $OUT/8192.jsonl --max-new-tokens 1024 --return-logprob 2>&1 | tail -n 3 | sed 's/^/   /'
$PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8_rep2_c1/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_8192.json 2>&1 | tail -n 6 | sed 's/^/   /'
log "=== TC-1 at 32k and 128k, c=1 (expect the known NUM class as in 1P/1D, needles must hold)"
for n in 32768 131072; do $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/$n.jsonl --url $LB --out $OUT/$n.jsonl --max-new-tokens 256 2>&1 | tail -n 1 | sed 's/^/   /'; [[ -f $R/golden/gpt-oss_fp8_rep2_c1/$n.jsonl ]] && $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8_rep2_c1/$n.jsonl --b $OUT/$n.jsonl --report $OUT/diff_$n.json --min-agreement 0 --exact-prefix-tokens 0 2>&1 | tail -n 3 | sed 's/^/   /'; done
log "=== TC-7 decode-phase client abort through the router (2 aborts, follow-ups must be served; both decodes polled)"
$PY $S/abort_test.py --url $LB --prompt-file $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --phase decode --abort-after 5 --repeat 2 --server-info http://$D1:30000 --server-info http://$D2:30000 2>&1 | tail -n 8 | sed 's/^/   /'
log "=== serving-time recompiles after the battery"; for e in "$P1 p1" "$P2 p2" "$D1 d1" "$D2 d2"; do set -- $e; bash $S/compile_audit.sh "$(rlog $1 $2)" $2 | grep -E 'serving' | sed "s/^/   $2 /"; done
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p0 done"
