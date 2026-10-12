#!/usr/bin/env bash
# 2P/2D TP-8, phase 3: rooms and n>1 through the router with two prefills (F-6/#1147 class), batches, consecutive calls, chat n=2.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/2p2d_tp8/p3; mkdir -p $OUT
LB=http://10.128.15.217:8080; log(){ echo "$(date -u +%FT%TZ) [2p2d-p3] $*"; }
gen(){ t0=$(date +%s.%N); code=$(curl -s -m $3 $LB$4 -H 'Content-Type: application/json' -d "$2" -o $OUT/$1.json -w '%{http_code}'); t1=$(date +%s.%N)
  n=$(python3 -c "import json; d=json.load(open('$OUT/$1.json')); print(len(d) if isinstance(d,list) else len(d.get('choices',[])) or 1)" 2>/dev/null); echo "   $1: HTTP ${code:-000} outputs=${n:-?} $(python3 -c "print(round($t1-$t0,1))")s"; }
log "=== n=2,3,8 single prompt (/generate)"; for n in 2 3 8; do gen n$n "{\"text\":\"Write one sentence about the sea.\",\"sampling_params\":{\"n\":$n,\"max_new_tokens\":32,\"temperature\":0.7}}" 300 /generate; done
log "=== batch of 4 prompts, n=1 and n=2"; gen batch4 '{"text":["A","B","C","D"],"sampling_params":{"max_new_tokens":16}}' 300 /generate; gen batch4_n2 '{"text":["A","B","C","D"],"sampling_params":{"n":2,"max_new_tokens":16,"temperature":0.7}}' 300 /generate
log "=== 6 consecutive n=2 calls (rooms must stay unique across both prefills)"; for i in 1 2 3 4 5 6; do gen consec$i '{"text":"Count to three.","sampling_params":{"n":2,"max_new_tokens":16,"temperature":0.7}}' 300 /generate; done
log "=== chat n=2"; gen chat_n2 '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Say hi."}],"n":2,"max_tokens":16}' 300 /v1/chat/completions
log "=== 12 concurrent n=2 requests"; for i in $(seq 1 12); do curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d '{"text":"Name a colour.","sampling_params":{"n":2,"max_new_tokens":8,"temperature":0.7}}' -o $OUT/conc$i.json -w "%{http_code}\n" & done | sort | uniq -c | sed 's/^/   HTTP /'
log "=== engines after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p3 done"
