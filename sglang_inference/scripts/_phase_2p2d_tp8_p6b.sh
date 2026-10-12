#!/usr/bin/env bash
# 2P/2D TP-8, phase 6: MT-Bench 80x2 through the router at c=1 and c=4; turn-2 cached tokens show whether the mini-LB keeps a conversation on one prefill.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p6b; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; LB=http://$P1:8080; Q=$HOME/pd_disagg/prompts/mt_bench/question.jsonl
SNAP=$(ls -d /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/* | head -1)
log(){ echo "$(date -u +%FT%TZ) [2p2d-p6b] $*"; }
cached(){ for u in $P1 $P2; do echo "   $u cached_tokens_total=$(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:cache_hit_rate\{[^}]*tp_rank="0"' | sed -E 's/.* //') prompt_tokens_total=$(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:prompt_tokens_total\{[^}]*tp_rank="0"' | sed -E 's/.* //')"; done; }
for c in 1 4; do log "=== MT-Bench 80x2 c=$c"; log "   prefill counters before:"; cached
  $PY $S/mtbench_runner.py --url $LB --questions $Q --snapshot $SNAP --n 80 --concurrency $c --max-new-tokens 512 --out $OUT/mtbench_c$c.jsonl --label 2p2d_B_c$c 2>&1 | grep -E '^\[' | tee -a $OUT/summary.txt | sed 's/^/   /'
  log "   prefill counters after:"; cached; done
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p6b done"
