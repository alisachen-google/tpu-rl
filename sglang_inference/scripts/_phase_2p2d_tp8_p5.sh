#!/usr/bin/env bash
# 2P/2D TP-8, phase 5: load through the router. bench_serving random 8k/256 at c=8 and c=16, then 131k/256 at c=8; queue gauges per engine must drain, 0 failures.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p5; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080
log(){ echo "$(date -u +%FT%TZ) [2p2d-p5] $*"; }
gauges(){ for u in $P1 $P2 $D1 $D2; do echo "   $u: $(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_prealloc_queue_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs|num_bootstrap_failed_reqs_total|num_transfer_failed_reqs_total)\{[^}]*tp_rank="0"' | sed -E 's/\{[^}]*\}//; s/sglang://' | tr '\n' ' ')"; done; }
run(){ # $1 label $2 isl $3 osl $4 conc $5 nprompts
  log "=== $1: random isl=$2 osl=$3 c=$4 n=$5"
  $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len $2 --random-output-len $3 --random-range-ratio 1.0 --num-prompts $5 --max-concurrency $4 --disable-tqdm --output-file $OUT/$1.jsonl 2>&1 | grep -E 'Successful requests|Failed requests|Request throughput|Output token throughput|Mean TTFT|P99 TTFT|Mean ITL|P99 ITL|Mean E2E|Total input|Benchmark duration' | sed 's/^/   /'
  sleep 20; log "   gauges after $1:"; gauges; }
log "=== gauges before"; gauges
run c8_8k 8192 256 8 64
run c16_8k 8192 256 16 96
run c8_32k 32768 256 8 32
run c4_128k 131072 256 4 12
log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p5 done"
