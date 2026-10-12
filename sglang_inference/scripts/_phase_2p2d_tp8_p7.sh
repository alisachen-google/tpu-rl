#!/usr/bin/env bash
# 2P/2D TP-8, phase 7: speculative decoding (config D = EAGLE3 + HiCache + bf16 KV on all four engines).
# Relaunch the group on gpt-oss_D_128k_2p2d.json, then: compile audit, spec fields on every engine, routing probe,
# TC-1 8k differential vs the bf16 c=1 golden, MT-Bench c=1/c=4 (acceptance + throughput vs config B), decode-phase aborts.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d_tp8/p7; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
RUN=2p2d_tp8_D; P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080
P="http://$P1:30000,http://$P2:30000"; D="http://$D1:30000,http://$D2:30000"
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p7] $*"; }
rlog(){ if [[ $1 == $D1 ]]; then echo $R/runs/$RUN/$2/server.log; else $SSH sa_101411256518727260534@$1 "sudo cat /home/alisachen_google_com/pd_disagg/results/runs/$RUN/$2/server.log" > $OUT/${2}_server.log 2>/dev/null; echo $OUT/${2}_server.log; fi; }
gauges(){ for u in $P1 $P2 $D1 $D2; do echo "   $u: $(curl -s -m 5 http://$u:30000/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs|num_bootstrap_failed_reqs_total|num_transfer_failed_reqs_total)\{[^}]*tp_rank="0"' | sed -E 's/sglang://; s/\{[^}]*\}//' | tr '\n' ' ')"; done; }
SNAP=$(ls -d /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/* | head -1); Q=$HOME/pd_disagg/prompts/mt_bench/question.jsonl
log "=== relaunch on config D (EAGLE3): stop B group, start D group"; bash $S/pd4.sh stop | sed 's/^/   /'; sleep 20
t0=$(date +%s); bash $S/pd4.sh start $HOME/pd_disagg/recipes/gpt-oss_D_128k_2p2d.json $RUN | sed 's/^/   /' || { log "BRINGUP FAILED"; bash $S/pd4.sh status | sed 's/^/   /'; exit 1; }
log "   group up in $(( $(date +%s)-t0 )) s"
log "=== spec / kv fields per engine"; for u in $P1 $P2 $D1 $D2; do echo "   $u $(curl -s -m 10 http://$u:30000/get_server_info | $PY -c "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ('disaggregation_mode','speculative_algorithm','speculative_num_draft_tokens','kv_cache_dtype','page_size','enable_hierarchical_cache')})")"; done
log "=== compile audit per engine (startup vs serving-time)"; for e in "$P1 p1" "$P2 p2" "$D1 d1" "$D2 d2"; do set -- $e; bash $S/compile_audit.sh "$(rlog $1 $2)" $2 | sed 's/^/   /'; done
log "=== smoke through the router"; curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' | $PY -c "import json,sys; d=json.load(sys.stdin); print('   ', repr(d['text']), d['meta_info']['completion_tokens'],'tokens, spec_verify_ct', d['meta_info'].get('spec_verify_ct'))"
log "=== routing probe (24 requests, c=4)"; $PY $S/route_probe.py --router $LB --prefills $P --decodes $D --n 24 --concurrency 4 2>&1 | tee $OUT/route_probe.txt | sed 's/^/   /' | cut -c1-240
log "=== TC-1: 8k greedy 1024 tokens c=1 vs the bf16 c=1 golden (spec decode is lossless only up to numerics: expect NUM, needles found)"
$PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url $LB --out $OUT/8192.jsonl --max-new-tokens 1024 --return-logprob 2>&1 | tail -n 3 | sed 's/^/   /'
$PY $S/golden_diff.py diff --a $R/golden/gpt-oss_bf16_c1/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_8192_bf16.json 2>&1 | tail -n 6 | sed 's/^/   /'
log "=== MT-Bench 80x2 c=1 and c=4 (acceptance from spec_verify_ct; throughput vs config B: 187.7 / 640.7 tok/s)"
for c in 1 4; do $PY $S/mtbench_runner.py --url $LB --questions $Q --snapshot $SNAP --n 80 --concurrency $c --max-new-tokens 512 --out $OUT/mtbench_c$c.jsonl --label 2p2d_D_c$c 2>&1 | grep -E '^\[' | tee -a $OUT/summary.txt | sed 's/^/   /'; done
log "=== avg_spec_accept_length per decode"; for u in $D1 $D2; do echo "   $u $(curl -s -m 10 http://$u:30000/get_server_info | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('avg_spec_accept_length'))")"; done
log "=== TC-7: decode-phase aborts through the router"; $PY $S/abort_test.py --url $LB --prompt-file $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --phase decode --abort-after 5 --repeat 2 --server-info http://$D1:30000 --server-info http://$D2:30000 2>&1 | tail -n 4 | sed 's/^/   /'
log "=== gauges and counters after"; gauges; log "=== status after"; bash $S/pd4.sh status | sed 's/^/   /'; log "phase p7 done"
