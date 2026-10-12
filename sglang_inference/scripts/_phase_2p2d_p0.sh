#!/bin/bash
# 2P/2D phase 0-2: bring-up checks, routing correctness, 8k differential vs the TP-8 golden, decode-phase abort. Assumes pd2.sh start already ran.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d/p0; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
A=10.128.15.217; B=10.128.0.90; LB=http://$A:8080; P="http://$A:30000,http://$A:30001"; D="http://$B:30000,http://$B:30001"
log(){ echo "$(date -u +%FT%TZ) [2p2d-p0] $*"; }
log "=== status"; bash $S/pd2.sh status | sed 's/^/   /'
log "=== compile audit per engine (serving-time recompiles so far)"; for e in d1 d2; do echo "   $e: $(bash $S/compile_audit.sh $R/runs/2p2d_bringup/$e/server.log 2>/dev/null | tail -n 2 | tr '\n' ' ')"; done
log "=== routing probe"; $PY $S/route_probe.py --router $LB --prefills $P --decodes $D --n 24 --concurrency 4 2>&1 | tee $OUT/route_probe.txt | sed 's/^/   /' | cut -c1-240
log "=== TC-1 differential: 8k prompts, greedy 1024 tokens, c=1, vs TP-8 golden"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url $LB --out $OUT/8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen_8192.log 2>&1; $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_8192.json 2>&1 | tail -n 6 | sed 's/^/   /' | cut -c1-220
log "=== TC-7 decode-phase client abort through the router"; $PY $S/abort_test.py --url $LB --prompt-file $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --phase decode --abort-after 5 --repeat 2 --server-info http://$A:30000 --server-info http://$A:30001 --server-info http://$B:30000 --server-info http://$B:30001 --settle 60 > $OUT/abort_decode.log 2>&1; tail -n 6 $OUT/abort_decode.log | sed 's/^/   /' | cut -c1-220
log "=== engine health after"; bash $S/pd2.sh status | sed 's/^/   /'; log "phase p0 done"
