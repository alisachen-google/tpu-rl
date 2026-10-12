#!/bin/bash
# TP-4 monolithic golden on host b (chips 0,1) after the TP-4 pair baseline: 8k greedy outputs, then diff PD(TP-4) vs golden(TP-4) and golden(TP-4) vs golden(TP-8).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d/tp4_golden; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
log(){ echo "$(date -u +%FT%TZ) [tp4-golden] $*"; }
while pgrep -f '_phase_tp4_[1]p1d.sh' >/dev/null; do sleep 15; done
log "stopping d1 and the b-side router (p1 on a stays up)"; pkill -u $(id -un) -f 'sglang_router.launch_router.*--port 8081'; bash $S/cleanup.sh >/dev/null 2>&1; sleep 5
until [ -z "$(sudo -n fuser /dev/vfio/0 2>/dev/null)" ]; do sleep 5; done
mkdir -p $R/runs/2p2d_golden_tp4/golden; (ENABLE_COMPILE_CACHE=1 setsid nohup bash $S/launch_engine.sh $HOME/pd_disagg/recipes/2p2d/gpt-oss_G_128k_tp4.json 2p2d_golden_tp4 golden > $R/runs/2p2d_golden_tp4/launcher_golden.out 2>&1 < /dev/null &)
for i in $(seq 1 200); do curl -sf -m 5 http://10.128.0.90:30002/health >/dev/null 2>&1 && { log "TP-4 golden up after $((i*15)) s"; break; }; sleep 15; done
curl -sf -m 5 http://10.128.0.90:30002/health >/dev/null || { log "golden NOT ready; aborting phase"; exit 1; }
log "generate 8k (greedy 1024, c=1)"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url http://10.128.0.90:30002 --out $OUT/8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen.log 2>&1
log "=== PD(TP-4) vs golden(TP-4):"; $PY $S/golden_diff.py diff --a $OUT/8192.jsonl --b $R/2p2d/tp4_1p1d/8192.jsonl --report $OUT/diff_pd_vs_golden_tp4.json 2>&1 | tail -n 4 | sed 's/^/   /' | cut -c1-200
log "=== golden(TP-4) vs golden(TP-8):"; $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_golden_tp4_vs_tp8.json 2>&1 | tail -n 4 | sed 's/^/   /' | cut -c1-200
log "=== golden(TP-4) repeat for determinism:"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url http://10.128.0.90:30002 --out $OUT/8192_rep2.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen2.log 2>&1; $PY $S/golden_diff.py diff --a $OUT/8192.jsonl --b $OUT/8192_rep2.jsonl --report $OUT/diff_rep.json 2>&1 | tail -n 2 | sed 's/^/   /' | cut -c1-200
bash $S/cleanup.sh >/dev/null 2>&1; log "phase tp4-golden done (b freed; p1 on a still up)"
