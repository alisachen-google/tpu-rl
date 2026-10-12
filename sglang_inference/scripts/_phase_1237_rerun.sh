#!/bin/bash
# #1237 rerun on the TPU-patch head: Qwen3-VL-30B-A3B decoder compiled via sglang_tpu/patches.py annotation.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/qwen3-vl; OUT=$R/fix_validate_1136; mkdir -p $OUT; REC=$HOME/pd_disagg/recipes/qwen3-vl_G_128k_bfloat16.json
log(){ echo "$(date -u +%FT%TZ) [1237r] $*"; }
cd $HOME/sglang-torchtpu && git fetch -q origin qwen3-vl-moe-decoder-compile && git checkout -q --detach origin/qwen3-vl-moe-decoder-compile && log "=== fix2: $(git log -1 --format='%h %s' | cut -c1-90)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 5400 bash $S/pd.sh golden $REC G_qwen_1237_fix2 b 2>&1 | grep -E 'ready|NOT'
L=$R/runs/G_qwen_1237_fix2/golden/server.log
echo "   policy lines (TP0): $(grep -E 'TP0 EP0.*(CompilePolicy|compile polic|decode steps run eager)' $L | grep -vE 'precompile: warming|\[[0-9]+/[0-9]+\]' | head -n 6 | sed -E 's/.*TP0 EP0\] //' | cut -c1-150 | tr '\n' '|')"
echo "   eager-decoder warning present: $(grep -c 'decode steps run eager' $L)"
curl -s -m 1500 $URL/generate -H 'Content-Type: application/json' -d '{"text":"Write a long story about a lighthouse keeper.","sampling_params":{"temperature":0,"max_new_tokens":1024,"ignore_eos":true}}' > $OUT/story_fix2.json
echo "   story 1024 tokens: $(python3 -c "import json; d=json.load(open('$OUT/story_fix2.json')); m=d.get('meta_info',{}); print('completion', m.get('completion_tokens'), 'e2e_s', round(m.get('e2e_latency',0),1), 'tok/s', round(m.get('completion_tokens',0)/max(m.get('e2e_latency',1),1e-9),1))" 2>&1)"
echo "   decode gen throughput (last 3 batches): $(grep -E 'Decode batch' $L | tail -n 3 | grep -oE 'gen throughput \(token/s\): [0-9.]+' | sed -E 's/.*: //' | tr '\n' ' ')"
$PY $S/golden_diff.py generate --prompts $P/8192.jsonl --url $URL --out $OUT/8k_fix2.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 900 --stream > $OUT/8k_fix2.log 2>&1
echo "   8k prompts: $(grep -cE 'tokens=64' $OUT/8k_fix2.log) completed; $(grep -oE 'ttft=[0-9.]+ms' $OUT/8k_fix2.log | head -n 1); $(grep -oE 'e2e=[0-9.]+ms' $OUT/8k_fix2.log | head -n 1); errors $(grep -ciE 'error|traceback' $OUT/8k_fix2.log)"
echo "   recompiles after warmup: $(grep -cE 'Detected recompile|__recompiles' $L); exceptions: $(grep -cE 'Scheduler hit an exception' $L); health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)"
bash $S/pd.sh stop >/dev/null 2>&1
log "=== output identity vs main baseline and vs the previous fix head"; $PY $S/golden_diff.py diff --a $OUT/8k_main.jsonl --b $OUT/8k_fix2.jsonl --report $OUT/8k_diff2.json --expect-tokens 64 2>&1 | tail -n 4
python3 - <<'PY'
import json
a=json.load(open('/home/alisachen_google_com/pd_disagg/results/fix_validate_1136/story_main.json')); b=json.load(open('/home/alisachen_google_com/pd_disagg/results/fix_validate_1136/story_fix2.json')); c=json.load(open('/home/alisachen_google_com/pd_disagg/results/fix_validate_1136/story_fix.json'))
print('   story text identical to main:', a.get('text')==b.get('text'), '| to previous fix head:', c.get('text')==b.get('text'), '| main e2e', round(a['meta_info']['e2e_latency'],1), 's; fix e2e', round(b['meta_info']['e2e_latency'],1), 's; previous fix e2e', round(c['meta_info']['e2e_latency'],1), 's')
PY
cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1237 rerun done"
