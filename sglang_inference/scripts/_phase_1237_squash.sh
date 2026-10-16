#!/usr/bin/env bash
# #1237 squashed head on top of main: monolithic Qwen3-VL-30B-A3B golden on host b (this VM).
# Checks: decoder policy installed on Qwen3MoeLLMModel (from main's core annotations), the new
# "decode steps run eager" warning absent, 1024-token story speed, 8k prompts, recompiles, exceptions.
# d1 of the 2P/2D group is stopped for the run and restored to main afterwards.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; ME=alisachen_google_com
P1=10.128.15.217; P2=10.128.15.239; D1=10.128.0.90; D2=10.128.0.20; LB=http://$P1:8080
WT=$HOME/wt-1136; URL=http://$D1:30000; P=$HOME/pd_disagg/prompts/qwen3-vl; OUT=$R/fix_validate_1237; mkdir -p $OUT
REC=$HOME/pd_disagg/recipes/qwen3-vl_G_128k_bfloat16.json; RUN=G_qwen_1237_squash
log(){ echo "$(date -u +%FT%TZ) [1237] $*"; }
kill_engine(){ pkill -9 -u $ME -f 'sglang::[s]cheduler'; pkill -9 -u $ME -f 'sglang.launch_[s]erver'; pkill -9 -u $ME -f 'benchmark.tpu.launch_[s]erver'; pkill -9 -u $ME -f 'sglang::[d]etokenizer'; sleep 3
  local t0=$(date +%s); while curl -sf -m 3 $URL/health >/dev/null && (( $(date +%s)-t0 < 90 )); do sleep 3; done; log "   b engine down after $(( $(date +%s)-t0 )) s"; }
start_engine(){ local recipe=$1 run=$2 root=${3:-/home/$ME/sglang-torchtpu}; ( cd /tmp && exec >/dev/null 2>&1 </dev/null; SGLANG_ROOT=$root PD_HOSTS=$P1,$P2,$D1,$D2 setsid nohup bash $S/launch_rank.sh $HOME/pd_disagg/recipes/$recipe 2 $run & ); }
wait_up(){ local t0=$(date +%s); until curl -sf -m 5 $URL/health >/dev/null || (( $(date +%s)-t0 > 1500 )); do sleep 15; done; log "   d1 health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health) after $(( $(date +%s)-t0 )) s"; }
traffic(){ local n=$1 c=$2 tok=$3 file=$4; $PY - $n $c $tok $LB > $file <<'PY'
import json,sys,time,concurrent.futures as cf,urllib.request
n,c,tok,lb=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
text=json.loads(open('/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl').readline())['text']
def one(i):
    body=json.dumps({'text':f'request {i} {time.time():.3f}: '+text,'sampling_params':{'max_new_tokens':tok,'temperature':0,'ignore_eos':True},'stream':False}).encode()
    try:
        r=urllib.request.urlopen(urllib.request.Request(lb+'/generate',data=body,headers={'Content-Type':'application/json'}),timeout=600); code=r.status; r.read()
    except urllib.error.HTTPError as e: code=e.code
    except Exception as e: code=f'ERR:{type(e).__name__}'
    return str(code)
with cf.ThreadPoolExecutor(c) as ex:
    for line in ex.map(one, range(n)): print(line, flush=True)
PY
  sort $file | uniq -c | awk '{printf "%s x%s ", $2, $1}'; }

log "=== squashed head: $(git -C $WT log -1 --format='%h %s' | cut -c1-100); main: $(git -C $WT rev-parse --short origin/main)"
log "=== A. stop d1 of the P/D group, start the monolithic Qwen3-VL golden on b from $WT"
kill_engine; sleep 5
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
SGLANG_ROOT=$WT timeout 5400 bash $S/pd.sh golden $REC $RUN b 2>&1 | grep -E 'ready|NOT|REFUS' | sed 's/^/   /'
L=$R/runs/$RUN/golden/server.log
log "   launched from: $(grep -o 'sglang=[0-9a-f]*' $R/runs/$RUN/golden/launch_plan.txt | head -1)   (must be the squashed head)"
log "   health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)"
log "=== B. compile policies"
echo "   policies: $(grep -oE '(VisionTpuCompilePolicy: installed [a-z0-9_]+|Installed Qwen3VLDecoderTpuCompilePolicy on [A-Za-z0-9_.]+|decode steps run eager|declares no TPU compile policies[^.]*)' $L | sort | uniq -c | sed 's/^ */ /' | tr '\n' ';')"
echo "   'decode steps run eager' warning lines: $(grep -c 'decode steps run eager' $L)   (expected 0)"
echo "   policy lines (TP0): $(grep -E 'TP0 EP0.*(CompilePolicy|compile polic)' $L | grep -vE 'precompile: warming|\[[0-9]+/[0-9]+\]' | head -n 4 | sed -E 's/.*TP0 EP0\] //' | cut -c1-140 | tr '\n' '|')"
log "=== C. generation"
t0=$(date +%s.%N); curl -s -m 1500 $URL/generate -H 'Content-Type: application/json' -d '{"text":"Write a long story about a lighthouse keeper.","sampling_params":{"temperature":0,"max_new_tokens":1024,"ignore_eos":true}}' > $OUT/story.json; t1=$(date +%s.%N)
echo "   story 1024 tokens: $(python3 -c "import json; d=json.load(open('$OUT/story.json')); m=d.get('meta_info',{}); print('completion', m.get('completion_tokens'), 'e2e_ms', round(m.get('e2e_latency',0)*1000) if m.get('e2e_latency') else '?')" 2>/dev/null) wall $(python3 -c "print(round($t1-$t0,1))")s   (eager main was 161.5 s; compiled 6.3 s)"
echo "   decode gen throughput (last 3 batches): $(grep -E 'Decode batch' $L | tail -n 3 | grep -oE 'gen throughput \(token/s\): [0-9.]+' | sed -E 's/.*: //' | tr '\n' ' ')"
$PY $S/golden_diff.py generate --prompts $P/8192.jsonl --url $URL --out $OUT/8k.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 900 --stream > $OUT/8k.log 2>&1
echo "   8k prompts: $(grep -oE 'tokens=[0-9]+' $OUT/8k.log | tr '\n' ' '); $(grep -oE 'ttft=[0-9.]+ms' $OUT/8k.log | tr '\n' ' '); $(grep -oE 'e2e=[0-9.]+ms' $OUT/8k.log | tr '\n' ' '); errors $(grep -cE 'ERROR' $OUT/8k.log)"
echo "   identity vs previous fix head (8k, 64 greedy tokens): $($PY $S/golden_diff.py diff --a $R/fix_validate_1136/8k_fix2.jsonl --b $OUT/8k.jsonl --report $OUT/8k_diff.json --expect-tokens 64 2>&1 | tail -n 1 | cut -c1-160)"
echo "   story identical to previous fix head: $(python3 -c "import json; a=json.load(open('$R/fix_validate_1136/story_fix2.json')); b=json.load(open('$OUT/story.json')); print(a.get('text')==b.get('text'))")"
echo "   recompiles after warmup: $(grep -cE 'Detected recompile|__recompiles' $L); exceptions: $(grep -cE 'Scheduler hit an exception|Traceback' $L); health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)"
log "=== D. restore d1 to main in the P/D group"
kill_engine; start_engine gpt-oss_B_128k_2p2d.json 2p2d_tp8_B3; wait_up
log "   traffic via router: $(traffic 4 2 32 $OUT/D_traffic.txt)"; bash $S/pd4.sh status | sed 's/^/   /'
log "phase 1237 done"
