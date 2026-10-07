#!/usr/bin/env bash
# TC-2 prefix caching (prefill side = Config B/C, decode side = Config E).
#   run_tc2.sh <model> <lb_url> <prefill_url> <decode_url> <out_dir> <prefix_len> [question_len=16384] [conc=2]
# Pass 1 cold: 8 prompts sharing a <prefix_len> system prefix (bench_serving generated-shared-prefix).
# Pass 2 warm: same seed again -> cached_tokens >= 0.95*prefix (page-rounded), TTFT <= 50% cold, identical output text.
# Pass 3 full-prompt hit: golden_diff generate on the same prompt file twice; second run must report cached_tokens ~ prompt_tokens and identical text.
set -uo pipefail
model=${1:?}; lb=${2:?}; pre=${3:?}; dec=${4:?}; out=${5:?}; plen=${6:?}; qlen=${7:-16384}; conc=${8:-2}
PY=${PYTHON:-$HOME/venv/sglang/bin/python}; S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$out"; exec > >(tee -a "$out/tc2.log") 2>&1
echo "=== $(date -u +%FT%TZ) TC-2 model=$model lb=$lb prefix=$plen question=$qlen conc=$conc"
flush() { for u in "$pre" "$dec"; do printf '%s flush_cache -> %s\n' "$u" "$(curl -s -m 20 -X POST "$u/flush_cache" -o /dev/null -w '%{http_code}')"; done; }
cache_metrics() { for u in "$pre" "$dec"; do echo "$u: $(curl -s -m 10 "$u/metrics" | grep -E '^sglang:(cache_hit_rate|cached_tokens|prompt_tokens_total|num_running_reqs)' | tr '\n' ' ' | cut -c1-300)"; done; }
bench() { # tag
  cd "$HOME/sglang-torchtpu" && "$PY" -m sglang.bench_serving --backend sglang --base-url "$lb" --dataset-name generated-shared-prefix \
    --gsp-num-groups 1 --gsp-prompts-per-group 8 --gsp-system-prompt-len "$plen" --gsp-question-len "$qlen" --gsp-output-len 256 \
    --max-concurrency "$conc" --request-rate inf --num-prompts 8 --disable-ignore-eos --seed 1234 \
    --extra-request-body '{"temperature":0}' --output-file "$out/gsp_$1.jsonl" --output-details 2>&1 | grep -E 'Successful|Mean TTFT|Median TTFT|P99 TTFT|Mean ITL|Total input|cached|Request throughput|Error|error'
}
flush; echo "--- pass 1 (cold)"; bench cold; cache_metrics
echo "--- pass 2 (warm, same prompts)"; bench warm; cache_metrics
"$PY" - "$out/gsp_cold.jsonl" "$out/gsp_warm.jsonl" <<'EOF'
import json,sys
def load(p):
    rows=[json.loads(l) for l in open(p)]
    return rows[-1] if len(rows)==1 else rows
a,b=load(sys.argv[1]),load(sys.argv[2])
def pick(r):
    d=r if isinstance(r,dict) else r[-1]
    return d.get('mean_ttft_ms'), d.get('median_ttft_ms'), d.get('completed'), d.get('output_lens') and len(d['output_lens'])
print('cold mean/median TTFT ms, completed:', pick(a)); print('warm mean/median TTFT ms, completed:', pick(b))
try:
    ga=a if isinstance(a,dict) else a[-1]; gb=b if isinstance(b,dict) else b[-1]
    ta=ga.get('generated_texts'); tb=gb.get('generated_texts')
    if ta and tb: print('identical outputs cold vs warm:', sum(x==y for x,y in zip(ta,tb)), '/', len(ta))
    if gb.get('median_ttft_ms') and ga.get('median_ttft_ms'): print('warm/cold median TTFT ratio:', round(gb['median_ttft_ms']/ga['median_ttft_ms'],3), '(gate <= 0.5)')
except Exception as e: print('compare error', e)
EOF
echo "--- pass 3 (full-prompt hit via /generate, 8k prompt set twice)"
p="$HOME/pd_disagg/prompts/$model/8192.jsonl"
"$PY" "$S/golden_diff.py" generate --prompts "$p" --url "$lb" --out "$out/full_hit_1.jsonl" --max-new-tokens 128 --ignore-eos --timeout 1800
"$PY" "$S/golden_diff.py" generate --prompts "$p" --url "$lb" --out "$out/full_hit_2.jsonl" --max-new-tokens 128 --ignore-eos --timeout 1800
"$PY" - "$out/full_hit_1.jsonl" "$out/full_hit_2.jsonl" <<'EOF'
import json,sys
a={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}; b={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[2])}
for k in a:
    x,y=a[k],b.get(k,{})
    print(k, 'run1 cached=',x.get('cached_tokens'),'prompt=',x.get('prompt_tokens'),'| run2 cached=',y.get('cached_tokens'),'ttft1/2 ms=',round(x.get('ttft_ms') or 0),round(y.get('ttft_ms') or 0),'identical=',x.get('text')==y.get('text'),'err=',x.get('error') or y.get('error'))
EOF
cache_metrics
echo "=== $(date -u +%FT%TZ) TC-2 finished"
