#!/usr/bin/env bash
# Generate golden (or PD) outputs for a prompt set once the endpoint is healthy.
#   run_golden.sh <model: gpt-oss|qwen3-vl> <url> <out_dir> <lengths csv> [concurrency] [max_new_tokens]
# Writes <out_dir>/<isl>.jsonl per length plus timing in <out_dir>/run.log
set -uo pipefail
model=${1:?model}; url=${2:?url}; out=${3:?out_dir}; lengths=${4:?lengths}; conc=${5:-2}; mnt=${6:-1024}
PY=${PYTHON:-$HOME/venv/sglang/bin/python}; S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"; exec >> "$out/run.log" 2>&1
echo "=== $(date -u +%FT%TZ) run_golden model=$model url=$url lengths=$lengths conc=$conc max_new_tokens=$mnt"
until curl -sf -m 5 "$url/health" >/dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) endpoint healthy"
curl -s -m 10 "$url/get_server_info" > "$out/server_info.json"
for L in ${lengths//,/ }; do
  p="$HOME/pd_disagg/prompts/$model/$L.jsonl"; [ -f "$p" ] || { echo "missing $p"; continue; }
  t0=$(date +%s); mnt_L=$mnt; [ "$L" = tiny ] && mnt_L=160
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" "$S/golden_diff.py" generate --prompts "$p" --url "$url" --out "$out/$L.jsonl" \
      --max-new-tokens "$mnt_L" --ignore-eos --return-logprob --concurrency "$conc" --timeout 7200 --stream
  echo "$(date -u +%FT%TZ) isl=$L done in $(( $(date +%s) - t0 )) s rc=$?"
  python3 -c "
import json,sys
for l in open('$out/$L.jsonl'):
    d=json.loads(l); print('   ', d.get('id'), 'tokens=',d.get('completion_tokens'), 'finish=',d.get('finish_reason'), 'ttft_ms=',round(d.get('ttft_ms') or 0), 'e2e_ms=',round(d.get('e2e_ms') or 0), 'err=',d.get('error'))" 2>/dev/null
done
echo "=== $(date -u +%FT%TZ) run_golden finished"
