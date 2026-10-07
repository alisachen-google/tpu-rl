#!/usr/bin/env bash
# TC-1 baseline functional + long-context sweep against a PD load balancer (or any endpoint).
#   run_tc1.sh <model: gpt-oss|qwen3-vl> <lb_url> <out_dir> <lengths csv> <golden_dir> [concurrency]
# Steps: (1) workers/health/needle via verify_pd_disagg.py  (2) per-ISL generation through the LB with golden_diff generate
#        (3) differential diff vs goldens  (4) boundary: context_length-1024 accepted, context_length+1 → clean 4xx
set -uo pipefail
model=${1:?model}; url=${2:?lb url}; out=${3:?out}; lengths=${4:?lengths}; golden=${5:?golden dir}; conc=${6:-1}
PY=${PYTHON:-$HOME/venv/sglang/bin/python}; S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"; exec > >(tee -a "$out/tc1.log") 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
echo "=== $(date -u +%FT%TZ) TC-1 model=$model url=$url lengths=$lengths conc=$conc"
echo "--- (1) verify_pd_disagg (workers, health, smoke, needle 1k/8k/32k)"
"$PY" "$S/verify_pd_disagg.py" --url "$url" --workers --lengths 1024,8192,32768 --timeout 1800 2>&1 | tee "$out/verify_short.txt"
for L in ${lengths//,/ }; do
  p="$HOME/pd_disagg/prompts/$model/$L.jsonl"; [ -f "$p" ] || { echo "missing $p"; continue; }
  echo "--- (2) ISL=$L through LB"; t0=$(date +%s)
  "$PY" "$S/golden_diff.py" generate --prompts "$p" --url "$url" --out "$out/$L.jsonl" --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency "$conc" --timeout 7200 --stream
  echo "    done in $(( $(date +%s) - t0 )) s"
  if [ -f "$golden/$L.jsonl" ]; then
    echo "--- (3) diff vs golden ISL=$L"
    "$PY" "$S/golden_diff.py" diff --a "$golden/$L.jsonl" --b "$out/$L.jsonl" --report "$out/golden_diff_$L.json" --expect-tokens 1024 --exact-prefix-tokens 256 --min-agreement 0.95 2>&1 | tail -n 15
    echo "    diff rc=$?"
  else echo "    (no golden for $L yet)"; fi
done
echo "=== $(date -u +%FT%TZ) TC-1 finished"
