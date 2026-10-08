#!/usr/bin/env bash
# HiCache host-reload timing on an idle Config C pair (gpt-oss): cold prefill TTFT vs HBM-hit TTFT vs host-hit TTFT for 32k prompts,
# plus output identity across the three. Runs after the B/C/D/E chain.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/C/gpt-oss/hicache_reload; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [hicache-reload] $*"; }
m(){ curl -s -m 10 $PRE/metrics | grep -E '^sglang:(hicache_host_used_tokens|hicache_backup_tokens_total|cache_hit_rate)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' '; }
gen(){ $PY $S/golden_diff.py generate --prompts $P/32768.jsonl --url $LB --out $OUT/$1.jsonl --max-new-tokens 64 --ignore-eos --return-logprob --concurrency 1 --timeout 900 --stream > $OUT/$1.log 2>&1; grep -E 'ttft=' $OUT/$1.log | sed "s/^/   $1 /" | cut -c1-140; }
hits(){ ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n grep -E 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | grep -oE '#new-token: [0-9]+, #cached-token: [0-9]+, #hbm-hit-token: [0-9]+, #host-hit-token: [0-9]+' | tail -n ${1:-2}" | sed 's/^/   prefill /'; }
until grep -q 'phase BCDE done' $R/phase_BCDE.log 2>/dev/null; do sleep 120; done; log "chain done"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; log "TPU free"
RUN=C_gpt-oss_128k_hicache; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k.json $RUN 2>&1 | grep -E 'ready|NOT'
curl -sf -m 30 $LB/health_generate >/dev/null || { log "pair did not come up"; bash $S/pd.sh stop; exit 2; }
log "(1) cold: 32k prompts, c=1, streaming"; gen cold; hits
log "(2) HBM hit: same prompts again"; gen hbm_hit; hits; echo "   $(m)"
log "(3) evict: 12 random 131k prompts at c=4 (≈1.6M tokens > 786k device pool)"
cd $HOME/sglang-torchtpu && $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 131072 --random-output-len 16 --random-range-ratio 1.0 --num-prompts 12 --max-concurrency 4 --request-rate inf --disable-ignore-eos --extra-request-body '{"temperature":0}' --output-file $OUT/evict.jsonl > $OUT/evict.txt 2>&1; cd $HOME; echo "   $(m)"
sleep 20
log "(4) host hit: same 32k prompts on the idle pair"; gen host_hit; hits; echo "   $(m)"
log "(5) identity across cold / hbm_hit / host_hit"
$PY - $OUT <<'PY'
import json,sys,os
o=sys.argv[1]; r={n:{json.loads(l)['id']:json.loads(l) for l in open(f'{o}/{n}.jsonl')} for n in ('cold','hbm_hit','host_hit')}
for k in r['cold']:
    c,h,s=r['cold'][k],r['hbm_hit'][k],r['host_hit'][k]
    print(f"   {k}: ttft cold={c['ttft_ms']:.0f}ms hbm_hit={h['ttft_ms']:.0f}ms host_hit={s['ttft_ms']:.0f}ms | cached {c.get('cached_tokens')}/{h.get('cached_tokens')}/{s.get('cached_tokens')} | text cold==hbm {c['text']==h['text']} hbm==host {h['text']==s['text']} | first_logprob {c.get('first_token_logprob'):.5f}/{h.get('first_token_logprob'):.5f}/{s.get('first_token_logprob'):.5f}")
PY
bash $S/pd.sh stop >/dev/null 2>&1; log "phase hicache reload done"
