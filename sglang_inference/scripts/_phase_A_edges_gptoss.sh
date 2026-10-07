#!/usr/bin/env bash
# Config A (gpt-oss, 128k shape) edge-case chain through the mini-LB. Sequential so each step sees a quiet system.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results/A; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; M=gpt-oss; P=$HOME/pd_disagg/prompts/$M
log(){ echo "$(date -u +%FT%TZ) $*"; }
state(){ for u in $PRE $DEC; do curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|token_usage|num_used_tokens)' | awk -v u=$u '{printf "%s %s=%s  ", u, $1, $2} END{print ""}'; done; }
settle(){ sleep ${1:-30}; log "state after settle:"; state; }
mkdir -p $R/tiny/$M $R/params/$M $R/chat/$M $R/conc/$M $R/retract/$M $R/abort/$M
log "baseline state"; state
log "(0) PD self-determinism: repeat 32768 and 131072 at c=1 and diff against the first PD pass"
mkdir -p $R/tc1/$M/rep2
for L in 32768 131072; do
  $PY $S/golden_diff.py generate --prompts $P/$L.jsonl --url $LB --out $R/tc1/$M/rep2/$L.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 --timeout 1800 --stream > $R/tc1/$M/rep2/generate_$L.log 2>&1
  $PY $S/golden_diff.py diff --a $R/tc1/$M/$L.jsonl --b $R/tc1/$M/rep2/$L.jsonl --report $R/tc1/$M/rep2/pd_self_$L.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | sed 's/^/   PDvsPD /'
done
settle 15
log "(1) tiny SWA/page/chunk boundary sweep, c=1, 160 new tokens"
$PY $S/golden_diff.py generate --prompts $P/tiny.jsonl --url $LB --out $R/tiny/$M/tiny.jsonl --max-new-tokens 160 --ignore-eos --return-logprob --concurrency 1 --timeout 600 > $R/tiny/$M/generate.log 2>&1
$PY - $P/tiny.jsonl $R/tiny/$M/tiny.jsonl <<'EOF' | tee $R/tiny/$M/needles.txt
import json,sys
pr={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}; out={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[2])}
bad=0
for k,p in pr.items():
    o=out.get(k,{}); t=o.get('text') or ''; nd=p.get('needle'); ok=(nd is None) or (nd in t)
    bad+= (not ok) or bool(o.get('error')) or (o.get('completion_tokens')!=160)
    print(f"{k:12s} isl={p['isl']:6d} tokens={o.get('completion_tokens')} finish={o.get('finish_reason')} needle={'n/a' if nd is None else ('FOUND' if ok else 'MISSING')} err={o.get('error')} head={t[:40]!r}")
print(f"tiny: {len(pr)} prompts, {bad} problems")
EOF
settle 20
log "(2) sampling-params matrix (PD only; compare to golden later)"
$PY $S/params_matrix.py --a $LB --prompt-file $P/8192.jsonl --report $R/params/$M/params_pd_A.json --timeout 300 > $R/params/$M/params_pd_A.txt 2>&1; log "params rc=$?"
settle 20
log "(3) chat suite (harmony/reasoning/tools/stream) PD only"
$PY $S/chat_suite.py --model gpt-oss-120b --kind gptoss --a $LB --report $R/chat/$M/chat_pd_A.json --timeout 600 > $R/chat/$M/chat_pd_A.txt 2>&1; log "chat rc=$?"
settle 20
log "(4) concurrency sweep c=1,2,4 at ISL 131072 / OSL 256 (bench_serving random)"
cd $HOME/sglang-torchtpu
for c in 1 2 4; do
  $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 131072 --random-output-len 256 --random-range-ratio 1.0 \
     --num-prompts $((c*2)) --max-concurrency $c --request-rate inf --disable-ignore-eos --extra-request-body '{"temperature":0}' \
     --output-file $R/conc/$M/random_131072_256_c$c.jsonl > $R/conc/$M/random_131072_256_c$c.txt 2>&1
  grep -E 'Successful|Failed|Mean TTFT|P99 TTFT|Mean ITL|P99 ITL|Output token throughput|Total token throughput' $R/conc/$M/random_131072_256_c$c.txt | sed "s/^/   c=$c /"
  settle 15
done
log "(5) retraction test: 8 x 131072-token requests at c=8 with 1024 new tokens (pool 786k < 8 x 132k)"
$PY - $P/131072.jsonl $R/retract/$M/prompts8.jsonl <<'EOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(8):
        r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_dup{i}"; f.write(json.dumps(r)+'\n')
EOF
$PY $S/golden_diff.py generate --prompts $R/retract/$M/prompts8.jsonl --url $LB --out $R/retract/$M/out8.jsonl --max-new-tokens 1024 --ignore-eos --concurrency 8 --timeout 3600 --stream > $R/retract/$M/generate.log 2>&1
grep -E 'tokens=|errors' $R/retract/$M/generate.log | cut -c1-160
$PY - $P/131072.jsonl $R/retract/$M/out8.jsonl <<'EOF' | tee $R/retract/$M/check.txt
import json,sys
pr={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}
outs=[json.loads(l) for l in open(sys.argv[2])]
groups={}
for o in outs:
    base=o['id'].rsplit('_dup',1)[0]; groups.setdefault(base,[]).append(o)
for base,g in groups.items():
    nd=list(pr[base]['needles'].values()); texts=[x.get('text') or '' for x in g]
    found=[sum(w in t for w in nd) for t in texts]
    ident=len(set(texts))
    print(f"{base}: {len(g)} copies, needles found per copy={found}, distinct texts={ident}, tokens={[x.get('completion_tokens') for x in g]}, errors={[x.get('error') for x in g if x.get('error')]}")
EOF
grep -ciE 'retract' $HOME/pd_disagg/results/runs/A_gptoss_128k_run1/decode/server.log | sed 's/^/   decode log retraction lines: /'
settle 60
log "(6) client aborts: prefill phase then decode phase"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 2 --repeat 3 --server-info $PRE --server-info $DEC --settle 60 --report $R/abort/$M/abort_prefill.json > $R/abort/$M/abort_prefill.txt 2>&1; log "abort prefill rc=$?"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase decode --abort-after 5 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 60 --report $R/abort/$M/abort_decode.json > $R/abort/$M/abort_decode.txt 2>&1; log "abort decode rc=$?"
settle 30
log "(7) compile audit + final state"
bash $S/compile_audit.sh $HOME/pd_disagg/results/runs/A_gptoss_128k_run1/decode/server.log decode_A | tee $R/compile_audit_decode_A.txt
state
log "(8) GSM8K-100 through the LB (absolute accuracy; golden scored 0.99)"
cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $LB --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 > $R/gsm8k_pd_A.log 2>&1; grep -E 'gsm8k_score|Error|Traceback' $R/gsm8k_pd_A.log | head -3
settle 30; state
log "phase A edges done"
