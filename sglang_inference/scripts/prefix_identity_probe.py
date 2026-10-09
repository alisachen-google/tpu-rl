#!/usr/bin/env python3
"""Cold vs prefix-cached greedy outputs at c=1 (Config B). Separates 'priority' from 'cache hit' as the cause of a divergence."""
import argparse, json, requests, time
def gen(url, text, n, prio=None):
    body={"text":text,"sampling_params":{"temperature":0,"max_new_tokens":n,"ignore_eos":True}}
    if prio is not None: body["priority"]=prio
    d=requests.post(url.rstrip("/")+"/generate",json=body,timeout=900).json(); m=d.get("meta_info",{}); return d.get("text",""), m.get("cached_tokens"), m.get("first_token_latency")
def div(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    return i
ap=argparse.ArgumentParser(); ap.add_argument("--url",required=True); ap.add_argument("--prompts",required=True); ap.add_argument("--n",type=int,default=256); ap.add_argument("--priority",action="store_true")
a=ap.parse_args()
for isl in (8192,32768):
    text=json.loads(open(f"{a.prompts}/{isl}.jsonl").readline())["text"]+f" [cold-vs-cached {isl} {time.time():.0f}]"  # unique suffix: cold on first use
    p0=0 if a.priority else None; p1=50 if a.priority else None
    cold,c0,t0=gen(a.url,text,a.n,p0); hit,c1,t1=gen(a.url,text,a.n,p1); hit2,c2,t2=gen(a.url,text,a.n,p0)
    print(f"[prefix-identity {isl}{' prio' if a.priority else ''}] cold cached={c0} ttft={t0 and round(t0,2)} | hit cached={c1} ttft={t1 and round(t1,2)} identical={cold==hit} first_div_char={div(cold,hit)}/{len(cold)} | hit2 cached={c2} identical_to_hit={hit==hit2}", flush=True)
