#!/usr/bin/env python3
"""priority_probe2.py - second #1161 battery: starvation, transfer-queue reorder, prefix reuse across priorities, metrics label,
disconnect storm, HiCache host hit across priorities, mixed-priority soak, aborts around preemption (monolithic).
  --mode np2   (PD Config B, PS-np): B5 B6 B7 F3 F4
  --mode c2    (PD Config C, PS-np): B8
  --mode soak  (PD Config B, PS-np): F1 (--minutes)
  --mode pre2  (monolithic, preemption on): C9 C10 C15
"""
import argparse, json, random, subprocess, sys, threading, time
import requests
from priority_probe import gen, metric, say, hold_slots

def read_prompt(path, k=0):
    rows=[json.loads(l) for l in open(path)]; return rows[k % len(rows)]["text"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--decode", required=True); ap.add_argument("--prefill", default=None)
    ap.add_argument("--mode", required=True, choices=["np2","c2","soak","pre2"]); ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--prompts", default="/home/alisachen_google_com/pd_disagg/prompts/gpt-oss"); ap.add_argument("--minutes", type=float, default=15)
    a=ap.parse_args(); U,D,P=a.url,a.decode,a.prompts
    used=lambda: metric(D,"num_used_tokens",'tp_rank="0"')

    if a.mode=="np2":
        # B5 starvation: 60 s stream of priority-10 short decodes keeping every slot busy; one priority-0 request submitted at t=5 s.
        stop=time.time()+60; lock=threading.Lock(); stream_done=[0]; stream_err=[0]
        def streamer():
            while time.time()<stop:
                r=gen(U,"Keep the slot busy with a short story. "*10,priority=10,max_new_tokens=400,stream=True,timeout=600)
                with lock:
                    stream_done[0]+=1; stream_err[0]+= (r["status"]!=200)
        ths=[threading.Thread(target=streamer,daemon=True) for _ in range(a.slots+2)]; [t.start() for t in ths]; time.sleep(5)
        t0=time.time(); r=gen(U,"Low priority: what is 1+1?",priority=0,max_new_tokens=8,stream=True,timeout=900); low=time.time()-t0
        for t in ths: t.join(timeout=900)
        say("B5-starvation", r["status"]==200, f"priority-0 request under a 60 s priority-10 stream ({stream_done[0]} stream requests, {stream_err[0]} errors): served after {low:.1f}s (stream ended at ~{60-5}s after submit); {'starved until the stream ended' if low>45 else 'admitted while the stream ran'}")
        # B6 transfer-queue reorder: 7 holders leave one slot; a low-priority 131k request and, 0.5 s later, a high-priority 8k request compete for it.
        res={}; ths=hold_slots(U,a.slots-1,0,2500,res); time.sleep(12)
        out={}
        def w(name,text,prio):
            out[name]=gen(U,text,priority=prio,max_new_tokens=8,stream=True,timeout=900); out[name]["t_first"]=out[name]["sent"]+(out[name]["ttft"] or 1e9)
        t1=threading.Thread(target=w,args=("low131k",read_prompt(f"{P}/131072.jsonl"),0)); t2=threading.Thread(target=w,args=("high8k",read_prompt(f"{P}/8192.jsonl"),50))
        t1.start(); time.sleep(0.5); t2.start(); t1.join(900); t2.join(900)
        say("B6-transfer-reorder", out["high8k"]["t_first"]<out["low131k"]["t_first"], f"one free slot: high-priority 8k (sent 0.5 s later) first token at +{out['high8k']['t_first']-out['low131k']['sent']:.1f}s, low-priority 131k at +{out['low131k']['t_first']-out['low131k']['sent']:.1f}s (statuses {out['high8k']['status']}/{out['low131k']['status']})")
        for t in ths: t.join(timeout=900)
        # B7 prefix reuse across priorities (Config B radix on): warm at priority 0, hit at priority 50.
        text=read_prompt(f"{P}/32768.jsonl")
        r0=gen(U,text,priority=0,max_new_tokens=32,timeout=600); r1=gen(U,text,priority=50,max_new_tokens=32,timeout=600)
        c1=requests.post(U.rstrip("/")+"/generate",json={"text":text,"priority":50,"sampling_params":{"temperature":0,"max_new_tokens":32}},timeout=600).json()
        cached=(c1.get("meta_info") or {}).get("cached_tokens"); same=(r0.get("text")==r1.get("text")==c1.get("text"))
        say("B7-prefix-across-priorities", (cached or 0)>30000 and same, f"32k prompt warmed at priority 0, re-sent at priority 50: cached_tokens={cached}, outputs identical={same}, ttft {r0['ttft'] and round(r0['ttft'],2)}s -> {r1['ttft'] and round(r1['ttft'],2)}s")
        # F3 metrics label
        for name,u in (("decode",D),("prefill",a.prefill or D)):
            txt=requests.get(u.rstrip("/")+"/metrics",timeout=10).text; line=[l for l in txt.splitlines() if l.startswith("sglang:num_queue_reqs{")]
            say(f"F3-metrics-label-{name}", any('priority=' in l for l in line), f"num_queue_reqs labels: {line[0][:140] if line else 'none'}")
        # F4 disconnect storm with priorities: 20 streaming 8k requests, mixed priorities, aborted after 2 s, 10 concurrent.
        base=used(); text=read_prompt(f"{P}/8192.jsonl"); errs=[0]
        def ab(prio):
            try:
                r=requests.post(U.rstrip("/")+"/generate",json={"text":text,"priority":prio,"stream":True,"sampling_params":{"temperature":0,"max_new_tokens":2048,"ignore_eos":True}},stream=True,timeout=60)
                t0=time.time()
                for _ in r.iter_lines():
                    if time.time()-t0>2: break
                r.close()
            except Exception: errs[0]+=1
        for batch in range(2):
            ths=[threading.Thread(target=ab,args=(random.choice([0,5,10,50]),)) for _ in range(10)]; [t.start() for t in ths]; [t.join(120) for t in ths]
        time.sleep(30); after=used(); run=metric(D,"num_running_reqs",'tp_rank="0"')
        f=gen(U,"After the storm: say hi.",priority=5,max_new_tokens=8,timeout=120)
        say("F4-abort-storm", errs[0]==0 and (after or 0)<1024 and f["status"]==200, f"20 aborts (10 concurrent, mixed priorities): client errors {errs[0]}, decode used tokens {base}->{after}, running {run}, follow-up HTTP {f['status']}")

    if a.mode=="c2":
        # B8 HiCache host hit across priorities: 8 distinct 131k prompts (> device pool 786k) warm at priority 0, then doc 0 again at priority 50.
        base=read_prompt(f"{P}/131072.jsonl"); docs=[f"[document {i}] "+base for i in range(8)]
        firsts={}
        def w(i):
            firsts[i]=gen(U,docs[i],priority=0,max_new_tokens=16,timeout=900)
        for k in range(0,8,2):
            ths=[threading.Thread(target=w,args=(i,)) for i in (k,k+1)]; [t.start() for t in ths]; [t.join(900) for t in ths]
        ok=sum(1 for v in firsts.values() if v["status"]==200)
        r=requests.post(U.rstrip("/")+"/generate",json={"text":docs[0],"priority":50,"sampling_params":{"temperature":0,"max_new_tokens":16}},timeout=900).json()
        m=r.get("meta_info") or {}; same=(r.get("text")==firsts[0].get("text"))
        say("B8-hicache-across-priorities", ok==8 and (m.get("cached_tokens") or 0)>100000 and same, f"8 x 131k warmed ({ok}/8 ok, > device pool so the first ones are evicted to host); doc 0 re-sent at priority 50: cached_tokens={m.get('cached_tokens')}, ttft={m.get('first_token_latency') and round(m['first_token_latency'],2)}s, output identical={same} (host-hit lines: see prefill log)")

    if a.mode=="soak":
        files={1024:f"{P}/1024.jsonl",8192:f"{P}/8192.jsonl",32768:f"{P}/32768.jsonl"}; texts={k:[json.loads(l)["text"] for l in open(v)] for k,v in files.items()}
        end=time.time()+a.minutes*60; lock=threading.Lock(); stats={"n":0,"err":0,"by_prio":{}}
        def client(seed):
            rnd=random.Random(seed)
            while time.time()<end:
                isl=rnd.choice([1024,1024,8192,8192,32768]); prio=rnd.choice([0,5,10,50]); r=gen(U,rnd.choice(texts[isl]),priority=prio,max_new_tokens=rnd.choice([64,128,256]),stream=True,timeout=900)
                with lock:
                    stats["n"]+=1; stats["err"]+=(r["status"]!=200); d=stats["by_prio"].setdefault(prio,[0,0.0]); d[0]+=1; d[1]+=(r["ttft"] or 0)
        ths=[threading.Thread(target=client,args=(i,),daemon=True) for i in range(a.slots+4)]; [t.start() for t in ths]
        while time.time()<end: time.sleep(60); print(f"[soak] t+{int((a.minutes*60)-(end-time.time()))}s requests={stats['n']} errors={stats['err']} running={metric(D,'num_running_reqs','tp_rank=\"0\"')} queue={metric(D,'num_queue_reqs','tp_rank=\"0\"')} prealloc={metric(D,'num_decode_prealloc_queue_reqs','tp_rank=\"0\"')}", flush=True)
        for t in ths: t.join(timeout=900)
        time.sleep(20); u=used(); run=metric(D,"num_running_reqs",'tp_rank="0"')
        bp={p:(n,round(s/n,1)) for p,(n,s) in sorted(stats["by_prio"].items())}
        say("F1-soak", stats["err"]==0 and (u or 0)<1024 and (run or 0)==0, f"{a.minutes:.0f} min mixed priorities, {a.slots+4} closed-loop clients: {stats['n']} requests, {stats['err']} errors; per priority (count, mean TTFT s): {bp}; end used={u} running={run}")

    if a.mode=="pre2":
        # C9: abort the preempting request right after it preempts.
        res={}; ths=hold_slots(U,a.slots,0,3000,res); time.sleep(12); b=used()
        try:
            r=requests.post(U.rstrip("/")+"/generate",json={"text":"Preempt and leave. "*5,"priority":11,"stream":True,"sampling_params":{"temperature":0,"max_new_tokens":2000,"ignore_eos":True}},stream=True,timeout=120)
            t0=time.time(); first=None
            for _ in r.iter_lines():
                first=first or time.time()-t0
                if time.time()-t0>2: break
            r.close()
        except Exception as e: first=None
        for t in ths: t.join(timeout=900)
        ok=sum(1 for v in res.values() if v["status"]==200); time.sleep(10)
        say("C9-abort-preemptor", ok==a.slots and (used() or 0)<1024, f"priority-11 newcomer first token {first and round(first,2)}s (preempted a holder), aborted after 2 s; holders completed {ok}/{a.slots}; used tokens after: {used()}")
        # C10: abort the victims while requeued: start holders (streaming), newcomer preempts, then close all holder streams.
        hold=[]; stop=[]
        def h(i):
            try:
                r=requests.post(U.rstrip("/")+"/generate",json={"text":f"Holder {i}. "*10,"priority":0,"stream":True,"sampling_params":{"temperature":0,"max_new_tokens":3000,"ignore_eos":True}},stream=True,timeout=600)
                hold.append(r)
                for _ in r.iter_lines():
                    if stop: break
                r.close()
            except Exception: pass
        ths=[threading.Thread(target=h,args=(i,),daemon=True) for i in range(a.slots)]; [t.start() for t in ths]; time.sleep(12)
        nw=gen(U,"Newcomer that preempts.",priority=11,max_new_tokens=400,stream=True,timeout=600)
        stop.append(1); [t.join(60) for t in ths]; time.sleep(20)
        say("C10-abort-victims", nw["status"]==200 and (used() or 0)<1024 and (metric(D,"num_running_reqs",'tp_rank="0"') or 0)==0, f"holders' streams closed right after the preemption: newcomer HTTP {nw['status']} ttft={nw['ttft'] and round(nw['ttft'],2)}s; used={used()} running={metric(D,'num_running_reqs','tp_rank=\"0\"')}")
        # C15: repeated preemption: three newcomers (11, 12, 13) five seconds apart against fresh holders.
        res={}; ths=hold_slots(U,a.slots,0,3000,res); time.sleep(12); tt=[]
        for pr in (11,12,13):
            r=gen(U,f"Newcomer {pr}.",priority=pr,max_new_tokens=8,stream=True,timeout=600); tt.append((pr,r["status"],r["ttft"] and round(r["ttft"],2))); time.sleep(5)
        for t in ths: t.join(timeout=900)
        ok=sum(1 for v in res.values() if v["status"]==200); time.sleep(10)
        say("C15-repeated-preemption", all(s==200 and (t or 99)<3 for _,s,t in tt) and ok==a.slots, f"newcomers (priority, HTTP, ttft): {tt}; holders completed {ok}/{a.slots}; used after {used()}")

if __name__=="__main__":
    main()
