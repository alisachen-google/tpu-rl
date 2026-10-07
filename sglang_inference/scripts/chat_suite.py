#!/usr/bin/env python3
"""chat_suite.py - OpenAI chat-path equivalence between a golden and a PD SGLang endpoint.

  python3 chat_suite.py --model openai/gpt-oss-120b --kind gptoss --a http://golden:port \
      [--b http://pd:port] [--cases plain_chat,tool_call] [--timeout 300] [--report chat_report.json]

Cases: plain_chat, reasoning_low|medium|high (gptoss), multi_turn_3, tool_call (gptoss),
stop_string_chat, stream_with_usage, long_system_prompt, empty_user_message, oversized_max_tokens,
bad_image_payload (qwen).  Per case and endpoint we record status, finish_reason, content (first
200 chars in the report), reasoning length, usage, tool_calls, elapsed, error.  A vs B must have
identical status, finish_reason and content (tool_call: name + parsed arguments; reasoning cases:
finish_reason only, plus content+reasoning non-empty - an empty content with finish_reason
"length" is not a failure).  Self checks: stream == non-stream with [DONE] and usage, stop string
honoured, oversized/bad-image requests return a clean 4xx (never hang), server still answers after
the bad image.  Exit 1 on any divergence, failed self check or hang.
"""
import argparse
import json
import sys
import time

import requests

Q_PLAIN = "Explain in three sentences why the sky is blue."
Q_REASON = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
FILLER = ("policy archive ledger harbor meadow quartz signal compass ration vessel chapter margin "
          "lantern thistle glacier orchard cadence tunnel summit beacon").split()


def read_chat_sse(resp):
    out = {"content": "", "reasoning": "", "finish_reason": None, "usage": None, "done": False, "tc": {}}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            out["done"] = True
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        for c in chunk.get("choices") or []:
            d = c.get("delta") or {}
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = out["tc"].setdefault(tc.get("index", 0), {"name": None, "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["arguments"] += fn.get("arguments") or ""
            if c.get("finish_reason"):
                out["finish_reason"] = c["finish_reason"]
    out["tool_calls"] = [out["tc"][i] for i in sorted(out["tc"])]
    return out


def norm_tool_calls(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or tc
        argv = fn.get("arguments")
        parsed = None
        if isinstance(argv, str):
            try:
                parsed = json.loads(argv)
            except ValueError:
                parsed = None
        elif isinstance(argv, dict):
            parsed, argv = argv, json.dumps(argv)
        out.append({"name": fn.get("name"), "arguments": argv, "parsed": parsed})
    return out


def chat(base, args, messages, max_tokens, stream=False, **extra):
    res = {"status": None, "finish_reason": None, "content": "", "reasoning_len": 0, "usage": None,
           "tool_calls": None, "elapsed": None, "error": None, "hang": False, "done_seen": None}
    body = {"model": args.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    body.update(extra)
    if stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.time()
    try:
        r = requests.post(base.rstrip("/") + "/v1/chat/completions", json=body, timeout=args.timeout, stream=stream)
        res["status"] = r.status_code
        if r.status_code != 200:
            res["error"] = r.text[:300]
        elif stream:
            s = read_chat_sse(r)
            res.update({"content": s["content"], "reasoning_len": len(s["reasoning"]), "usage": s["usage"],
                        "finish_reason": s["finish_reason"], "done_seen": s["done"],
                        "tool_calls": norm_tool_calls(s["tool_calls"]) or None})
        else:
            j = r.json()
            ch = (j.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            res.update({"content": msg.get("content") or "", "reasoning_len": len(msg.get("reasoning_content") or ""),
                        "usage": j.get("usage"), "finish_reason": ch.get("finish_reason"),
                        "tool_calls": norm_tool_calls(msg.get("tool_calls")) or None})
    except requests.Timeout:
        res["hang"], res["error"] = True, f"HANG? no response within {args.timeout}s"
    except requests.RequestException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        res["error"] = f"bad response: {type(e).__name__}: {str(e)[:300]}"
    res["elapsed"] = round(time.time() - t0, 3)
    return res


def user(q):
    return [{"role": "user", "content": q}]


# ------------------------------------------------------------------ cases: fn(base, args) -> res
def c_plain(base, args):
    return chat(base, args, user(Q_PLAIN), 256)


def c_reasoning(effort):
    def run(base, args):
        r = chat(base, args, user(Q_REASON), 1024, reasoning_effort=effort)
        if r["status"] == 200 and not (r.get("content") or "") and r["reasoning_len"] == 0:
            r["self_fail"] = ["content and reasoning both empty"]
        return r
    return run


def c_multi_turn(base, args):
    msgs, turns = [], []
    for q in ("Pick one fruit and answer with just its name.",
              "Name one color that matches that fruit, one word only.",
              "Combine the fruit and the color into a short product name."):
        msgs.append({"role": "user", "content": q})
        r = chat(base, args, msgs, 64)
        turns.append(r.get("content") or "")
        if r["status"] != 200 or r["hang"]:
            break
        msgs.append({"role": "assistant", "content": r.get("content") or ""})
    r["turns"] = turns
    return r


def c_tool_call(base, args):
    r = chat(base, args, user("What is the weather in Paris right now?"), 512, tools=TOOLS, tool_choice="auto")
    if r["status"] == 200:
        tcs = r["tool_calls"] or []
        if not tcs:
            r["self_fail"] = [f"no tool_calls (finish_reason={r['finish_reason']})"]
        elif tcs[0]["name"] != "get_weather" or not isinstance(tcs[0]["parsed"], dict) or "city" not in tcs[0]["parsed"]:
            r["self_fail"] = [f"bad tool call: {json.dumps(tcs[0])[:120]}"]
    return r


def c_stop(base, args):
    r = chat(base, args, user("List five animals, one per line."), 128, stop=["\n"])
    if r["status"] == 200 and "\n" in (r.get("content") or ""):
        r["self_fail"] = ["stop string not honoured"]
    return r


def c_stream_usage(base, args):
    ns = chat(base, args, user(Q_PLAIN), 256)
    r = chat(base, args, user(Q_PLAIN), 256, stream=True)
    r["nonstream_content"] = ns["content"]
    fails = []
    if r["status"] == 200:
        fails += [] if r["done_seen"] else ["no [DONE]"]
        fails += [] if r["usage"] else ["no usage chunk"]
        fails += [] if (r.get("content") or "") == (ns.get("content") or "") else ["stream != nonstream"]
    r["hang"] = r["hang"] or ns["hang"]
    if fails:
        r["self_fail"] = fails
    return r


def c_long_system(base, args):
    words = [FILLER[(i * 7 + i // 13) % len(FILLER)] for i in range(args.system_words)]
    system = "You are a careful assistant. Reference material follows.\n" + " ".join(words)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "In one sentence, what does the reference material look like?"}]
    return chat(base, args, msgs, 128)


def c_empty_user(base, args):
    return chat(base, args, user(""), 64)


def c_oversized(base, args):
    r = chat(base, args, user(Q_PLAIN), 10_000_000)
    if r["status"] == 200:
        r["note"] = "no 4xx for max_tokens=10M (server clamped?)"
    return r


def c_bad_image(base, args):
    content = [{"type": "text", "text": "Describe this image."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64-at-all!!!"}}]
    r = chat(base, args, [{"role": "user", "content": content}], 64)
    fails = [] if r["status"] and 400 <= r["status"] < 500 else [f"expected 4xx, got {r['status']}"]
    f = chat(base, args, user(Q_PLAIN), 64)
    r["followup_status"], r["hang"] = f["status"], r["hang"] or f["hang"]
    if f["status"] != 200:
        fails.append(f"follow-up after bad image failed: {f['status']} {(f['error'] or '')[:80]}")
    if fails:
        r["self_fail"] = fails
    return r


CASES = [  # name, fn, kinds (None = all), compare mode, error_ok
    ("plain_chat", c_plain, None, "content", False),
    ("reasoning_low", c_reasoning("low"), {"gptoss"}, "reasoning", False),
    ("reasoning_medium", c_reasoning("medium"), {"gptoss"}, "reasoning", False),
    ("reasoning_high", c_reasoning("high"), {"gptoss"}, "reasoning", False),
    ("multi_turn_3", c_multi_turn, None, "content", False),
    ("tool_call", c_tool_call, {"gptoss"}, "tool", False),
    ("stop_string_chat", c_stop, None, "content", False),
    ("stream_with_usage", c_stream_usage, None, "content", False),
    ("long_system_prompt", c_long_system, None, "content", False),
    ("empty_user_message", c_empty_user, None, "content", True),
    ("oversized_max_tokens", c_oversized, None, "status", True),
    ("bad_image_payload", c_bad_image, {"qwen"}, "status", True),
]


def compare(mode, a, b):
    diffs = []
    if a["status"] != b["status"]:
        diffs.append(f"status {a['status']} vs {b['status']}")
    if a["finish_reason"] != b["finish_reason"]:
        diffs.append(f"finish_reason {a['finish_reason']} vs {b['finish_reason']}")
    if mode == "content" and a["content"] != b["content"]:
        n = min(len(a["content"]), len(b["content"]))
        i = next((k for k in range(n) if a["content"][k] != b["content"][k]), n)
        diffs.append(f"content differs at char {i}")
    if mode == "tool":
        ta = [(t["name"], t["parsed"]) for t in a["tool_calls"] or []]
        tb = [(t["name"], t["parsed"]) for t in b["tool_calls"] or []]
        if ta != tb:
            diffs.append("tool_calls differ")
    return diffs


def cell(r):
    if r is None:
        return "-"
    if r.get("hang"):
        return "HANG?"
    return f"{r.get('status')}/{r.get('finish_reason')}/{len(r.get('content') or '')}c/{r.get('reasoning_len')}r"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name sent in the request body")
    ap.add_argument("--a", required=True, help="golden endpoint http://host:port")
    ap.add_argument("--b", default=None, help="PD endpoint (omit to only report A)")
    ap.add_argument("--kind", choices=["gptoss", "qwen"], required=True)
    ap.add_argument("--cases", default=None, help="comma list of case names (default: all for --kind)")
    ap.add_argument("--system-words", type=int, default=4500, help="filler words for long_system_prompt (~6k tokens)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--report", default="chat_report.json")
    args = ap.parse_args()

    cases = [c for c in CASES if c[2] is None or args.kind in c[2]]
    if args.cases:
        want = set(args.cases.split(","))
        unknown = want - {c[0] for c in CASES}
        if unknown:
            sys.exit(f"error: unknown case(s): {sorted(unknown)}")
        cases = [c for c in cases if c[0] in want]
    endpoints = [("a", args.a)] + ([("b", args.b)] if args.b else [])
    print(f"model={args.model} kind={args.kind} {len(cases)} cases, endpoints: " + ", ".join(f"{k}={u}" for k, u in endpoints))
    hdr = f"{'case':<22}{'A st/fin/len':<30}{'B st/fin/len':<30}{'same':<6}{'verdict':<8}notes"
    print(hdr + "\n" + "-" * len(hdr))
    rows, n_fail = [], 0
    for name, fn, _, mode, error_ok in cases:
        row = {"name": name, "compare_mode": mode, "a": None, "b": None, "same": None, "notes": [], "verdict": "PASS"}
        for key, url in endpoints:
            r = fn(url, args)
            row[key] = r
            if r["hang"]:
                row["notes"].append(f"{key}: HANG?")
            elif r["status"] != 200 and not (error_ok and r["status"] and 400 <= r["status"] < 500):
                row["notes"].append(f"{key}: http {r['status']} {(r['error'] or '')[:80]}")
            if mode == "reasoning" and r["status"] == 200 and not (r.get("content") or "") and r["finish_reason"] == "length":
                r.pop("self_fail", None)  # truncated reasoning, not a failure by itself
            row["notes"] += [f"{key}: {s}" for s in r.get("self_fail", [])]
            if r.get("note"):
                row["notes"].append(f"info {key}: {r['note']}")
            r["content_head"] = (r.get("content") or "")[:200]
        if row["b"] is not None:
            diffs = compare(mode, row["a"], row["b"])
            row["same"] = not diffs
            row["notes"] += [f"a!=b: {d}" for d in diffs]
        fail = any(not n.startswith("info") for n in row["notes"])
        row["verdict"] = "FAIL" if fail else "PASS"
        n_fail += fail
        for key in ("a", "b"):  # keep report compact: full content only via content_head
            if row[key]:
                row[key] = {k: v for k, v in row[key].items() if k not in ("content", "nonstream_content")}
        rows.append(row)
        same = "-" if row["same"] is None else ("y" if row["same"] else "N")
        print(f"{name:<22}{cell(row['a']):<30}{cell(row['b']):<30}{same:<6}{row['verdict']:<8}{'; '.join(row['notes'])}", flush=True)
    report = {"model": args.model, "kind": args.kind, "a": args.a, "b": args.b, "timeout": args.timeout,
              "cases": len(rows), "failed": n_fail, "rows": rows}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(rows)} cases, {n_fail} failed -> {args.report}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
