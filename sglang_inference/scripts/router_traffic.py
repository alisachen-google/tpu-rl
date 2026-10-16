"""N non-streaming requests through the router. Usage: router_traffic.py <n> <concurrency> <max_new_tokens> [lb_url]
Prints the status codes, space separated."""

import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

n, c, tok = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lb = sys.argv[4] if len(sys.argv) > 4 else "http://10.128.15.217:8080"
text = json.loads(
    open("/home/alisachen_google_com/pd_disagg/prompts/gpt-oss/8192.jsonl").readline()
)["text"]


def one(i):
    body = json.dumps(
        {
            "text": f"r {i} {time.time():.3f}: " + text,
            "sampling_params": {"max_new_tokens": tok, "temperature": 0, "ignore_eos": True},
            "stream": False,
        }
    ).encode()
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(lb + "/generate", data=body, headers={"Content-Type": "application/json"}),
            timeout=400,
        )
        r.read()
        return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:
        return "ERR:" + type(e).__name__


with cf.ThreadPoolExecutor(c) as ex:
    print(" ".join(ex.map(one, range(n))))
