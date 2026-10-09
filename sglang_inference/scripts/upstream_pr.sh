#!/usr/bin/env bash
# Push the fix branches to sgl-project/sglang-torchtpu and open PRs with the prepared descriptions.
# Needs a GitHub token with access to the repo in ~/.git-credentials (https://<user>:<token>@github.com).
#   upstream_pr.sh check            # verify the credential sees the repo; print issues 1142 1145 1161 to results/upstream_issues/
#   upstream_pr.sh push             # fetch origin main, rebase both branches on it, push them
#   upstream_pr.sh open 1142 1145   # open the two PRs (F-8 -> first number, F-11 -> second), bodies from results/prs/*.md
set -u
REPO=sgl-project/sglang-torchtpu; CLONE=$HOME/sglang-torchtpu; OUT=$HOME/pd_disagg/results/upstream_issues; PRS=$HOME/pd_disagg/results/prs
TOK=$(python3 -c "import re;s=open('$HOME/.git-credentials').read().strip().splitlines()[0];m=re.match(r'https://([^:]+):([^@]+)@github.com',s);print(m.group(2) if m else '')")
[ -n "$TOK" ] || { echo "no token in ~/.git-credentials"; exit 1; }
api(){ curl -s -m 30 -H "Authorization: token $TOK" -H "Accept: application/vnd.github+json" "$@"; }
B_F8=pd-chunked-abort-single-release; B_F11=pd-swa-chunk-cap-deadlock
case "${1:-check}" in
check)
  code=$(api -o /dev/null -w '%{http_code}' https://api.github.com/repos/$REPO); echo "repo access: HTTP $code (200 = ok)"; [ "$code" = 200 ] || exit 1
  mkdir -p $OUT; for n in 1142 1145 1161; do api https://api.github.com/repos/$REPO/issues/$n > $OUT/$n.json; api "https://api.github.com/repos/$REPO/issues/$n/comments?per_page=50" > $OUT/${n}_comments.json
    python3 - $OUT/$n.json <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); print(f"#{d.get('number')} {d.get('title')} [{d.get('state')}] by {d.get('user',{}).get('login')}\n{(d.get('body') or '')[:1500]}\n")
PY
  done;;
push)
  cd $CLONE && git fetch -q origin main && for b in $B_F8 $B_F11; do git -c user.name=alisachen-google -c user.email=alisa888cl@gmail.com rebase -q origin/main $b || { echo "rebase of $b needs manual resolution"; exit 1; }; done
  git push -u origin $B_F8 $B_F11 && git checkout -q main && echo "pushed $B_F8 $B_F11";;
open)
  F8=${2:?issue number for F-8}; F11=${3:?issue number for F-11}
  body(){ python3 -c "import sys,json; t=open(sys.argv[1]).read(); t=t.split('\n',1)[1]; print(json.dumps({'title':sys.argv[2],'head':sys.argv[3],'base':'main','body':t+'\n\nFixes #'+sys.argv[4]}))" "$1" "$2" "$3" "$4"; }
  body $PRS/PR_F8_chunked_abort_single_release.md "[TPU/PD] Release an aborted chunked prefill's kv row exactly once when the raiden sender must drain" $B_F8 $F8 | api -X POST -d @- https://api.github.com/repos/$REPO/pulls | python3 -c "import json,sys; d=json.load(sys.stdin); print('F-8 PR:', d.get('html_url') or d)"
  body $PRS/PR_F11_swa_chunk_cap_deadlock.md "[TPU/PD] Keep the chunked-prefill SWA continuation budget above zero (TPU chunk-cap alignment, early SWA release on raiden)" $B_F11 $F11 | api -X POST -d @- https://api.github.com/repos/$REPO/pulls | python3 -c "import json,sys; d=json.load(sys.stdin); print('F-11 PR:', d.get('html_url') or d)";;
*) echo "usage: $0 check|push|open <f8-issue> <f11-issue>"; exit 2;;
esac
