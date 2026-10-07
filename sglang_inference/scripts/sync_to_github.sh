#!/usr/bin/env bash
# Mirror the campaign (scripts, recipes, findings, results incl. gzipped server logs) into the user's GitHub repo
# alisachen-google/tpu-rl under sglang_inference/, commit and push. Large regenerable prompt sets are excluded
# (build_prompts.py / tiny_prompts.py recreate them); tiny.jsonl is kept.
#   sync_to_github.sh ["commit message"]
set -uo pipefail
SRC=$HOME/pd_disagg; REPO=$HOME/tpu-rl; DST=$REPO/sglang_inference; MSG=${1:-"sglang_inference: sync campaign results $(date -u +%FT%TZ)"}
mkdir -p $DST/results/logs
rsync -a --delete --exclude 'compile_cache' --exclude '__pycache__' --exclude 'prompts' $SRC/scripts $SRC/recipes $SRC/README.md $DST/
mkdir -p $DST/prompts/gpt-oss $DST/prompts/qwen3-vl; cp $SRC/prompts/gpt-oss/tiny.jsonl $DST/prompts/gpt-oss/ 2>/dev/null; cp $SRC/prompts/qwen3-vl/tiny.jsonl $DST/prompts/qwen3-vl/ 2>/dev/null
# results: everything except raw server logs (gzipped separately) and anything > 20 MB
rsync -a --delete --exclude 'compile_cache' --exclude '*.log' --exclude 'tpu_logs' --max-size=20m $SRC/results/ $DST/results/
# small text logs (phase/driver/run logs) go in as-is; server logs gzipped
find $SRC/results -name '*.log' -size -2M ! -name 'server*.log' ! -path '*/tpu_logs/*' | while read f; do rel=${f#$SRC/results/}; mkdir -p "$DST/results/$(dirname $rel)"; cp "$f" "$DST/results/$rel"; done
find $SRC/results/runs -name 'server*.log' | while read f; do rel=${f#$SRC/results/}; mkdir -p "$DST/results/logs/$(dirname $rel)"; gzip -c "$f" > "$DST/results/logs/$rel.gz"; done
cd $REPO && git add -A sglang_inference && git -c user.name="alisachen-google" -c user.email="alisa888cl@gmail.com" commit -q -m "$MSG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WzpS15R17maA4UqTPBUnv6" && git push -q origin HEAD:main && echo "pushed $(git rev-parse --short HEAD): $(git show --stat --oneline HEAD | tail -1)" || echo "nothing to commit or push failed"
du -sh $DST | cut -f1
