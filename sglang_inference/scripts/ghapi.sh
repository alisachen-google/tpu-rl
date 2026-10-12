#!/bin/bash
# ghapi.sh <path-under-/repos/sgl-project/sglang-torchtpu> [curl args...]  -- token from the git credential store
TOK=$(sed -nE 's#https://[^:]+:([^@]+)@github.com#\1#p' ~/.git-credentials | head -n1)
P=$1; shift; curl -s -H "Authorization: token $TOK" -H "Accept: application/vnd.github+json" "$@" "https://api.github.com/repos/sgl-project/sglang-torchtpu$P"
