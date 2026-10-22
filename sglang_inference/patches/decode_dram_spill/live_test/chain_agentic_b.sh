#!/usr/bin/env bash
D=$HOME/pd_disagg/spill
until grep -q 'chain-stress2' $D/logs/chain_stress2.log 2>/dev/null; do sleep 60; done
bash $D/phase_agentic_b.sh > $D/logs/phase_agentic_b.log 2>&1; echo "[chain-agentic-b] exit $?"
