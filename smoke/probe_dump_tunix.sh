#!/bin/bash
# Dump the image's tunix RL sources (for porting chunked-logps support). No TPU work.
T=$(python3 -c "import importlib.util as u;print(list(u.find_spec('tunix').submodule_search_locations)[0])")
ls $T/.. | grep -i tunix; cat $T/__init__.py 2>/dev/null | grep -i version
for f in rl/reward_manager.py rl/utils.py rl/inference/inference_worker.py; do
  echo "=====FILE $f ($(wc -l < $T/$f) lines)"; cat $T/$f; echo "=====END $f"
done
echo "PROBE: done"
sleep 900
