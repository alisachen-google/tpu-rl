#!/usr/bin/env bash
# Native TPU spill/recall round trip (single host, real registry + store processes).
export SGLANG_ROOT=$HOME/sglang-spill
source $HOME/pd_disagg/scripts/env_singlehost_v7x.sh >/dev/null
unset TPU_WORKER_ID TPU_WORKER_HOSTNAMES TPU_PROCESS_ADDRESSES TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY
for v in $(env | grep -o '^MEGASCALE[A-Z_]*'); do unset $v; done
export PATH=$HOME/venv/sglang/bin:$PATH TPU_LOG_DIR=$HOME/pd_disagg/spill/logs/tpu_logs WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29571; mkdir -p $TPU_LOG_DIR
cd $SGLANG_ROOT
# Same single-host slicebuilder/topology env the engine workers get (one rank, one chip).
exec timeout 600 $PYTHON -c '
import sys
from sglang_tpu.patches import _prepare_tpu_env
_prepare_tpu_env(world_size=1, resolve_single_host_topology=True)
import pytest
sys.exit(pytest.main(["-v", "-rs", "-p", "no:cacheprovider", "test/manual/tpu/test_raiden_dram_spill_recall_ondevice.py"]))
'
