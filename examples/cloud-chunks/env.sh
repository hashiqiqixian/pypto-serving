#!/usr/bin/env bash
set -eu
root=/data/pyptouser/chenshenai/tasks/pypto-official-chunks-20261123
source /usr/local/Ascend/cann-9.3.0/set_env.sh
old=$root/runtime/data/disk2/pyptouser/chenshenai
export PYTHONHOME=$root/runtime/home/server-toolkits/miniconda3/envs/py310
export PATH=$PYTHONHOME/bin:$root/serving/examples/cloud-chunks/bin:$PATH
export LD_LIBRARY_PATH=$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$root/serving:$root/lib:$root/simpler-site:$old/test2/bootstrap-official-20261119/pinned-site-nb2:$old/test2/decode-capacity-20261116/ptoas-site:$old/v41-cache/venv-step2-py310/lib/python3.10/site-packages:$old/v41-cache/qproj-validation-deps-20261104:$old/v41-cache/python:$old/v41-cache/asgi-client-site
export PTOAS_ROOT=$root/serving/examples/cloud-chunks
export PTO_ISA_ROOT=$root/pto-isa
export PYPTO_LIB_ROOT=$root/lib PYPTO_DSV41_DP_SIZE=4 PYPTO_V41_DECODE_BATCH_PER_DP=12
export PYPTO_DSV41_MODEL_DIR=/data/model/DeepSeek-V4.1-Flash-1017
export PYPTO_V41_SHARED_SHARD_DIR=$PYPTO_DSV41_MODEL_DIR
export PYPTO_V41_PREPACKED_WEIGHTS=$root/prepacked-r4.safetensors
export PYPTO_CACHE=1 PYPTO_CACHE_DIR=$root/cache TMPDIR=$root/tmp
export XDG_CACHE_HOME=$root/xdg-cache TORCH_EXTENSIONS_DIR=$root/torch-extensions
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 PYPTO_CODEGEN_MAX_WORKERS=4 PYTHONUNBUFFERED=1
export PYPTO_WORKER_INIT_TIMEOUT=2400 PYPTO_RUNTIME_LOG=0
export GOLDEN_BENCH_WARMUP=0 GOLDEN_BENCH_ROUNDS=0
mkdir -p "$TMPDIR" "$PYPTO_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$root/results"
cd "$root"
