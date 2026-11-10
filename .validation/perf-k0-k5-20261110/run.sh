set -e
b=/data/disk2/pyptouser/chenshenai/test0
source "$b/pypto-lib-v41-dspark-full-l3/.validation/env-serving.sh"
repo=$b/pypto-serving-pr302-cleanup-k5-exact-20261109
lib=$b/pypto-serving-pr302-lib-pin-20261109/pypto-lib
r=$b/pr302-k0-k5-perf-results-20261110
export PERF_SERVING_SHA=63654c91d36aa0d8d6c6d846da55f11c76584a14
export PERF_LIB_SHA=f9fad4eaab8cb9c12376b01cbeb6645d0622c78a
test "$(git -C "$repo" rev-parse HEAD)" = "$PERF_SERVING_SHA"
test "$(git -C "$lib" rev-parse HEAD)" = "$PERF_LIB_SHA"
test -f "$PYPTO_DSV41_MODEL_DIR/config.json"
test -f "$PYPTO_V41_PREPACKED_WEIGHTS"
export PYTHONPATH="$repo:$lib:$PYTHONPATH"
export PYPTO_LIB_ROOT="$lib"
export PYPTO_CACHE=1 PYPTO_CACHE_READONLY=0
export PYPTO_CACHE_DIR="$b/pr302-k5-cleanup-results-20261109/jit"
export PYPTO_WORKER_INIT_TIMEOUT=3600
export PERF_RESULTS="$r"
unset PYPTO_PROG_BUILD_DIR DSPARK_DIAG_DIR
cd "$repo"
printf 'SERVING=%s LIB=%s DEVICES=%s\n' "$PERF_SERVING_SHA" "$PERF_LIB_SHA" "$TASK_DEVICE"
python -c 'import sys, pypto, pypto_serving; print(sys.version, sys.executable, pypto.__file__, pypto_serving.__file__); assert pypto_serving.__file__.startswith(sys.argv[1] + "/"); assert hasattr(pypto, "configure_cache")' "$repo"
ptoas --version
printf 'PTOAS_ROOT=%s PTO_ISA_ROOT=%s ASCEND_HOME_PATH=%s\n' "$PTOAS_ROOT" "${PTO_ISA_ROOT:-}" "$ASCEND_HOME_PATH"
for mode in k0 k5; do
    python -u "$r/run.py" "$mode" > "$r/$mode-serving.log" 2>&1
done
printf 'PAIRED_BENCHMARK_COMPLETED\n'
