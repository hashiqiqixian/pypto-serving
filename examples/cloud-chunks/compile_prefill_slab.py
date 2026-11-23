"""Compile an isolated prefill artifact with a model-supported MoE slab size."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys

from pypto.runtime import device_runner
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.deepseek_v41.npu_runner import load_segment_modules, selected_decode_topology
from run import CompilePool

device_runner.ThreadPoolExecutor = CompilePool
lib = Path(os.environ["PYPTO_LIB_ROOT"])
slab = int(os.environ["PREFILL_SLAB"])
capacity = int(os.environ.get("PREFILL_CAPACITY", "512"))
load_segment_modules(lib, selected_decode_topology(lib, dp_size=4))
sys.argv = [sys.argv[0], "--ep", "4", "--dp", "4", "--capacity", str(capacity), "--moe-tokens", str(slab)]
mod = importlib.import_module("models.deepseek_v4_1_flash.prefill_fwd")
assert (mod.CAPACITY, mod.RANKS, mod.MOE_TOKENS) == (capacity, 4, slab)
assert sys.modules["prefill_moe"].MOE_TOKENS == slab
cfg = build_pypto_run_config(platform="a5", device_ids=[0, 1, 2, 3], pypto_build_dir=None)
cfg = replace(cfg, ring_heap=(1073741824, 2147483648, 4294967296, 1073741824), ring_dep_pool=262144)
compiled = KernelCompiler(run_config=cfg).compile("v41_prefill_fwd", mod.l3_prefill_fwd)
path = Path(compiled.compiled.output_dir).resolve()
for kernel in (path / "next_levels").glob("*/kernel_config.py"):
    device_runner._compile_and_assemble(kernel.parent, "a5")
manifest = json.loads(Path(os.environ["RECURRENT_MANIFEST"]).read_text())
manifest["prefill"] = str(path)
Path(os.environ["DECODE_MANIFEST"]).write_text(json.dumps(manifest, indent=2))
