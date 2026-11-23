"""Compile only the recurrent program used by the paired decode benchmark."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path

import pypto.language as pl
from pypto.runtime import device_runner
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.deepseek_v41.npu_runner import load_segment_modules, selected_decode_topology
from run import CompilePool

device_runner.ThreadPoolExecutor = CompilePool
root = Path(os.environ["PYPTO_LIB_ROOT"])
load_segment_modules(root, selected_decode_topology(root, dp_size=4))
module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd_dspark")
assert Path(module.__file__).resolve().is_relative_to(root.resolve())
cfg = build_pypto_run_config(platform="a5", device_ids=[0, 1, 2, 3], pypto_build_dir=None)
cfg = replace(cfg, ring_heap=(1073741824, 2147483648, 4294967296, 1073741824), ring_dep_pool=262144)
entry = module.l3_decode_fwd_dspark
kwargs = {key: pl.RUNTIME for key in ("moe_epoch", "logits_epoch") if key in entry.param_names}
program = KernelCompiler(run_config=cfg).compile("v41_dspark_decode", entry, **kwargs)
output = Path(program.compiled.output_dir).resolve()
for path in (output / "next_levels").glob("*/kernel_config.py"):
    device_runner._compile_and_assemble(path.parent, "a5")
Path(os.environ["DECODE_MANIFEST"]).write_text(json.dumps({"decode": str(output)}))
