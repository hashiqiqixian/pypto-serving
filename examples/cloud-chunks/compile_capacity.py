"""Build the capacity-specific graphs while retaining the pinned prefill and head."""
from dataclasses import replace
import json
import os
from pathlib import Path

from pypto.runtime import device_runner
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.deepseek_v41.npu_runner import (
    compile_decode_fwd, compile_decode_fwd_dspark, selected_decode_topology,
)
from run import CompilePool

device_runner.ThreadPoolExecutor = CompilePool
lib = Path(os.environ["PYPTO_LIB_ROOT"])
cfg = build_pypto_run_config(platform="a5", device_ids=[0, 1, 2, 3], pypto_build_dir=None)
cfg = replace(cfg, ring_heap=(1073741824, 2147483648, 4294967296, 1073741824), ring_dep_pool=262144)
compiler = KernelCompiler(run_config=cfg)
topology = selected_decode_topology(lib, dp_size=4)
assert topology.local_capacity == 16
backbone, _, _ = compile_decode_fwd(compiler, lib, topology)
graphs, _ = compile_decode_fwd_dspark(compiler, lib, topology)
graphs["backbone"] = backbone
manifest = {}
for name, graph in graphs.items():
    folder = Path(graph.compiled.output_dir).resolve()
    for path in (folder / "next_levels").glob("*/kernel_config.py"):
        device_runner._compile_and_assemble(path.parent, "a5")
    manifest[name] = str(folder)
Path(os.environ["DECODE_MANIFEST"]).write_text(json.dumps(manifest, indent=2))
