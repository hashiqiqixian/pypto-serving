"""Prepare official graphs on CPU before allocating NPUs."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys

from pypto.runtime import device_runner
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.deepseek_v41.npu_runner import (
    selected_decode_topology, compile_decode_fwd, compile_decode_fwd_dspark,
    compile_prefill_fwd, SegmentTopology,
)
from run import CompilePool

device_runner.ThreadPoolExecutor = CompilePool
root = Path(os.environ['PYPTO_LIB_ROOT'])
chunk = int(sys.argv[1])
config = build_pypto_run_config(platform='a5', device_ids=[0, 1, 2, 3], pypto_build_dir=None)
config = replace(config, ring_heap=(1073741824,2147483648,4294967296,1073741824), ring_dep_pool=262144)
compiler = KernelCompiler(run_config=config)
topology = selected_decode_topology(root, dp_size=4)
decode = compile_decode_fwd(compiler, root, topology)[0]
programs, _ = compile_decode_fwd_dspark(compiler, root, topology)
prefill = compile_prefill_fwd(compiler, root, SegmentTopology.for_prefill(dp=4, max_tokens=chunk))[0]
head = compiler.compile('v41_generation_head',
    importlib.import_module('models.deepseek_v4_1_flash.prefill_fwd').l3_generation_head)
programs.update(backbone=decode, prefill=prefill, head=head)
manifest = {name: str(program.compiled.output_dir) for name, program in programs.items()}
(root.parent/f'compiled-{chunk}.json').write_text(json.dumps(manifest, indent=2))
for name, program in programs.items():
    for path in (program.compiled.output_dir/'next_levels').glob('*/kernel_config.py'):
        print('ASSEMBLE', name, path, flush=True)
        device_runner._compile_and_assemble(path.parent, 'a5')
print('COMPILE_PASS', chunk, flush=True)
