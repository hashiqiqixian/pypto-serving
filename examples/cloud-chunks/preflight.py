"""Inspect official entry-point compatibility without opening a device."""
import inspect
import os
import sys
from pathlib import Path

import torch
import pypto
from pypto_serving.model.deepseek_v41.npu_runner import (
    SegmentTopology, compile_decode_fwd, compile_decode_fwd_dspark,
    compile_prefill_fwd, selected_decode_topology,
)

print('ENV', sys.version, torch.__version__, pypto.__file__, flush=True)
root = Path(os.environ['PYPTO_LIB_ROOT'])
print('TOPOLOGY', selected_decode_topology(root, dp_size=4), flush=True)

class Inspector:
    def compile(self, name, entry, **kwargs):
        print('ENTRY', name, len(entry.param_names), inspect.signature(entry._func), flush=True)
        return None

compiler = Inspector()
topology = SegmentTopology.for_decode(tp=1, dp=4)
compile_decode_fwd(compiler, root, topology)
compile_decode_fwd_dspark(compiler, root, topology)
compile_prefill_fwd(compiler, root, SegmentTopology.for_prefill(dp=4, max_tokens=int(sys.argv[1])))
print('PREFLIGHT_PASS', sys.argv[1], flush=True)
