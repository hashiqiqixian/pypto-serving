"""Prepare the official V4.1 sidecar once on CPU, without device allocation."""
import importlib
import json
import logging
import os
from pathlib import Path
import time

import torch
from safetensors.torch import save_file
from pypto_serving.model.deepseek_v41.npu_runner import selected_decode_topology, load_segment_modules
from pypto_serving.model.deepseek_v41.weight_loader import (
    DecodeWeightBank, DecodeWeightLayout, iter_decode_weight_slices,
    source_fingerprint, load_prepacked_bank, _FORMAT,
)

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(8)
root = Path(os.environ['PYPTO_LIB_ROOT'])
model = Path(os.environ['PYPTO_DSV41_MODEL_DIR'])
output = root.parent/'prepacked-r4.safetensors'
temporary = output.with_suffix('.partial')
assert not output.exists() and not temporary.exists()
topology = selected_decode_topology(root, dp_size=4)
load_segment_modules(root, topology)
abi = importlib.import_module('models.deepseek_v4_1_flash.decode_fwd')
layout = DecodeWeightLayout.from_lib(abi, tp_size=1)
names = tuple(abi.l3_decode_fwd.param_names)
bank = DecodeWeightBank(layout)
started = time.perf_counter()
for item in iter_decode_weight_slices(model, topology, parallel_experts=True, bank=bank):
    bank.append(item)
    if item.name == 'hc_attn_fn':
        print('PREPACK_LAYER', item.slot+1, 'elapsed_s', time.perf_counter()-started, flush=True)
bank.finish(names)
float8 = {name: str(tensor.dtype).removeprefix('torch.') for name, tensor in bank.tensors.items()
          if str(tensor.dtype).startswith('torch.float8_')}
stored = {name: tensor.view(torch.uint8) if name in float8 else tensor
          for name, tensor in bank.tensors.items()}
print('PREPACK_WRITE_BYTES', bank.allocated_bytes, flush=True)
save_file(stored, str(temporary), metadata={
    'format': _FORMAT, 'source_fingerprint': source_fingerprint(model, layout),
    'float8_fields': json.dumps(float8),
})
check = load_prepacked_bank(model, layout, names, path=temporary)
assert check is not None
for name, tensor in bank.tensors.items():
    restored = check.tensors[name]
    assert tensor.shape == restored.shape and tensor.dtype == restored.dtype, name
    a, b = tensor.view(torch.uint8).flatten(), restored.view(torch.uint8).flatten()
    for start in (0, max(0, a.numel()//2-512), max(0, a.numel()-1024)):
        assert torch.equal(a[start:start+1024], b[start:start+1024]), name
assert not output.exists()
temporary.rename(output)
print('PREPACK_PASS', output, 'elapsed_s', time.perf_counter()-started, flush=True)
