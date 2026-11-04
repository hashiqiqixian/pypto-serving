# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check legacy FP32/BF16 router sidecars without loading a model or NPU."""
import argparse
import hashlib
from pathlib import Path
import tempfile

from safetensors.torch import save_file
import torch

from pypto_serving.model.deepseek_v41 import weight_loader as loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-root', type=Path, required=True)
    args = parser.parse_args()
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    layout = loader.DecodeWeightLayout(1, 1, 1, 1, world=2)
    original_fields = loader._ALL
    fields = frozenset(('gate_weight', 'ffn_norm_weight'))
    loader._ALL = fields
    try:
        with tempfile.TemporaryDirectory(dir=args.tmp_root) as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text('{"weight_map":{}}')
            metadata = {'format': loader._FORMAT, 'source_fingerprint': loader.source_fingerprint(root, layout)}
            original = torch.linspace(-1.5, 1.5, 64).reshape(1, 2, 4, 8)
            norm = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)
            for index, dtype in enumerate((torch.float32, torch.bfloat16, torch.int32)):
                path = root / f'weights-{index}.safetensors'
                save_file({'gate_weight': original.to(dtype), 'ffn_norm_weight': norm}, path, metadata=metadata)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                try:
                    bank = loader.load_prepacked_bank(root, layout, fields, path=path)
                except ValueError as error:
                    if dtype != torch.int32 or 'router weights' not in str(error):
                        raise
                    continue
                if dtype == torch.int32:
                    raise AssertionError('Integer router weights were accepted')
                assert bank is not None and bank.prepacked
                assert bank.tensors['gate_weight'].dtype == torch.bfloat16
                assert torch.equal(bank.tensors['gate_weight'], original.to(torch.bfloat16))
                assert torch.equal(bank.tensors['ffn_norm_weight'], norm)
                assert hashlib.sha256(path.read_bytes()).hexdigest() == before
                if dtype == torch.float32:
                    assert bank.tensors['gate_weight'].is_shared()
            path = root / 'stale.safetensors'
            save_file({'gate_weight': original, 'ffn_norm_weight': norm}, path,
                      metadata={**metadata, 'source_fingerprint': 'stale'})
            assert loader.load_prepacked_bank(root, layout, fields, path=path) is None
    finally:
        loader._ALL = original_fields
    print('PASS: FP32 conversion, BF16 preservation, integer rejection, source fingerprint guard')


if __name__ == '__main__':
    main()
