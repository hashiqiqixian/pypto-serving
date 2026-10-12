# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Byte-level checks for routed FP4 checkpoint layout conversion."""

import pytest
import torch

from pypto_serving.model.deepseek_v41.weight_spec import pack_fp4_tiles


def test_fp4_tiles_preserve_codes_across_tile_boundaries():
    n, k = 512, 768
    source = (torch.arange(n * k // 2, dtype=torch.int32) * 73 + 19).remainder(256)
    source = source.to(torch.int8).reshape(n, k // 2)
    packed = pack_fp4_tiles(source).reshape(n // 256, k // 256, 256, 128)
    raw = source.view(torch.uint8)

    for nb in range(n // 256):
        for kb in range(k // 256):
            for row in (0, 1, 126, 127):
                for col in (0, 1, 127, 254, 255):
                    first = raw[nb * 256 + row * 2, kb * 128 + col // 2]
                    second = raw[nb * 256 + row * 2 + 1, kb * 128 + col // 2]
                    shift = 4 * (col % 2)
                    expected = ((int(first) >> shift) & 15) | (((int(second) >> shift) & 15) << 4)
                    assert int(packed[nb, kb, col, row]) == expected

    reference = torch.empty_like(packed)
    for nb in range(n // 256):
        for kb in range(k // 256):
            block = raw[nb * 256:(nb + 1) * 256, kb * 128:(kb + 1) * 128]
            codes = torch.empty((256, 256), dtype=torch.uint8)
            codes[:, 0::2] = block & 15
            codes[:, 1::2] = block >> 4
            transposed = codes.T
            reference[nb, kb] = transposed[:, 0::2] | (transposed[:, 1::2] << 4)
    assert torch.equal(packed, reference)


def test_fp4_tiles_reject_partial_tile():
    with pytest.raises(ValueError, match="complete 256x256 tiles"):
        pack_fp4_tiles(torch.empty((256, 129), dtype=torch.uint8))
