# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shape-specialized PyPTO leaves for the explicit BF16 execution path.

The bridge supplies BF16 inputs padded to M=16, N=64, K=64 multiples and an
FP32 output. Every output element is written. Quantization decoding and scaling
are upstream operations; this kernel never interprets FP4 bytes as another dtype.
"""

import pypto.language as pl


def make_bf16_matmul_kernel() -> object:
    """Create an independent JIT cache for one operator-provider lifetime.

    Uses the same M/N tiling and K accumulation as pypto-lib's gemm and
    multi_proj examples. Shapes come from JIT tensor metadata, without V4
    constants or dynamic dimensions sharing unrelated shape constraints.
    """

    @pl.jit
    def bf16_matmul(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        rows, inner = a.shape
        _, columns = b.shape
        for row in pl.parallel(0, rows, 16):
            for column in pl.parallel(0, columns, 64):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="bf16_matmul"):
                    acc = pl.matmul(a[row : row + 16, :64], b[:64, column : column + 64], out_dtype=pl.FP32)
                    for block in pl.range(1, inner // 64):
                        start = block * 64
                        acc = pl.matmul_acc(
                            acc,
                            a[row : row + 16, start : start + 64],
                            b[start : start + 64, column : column + 64],
                        )
                    c[row : row + 16, column : column + 64] = acc
        return c

    return bf16_matmul
