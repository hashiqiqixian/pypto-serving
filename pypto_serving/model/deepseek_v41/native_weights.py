# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack existing checkpoint tiles for the TP1 native attention parameter ABI."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .tensor_store import DeepSeekV41TensorStore


_FP8_NAMES = {
    "wq_a": "wq_a.weight",
    "wq_b": "wq_b.weight",
    "wkv": "wkv.weight",
    "wo_b": "wo_b.weight",
    "index_wq_b": "indexer.wq_b.weight",
}
_DENSE_NAMES = {
    "q_norm_weight": ("q_norm.weight", False),
    "kv_norm_weight": ("kv_norm.weight", False),
    "attn_sink": ("attn_sink", False),
    "wo_a": ("wo_a.weight", False),
    "compressor_wkv": ("compressor.wkv.weight", True),
    "compressor_wgate": ("compressor.wgate.weight", True),
    "compressor_norm_weight": ("compressor.norm.weight", False),
    "index_wk": ("indexer.wk.weight", True),
    "index_norm_weight": ("indexer.k_norm.weight", False),
    "index_weights_proj": ("indexer.weights_proj.weight", True),
}


def pack_native_matrix(store, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one TP1 FP8 projection or retain an expert's exact FP4 bytes."""
    if store.world_size != 1:
        raise ValueError("native matrix packing requires TP1")
    fmt = store.matrix_format(name)
    if fmt == "fp4":
        return store.packed_fp4(name)
    if fmt != "fp8":
        raise ValueError("native matrix packing requires FP8 or FP4 weights")
    from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale

    rows, width = store.matrix_shape(name)
    if width % 64 or rows % 128:
        raise ValueError("native shared projection requires K divisible by 64 and N by 128")
    # Include both scale layouts and conservative tile/conversion scratch.
    store._budget(width * rows * 2 + width // 32 * rows * 2)
    payload = torch.empty((width, rows), dtype=torch.float8_e4m3fn)
    codes = torch.empty((width // 32, rows), dtype=torch.uint8)
    for row, column, values, factors in store.matrix_tiles(name):
        end = row + len(values)
        if values.shape[1] != 32 or factors is None:
            raise ValueError("native projection requires normalized K32 tiles")
        payload[column:column + 32, row:end].copy_(values.T.to(torch.float8_e4m3fn))
        codes[column // 32, row:end].copy_((factors.contiguous().view(torch.int32) >> 23).to(torch.uint8))
    return payload, pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)


def pack_native_attention_weights(
    store: DeepSeekV41TensorStore, layer_id: int, ratio: int, names: Collection[str],
) -> dict[str, torch.Tensor]:
    """Return only requested CPU-contiguous native parameters for one TP1 layer.

    FP8 payloads and UE8M0 scales are reconstructed from the store's exact
    normalized K32 tiles, without dequantizing a full projection. The caller
    owns placement and the resident layer budget, as for other store consumers.
    """
    if store.world_size != 1:
        raise ValueError("native attention weight packing requires a TP1 tensor store")
    if type(layer_id) is not int or layer_id < 0 or type(ratio) is not int or ratio not in (0, 1, 2):
        raise ValueError("native attention requires a nonnegative layer and ratio 0, 1 or 2")
    allowed = set(_DENSE_NAMES) | set(_FP8_NAMES) | {name + "_scale" for name in _FP8_NAMES}
    if isinstance(names, str) or any(not isinstance(name, str) or name not in allowed for name in names):
        raise ValueError("unknown native attention weight parameter")
    requested = set(names)
    prefix = f"layers.{layer_id}.attn."
    result = {}
    for name, suffix in _FP8_NAMES.items():
        want_weight, want_scale = name in requested, name + "_scale" in requested
        if not want_weight and not want_scale:
            continue
        # The native adapter has already loaded/configured the library. Keep
        # this dependency lazy so ordinary serving imports need no frontend.
        from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale

        source = prefix + suffix
        if store.matrix_format(source) != "fp8":
            raise ValueError(f"native attention requires an FP8 checkpoint projection: {source}")
        output_width, input_width = store.matrix_shape(source)
        if input_width % 64 or output_width % 16:
            raise ValueError(f"native MX_B_NN requires K divisible by 64 and N divisible by 16: {source}")
        payload = torch.empty(input_width, output_width, dtype=torch.float8_e4m3fn) if want_weight else None
        scales = torch.empty(input_width // 32, output_width, dtype=torch.uint8) if want_scale else None
        for row, column, values, factors in store.matrix_tiles(source):
            if values.shape[1] != 32 or factors is None:
                raise ValueError(f"native FP8 projection requires normalized K32 tiles and scales: {source}")
            end = row + values.shape[0]
            if want_weight:
                payload[column:column + 32, row:end].copy_(values.T.to(torch.float8_e4m3fn))
            if want_scale:
                # E8M0 byte 0 is 2^-127 (FP32 subnormal); its biased exponent
                # is also zero. All other finite codes equal the IEEE exponent.
                codes = (factors.contiguous().view(torch.int32) >> 23).to(torch.uint8)
                scales[column // 32, row:end].copy_(codes)
        if want_weight:
            result[name] = payload
        if want_scale:
            result[name + "_scale"] = pack_mx_b_scale(scales).view(torch.float8_e8m0fnu)
    for name, (suffix, transpose) in _DENSE_NAMES.items():
        if name not in requested:
            continue
        dtype = torch.float32 if name == "attn_sink" or (
            ratio == 2 and name in ("compressor_wkv", "compressor_wgate")
        ) else torch.bfloat16
        value = store.weight(prefix + suffix).to(device="cpu", dtype=dtype)
        result[name] = (value.T if transpose else value).contiguous()
    return result
