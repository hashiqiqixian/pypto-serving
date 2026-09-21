# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 tensor arithmetic shared by CPU comparison and the Ascend execution path.

Quantized GEMMs accumulate normalized 32-element products in FP32, apply the
activation and weight scales, then accumulate blocks, as in the pinned reference.
FP4 is decoded explicitly. No packed integer dtype is treated as an FP8 tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Any

import torch
import torch.nn.functional as F


def _float_bits(value: torch.Tensor) -> torch.Tensor:
    return value.float().contiguous().view(torch.int32)


def _pow2(exponents: torch.Tensor) -> torch.Tensor:
    """Exact FP32 powers for integer exponents [-127, 127], including UE8M0 byte 0."""
    exponents = exponents.to(torch.int32)
    bits = (exponents + 127) << 23
    bits = torch.where(exponents == -127, 1 << 22, bits)
    return bits.contiguous().view(torch.float32)


def _sign_bit(value: torch.Tensor) -> torch.Tensor:
    # Ascend's comparison-based signbit loses -0. Read the IEEE sign directly.
    return ((_float_bits(value) >> 31) & 1).to(torch.uint8)


def decode_e4m3(raw: torch.Tensor) -> torch.Tensor:
    bits = raw.to(torch.int32)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    value = torch.where(exponent == 0, mantissa.float() * (2.0 ** -9),
                        (1 + mantissa.float() * .125) * _pow2(exponent - 7))
    value = torch.where((bits & 128) != 0, -value, value)
    return value.masked_fill((bits & 127) == 127, float("nan"))


def encode_e4m3(value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("E4M3 quantization requires finite inputs")
    magnitude = value.float().abs().clamp_max(448)
    exponent = ((_float_bits(magnitude.clamp_min(2.0 ** -6)) >> 23) & 255) - 127
    normal = (exponent + 6) * 8 + torch.round(magnitude * _pow2(3 - exponent))
    bits = torch.where(magnitude < 2.0 ** -6, torch.round(magnitude * 512), normal).to(torch.uint8)
    return bits | (_sign_bit(value) << 7)


def decode_e2m1(raw: torch.Tensor) -> torch.Tensor:
    bits = raw.to(torch.int32)
    codes = torch.stack((bits & 15, bits >> 4), dim=-1).flatten(-2)
    table = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), device=raw.device, dtype=torch.float32)
    values = table[codes & 7]
    return torch.where((codes & 8) != 0, -values, values)


def encode_e2m1(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] % 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("packed E2M1 requires an even last dimension and finite values")
    magnitude = value.float().abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    for index, midpoint in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        codes += ((magnitude > midpoint) | ((magnitude == midpoint) & bool(index % 2))).to(torch.uint8)
    codes |= _sign_bit(value) << 3
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def decode_ue8m0(raw: torch.Tensor) -> torch.Tensor:
    return _pow2(raw.to(torch.int32) - 127).masked_fill(raw == 255, float("nan"))


@dataclass(frozen=True)
class QuantizedRows:
    values: torch.Tensor
    scales: torch.Tensor
    fmt: str
    block_size: int

    def normalized(self) -> torch.Tensor:
        return decode_e4m3(self.values) if self.fmt.startswith("fp8_") else decode_e2m1(self.values)

    def scale_values(self) -> torch.Tensor:
        return decode_ue8m0(self.scales) if self.fmt.endswith("ue8m0") else decode_e4m3(self.scales)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        normalized = self.normalized()
        return (normalized.unflatten(-1, (-1, self.block_size)) *
                self.scale_values().unsqueeze(-1)).flatten(-2).to(dtype)


def quantize_rows(x: torch.Tensor, fmt: str, block_size: int) -> QuantizedRows:
    formats = {"fp8_e4m3_ue8m0", "fp4_e2m1_ue8m0", "fp4_e2m1_e4m3"}
    if fmt not in formats or block_size not in (16, 32) or x.shape[-1] % block_size:
        raise ValueError("unsupported V4.1 quantization format/block or incomplete block")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("quantization requires finite floating inputs")
    values = x.float().unflatten(-1, (-1, block_size))
    amax = values.abs().amax(-1)
    if fmt == "fp4_e2m1_e4m3":
        if bool((amax / 6 > 464).any()):
            raise ValueError("main KV scale overflows finite E4M3")
        scale_bits = encode_e4m3(amax.clamp_min(6 * 2.0 ** -9) / 6)
        scales = decode_e4m3(scale_bits)
        normalized = values / scales.unsqueeze(-1)
    else:
        maximum = 448 if fmt.startswith("fp8") else 6
        floor = 1e-4 if maximum == 448 else 6 * 2.0 ** -126
        # Match pinned fast_round_scale: multiply in FP32, then ceil-log2 by
        # exponent/mantissa bits. log2/pow approximations can cross rounding ties.
        required = _float_bits(amax.clamp_min(floor) * (1.0 / maximum))
        exponents = ((required >> 23) & 255) - 127 + ((required & 0x7fffff) != 0).to(torch.int32)
        if bool(((exponents < -127) | (exponents > 127)).any()):
            raise ValueError("activation scale is outside finite UE8M0 range")
        scale_bits = (exponents + 127).to(torch.uint8)
        # Ascend division flushes subnormal inputs. Multiplication by the exact
        # inverse power of two preserves them without changing midpoint rounding.
        normalized = values * _pow2(-exponents).unsqueeze(-1)
    normalized = normalized.flatten(-2)
    packed = encode_e4m3(normalized) if fmt.startswith("fp8") else encode_e2m1(normalized)
    return QuantizedRows(packed, scale_bits, fmt, block_size)


class TensorOps:
    """Bounded checkpoint operations and optional real PyPTO BF16 GEMM dispatch.

    ``weights`` provides shape/format, matrix tiles, small weights and selected
    embedding rows. All-reduces are explicit at model ownership boundaries.
    Non-BF16 arithmetic remains FP32; it is never silently downcast for the cube.
    """

    def __init__(self, weights: Any, *, device: str = "cpu", rank: int = 0,
                 world_size: int = 1, matmul_provider: Any = None, collective: Any = None):
        self.weights, self.device = weights, torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.dtype = torch.bfloat16
        self.matmul_provider, self.collective = matmul_provider, collective

    def weight(self, name: str) -> torch.Tensor:
        return self.weights.weight(name).to(self.device)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("matmul reduction dimensions disagree")
        if self.matmul_provider is None or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            return torch.matmul(a.float(), b.float())
        batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        left = a.expand(*batch, *a.shape[-2:]).reshape(-1, *a.shape[-2:])
        right = b.expand(*batch, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
        outputs = [self.matmul_provider.matmul(x.cpu().contiguous(), y.cpu().contiguous()).to(a.device)
                   for x, y in zip(left, right)]
        return torch.stack(outputs).reshape(*batch, a.shape[-2], b.shape[-1])

    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return value
        if self.collective is None:
            raise RuntimeError("multi-rank execution requires an initialized collective")
        return self.collective.all_reduce(value)

    def all_gather(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return value
        if self.collective is None:
            raise RuntimeError("multi-rank execution requires an initialized collective")
        return self.collective.all_gather(value)

    def quantize(self, x: torch.Tensor, fmt: str, block: int) -> torch.Tensor:
        packed = quantize_rows(x, fmt, block)
        value = packed.dequantize(x.dtype)
        value._pypto_quantized_payload = packed
        return value

    def linear(self, x: torch.Tensor, name: str, bias: bool = False) -> torch.Tensor:
        out_features, in_features = self.weights.matrix_shape(name)
        if x.shape[-1] != in_features:
            raise ValueError(f"{name}: input width {x.shape[-1]} != {in_features}")
        shape = x.shape[:-1]
        flat = x.reshape(-1, in_features)
        fmt = self.weights.matrix_format(name)
        quantized = fmt in ("fp8", "fp4")
        if quantized:
            activation = quantize_rows(flat, "fp8_e4m3_ue8m0", 32)
            normalized = activation.normalized().to(torch.bfloat16)
            activation_scales = activation.scale_values()
        scaled_matmul = getattr(self.matmul_provider, "block_scaled_matmul", None) if quantized else None
        if callable(scaled_matmul):
            # Transfer activations once for the projection, not once per K32.
            normalized = normalized.cpu().contiguous()
            activation_scales = activation_scales.cpu().contiguous()
        elif not quantized:
            normalized = flat.float() if fmt == "float32" else flat.to(torch.bfloat16)
        output = torch.zeros(flat.shape[0], out_features, device=x.device, dtype=x.dtype)
        bias_value = self.weight(name.removesuffix(".weight") + ".bias").float() if bias else None
        if bias_value is not None:
            output.copy_(bias_value)
        # Checkpoint tiles are ordered by output block, then reduction block.
        # Complete one FP32 accumulator before casting, without retaining an
        # FP32 copy of the entire Engram projection alongside the BF16 result.
        for out_start, tiles in groupby(self.weights.matrix_tiles(name), key=lambda tile: tile[0]):
            accumulated = None
            if callable(scaled_matmul):
                weight_blocks, scale_blocks = [], []
                for _, in_start, values, scales in tiles:
                    if in_start != 32 * len(weight_blocks) or values.shape[1] != 32:
                        raise ValueError("quantized matrix tiles must cover K in ordered 32-value blocks")
                    weight_blocks.append(values.T.cpu())
                    scale_blocks.append(scales.cpu())
                if len(weight_blocks) * 32 != in_features:
                    raise ValueError("quantized matrix tiles do not cover the projection input")
                accumulated = scaled_matmul(normalized, torch.cat(weight_blocks).contiguous(),
                                            activation_scales, torch.stack(scale_blocks).contiguous()).to(x.device)
            else:
                for _, in_start, values, scales in tiles:
                    values = values.to(x.device)
                    partial = self.matmul(normalized[:, in_start:in_start + values.shape[1]], values.T)
                    if quantized:
                        partial *= activation_scales[:, in_start // 32, None]
                        partial *= scales.to(x.device)[None, :]
                    if accumulated is None:
                        accumulated = torch.zeros_like(partial)
                    accumulated += partial
            stop = out_start + accumulated.shape[1]
            if bias_value is not None:
                accumulated += bias_value[out_start:stop]
            output[:, out_start:stop].copy_(accumulated)
        return output.reshape(*shape, out_features)

    def embedding(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        rows = self.weights.embedding(name, ids.detach().cpu()).to(self.device)
        return self.all_reduce(rows)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class ModelMath:
    """Hyper-Connections, routed/shared FFNs, and mandatory Engram computation."""

    def __init__(self, config: Any, ops: TensorOps):
        self.config, self.ops = config, ops
        self.text = config.text_config
        self.eps = float(self.text["rms_norm_eps"])
        self.hc = int(self.text["hc_mult"])

    def normalize(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return rms_norm(x, self.ops.weight(name), self.eps)

    def hc_mixes(self, x: torch.Tensor, prefix: str) -> tuple[torch.Tensor, ...]:
        flat = x.flatten(-2).float()
        mixes = self.ops.linear(flat, prefix + "_fn")
        mixes *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        scale, base = self.ops.weight(prefix + "_scale"), self.ops.weight(prefix + "_base")
        hc, eps = self.hc, float(self.text["hc_eps"])
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(int(self.text["hc_sinkhorn_iters"]) - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (x.float() * pre.unsqueeze(-1)).sum(-2).to(x.dtype)

    @staticmethod
    def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                comb: torch.Tensor) -> torch.Tensor:
        hc, dim = residual.shape[-2:]
        value = x.reshape(-1, dim)
        source = residual.reshape(-1, hc, dim)
        post = post.reshape(-1, hc)
        comb = comb.reshape(-1, hc, hc)
        output = torch.empty_like(source, dtype=x.dtype)
        # Bound the HC-by-HC broadcast product to 16 MiB. Token rows are
        # independent; retain the original multiply/reduce order within each.
        chunk_rows = max(1, (16 << 20) // (hc * hc * dim * 4))
        for start in range(0, len(value), chunk_rows):
            end = start + chunk_rows
            mixed = (comb[start:end].unsqueeze(-1) * source[start:end].float().unsqueeze(-2)).sum(-3)
            result = post[start:end].unsqueeze(-1) * value[start:end].float().unsqueeze(-2) + mixed
            output[start:end].copy_(result)
        return output.reshape(residual.shape)

    def expert(self, x: torch.Tensor, prefix: str, routing_weight: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.ops.linear(x, prefix + ".w1").float()
        up = self.ops.linear(x, prefix + ".w3").float()
        limit = float(self.text.get("swiglu_limit", 0))
        if limit > 0:
            gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
        hidden = F.silu(gate) * up
        if routing_weight is not None:
            hidden *= routing_weight
        return self.ops.linear(hidden.to(x.dtype), prefix + ".w2")

    def moe(self, x: torch.Tensor, prefix: str, image_mask: torch.Tensor | None = None,
            *, draft: bool = False) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = self.ops.linear(flat.float(), prefix + ".gate") / float(self.text.get("gate_temp", 1))
        scoring = self.text["scoring_func"]
        scores = scores.softmax(-1) if scoring == "softmax" else (
            scores.sigmoid() if scoring == "sigmoid" else F.softplus(scores).sqrt())
        bias = self.ops.weight(prefix + ".gate.bias").float()
        if image_mask is not None:
            bias = torch.where(image_mask.reshape(-1, 1), self.ops.weight(prefix + ".gate.bias_vl"), bias)
        count = int(self.text["dspark_n_routed_experts"] if draft else self.text["n_routed_experts"])
        topk = int(self.text["dspark_num_experts_per_tok"] if draft else self.text["num_experts_per_tok"])
        indices = (scores + bias).topk(topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.text["norm_topk_prob"] and topk > 1:
            weights /= weights.sum(-1, keepdim=True) + 1e-20
        weights *= float(self.text["routed_scaling_factor"])
        if count % self.ops.world_size:
            raise ValueError("routed expert count must be divisible by world_size")
        start = self.ops.rank * (count // self.ops.world_size)
        stop = start + count // self.ops.world_size
        output = torch.zeros_like(flat, dtype=torch.float32)
        for expert_id in range(start, stop):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                result = self.expert(flat[rows], f"{prefix}.experts.{expert_id}", weights[rows, slots, None])
                output.index_add_(0, rows, result.float())
        output = self.ops.all_reduce(output)
        output += self.expert(flat, prefix + ".shared_experts").float()
        return output.to(x.dtype).reshape(shape)

    def engram(self, x: torch.Tensor, hashes: torch.Tensor, prefix: str,
               token_mask: torch.Tensor | None = None) -> torch.Tensor:
        rows = self.ops.embedding(prefix + ".embed.weight", hashes)
        kv = self.ops.linear(rows.flatten(-2), prefix + ".wkv")
        dim = x.shape[-1]
        key, value = kv.split((self.hc * dim, dim), -1)
        key = key.float().unflatten(-1, (self.hc, dim))
        h = x.float()
        weight = self.ops.weight(prefix + ".q_weight").float() * self.ops.weight(prefix + ".k_weight").float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * key * weight).sum(-1) * rstd * dim ** -.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def block(self, x: torch.Tensor, pre_mix: torch.Tensor, prefix: str, attention: Any,
              *, image_mask: torch.Tensor | None = None, draft: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        attn_pre, post, comb = self.hc_mixes(x, prefix + ".hc_attn")
        collapsed = rms_norm(self.hc_pre(x, pre_mix), self.ops.weight(prefix + ".attn_norm.weight"), self.eps)
        x = self.hc_post(attention(collapsed), x, post, comb)
        ffn_pre, post, comb = self.hc_mixes(x, prefix + ".hc_ffn")
        collapsed = rms_norm(self.hc_pre(x, attn_pre), self.ops.weight(prefix + ".ffn_norm.weight"), self.eps)
        x = self.hc_post(self.moe(collapsed, prefix + ".ffn", image_mask, draft=draft), x, post, comb)
        return x, ffn_pre
