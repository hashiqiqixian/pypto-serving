# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TP1 entries composing library mHC and local MoE kernels, without EP transport."""


def load_kernels():
    """Create provider-owned entries after native_attention selects library shapes."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash.config import D, HC_MULT, HC_DIM, MIX_HC, T_DYN
    from models.deepseek_v4_1_flash.config import MOE_INTER, N_EXPERTS, TOPK, AUX_WIDTH, ROUTE_WIDTH
    from models.deepseek_v4_1_flash.mhc import mhc_mixes, mhc_pre, mhc_post
    from models.deepseek_v4_1_flash.moe import route, routed_up, routed_down, shared_up, shared_down, swiglu
    from models.deepseek_v4_1_flash.decode_swa import make_norm

    normalize = make_norm(D)

    @pl.jit
    def mixes(
        x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[MIX_HC], pl.FP32],
        pre: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
        post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
        comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
    ):
        x.bind_dynamic(0, T_DYN)
        pre.bind_dynamic(0, T_DYN)
        post.bind_dynamic(0, T_DYN)
        comb.bind_dynamic(0, T_DYN)
        mhc_mixes(x, function, scale, base, pre, post, comb)
        return pre, post, comb

    @pl.jit
    def pre(
        x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        coeff: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
        output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    ):
        x.bind_dynamic(0, T_DYN)
        coeff.bind_dynamic(0, T_DYN)
        output.bind_dynamic(0, T_DYN)
        mhc_pre(x, coeff, output)
        return output

    @pl.jit
    def post(
        sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
        residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        coeff: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
        comb: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
        output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    ):
        sublayer.bind_dynamic(0, T_DYN)
        residual.bind_dynamic(0, T_DYN)
        coeff.bind_dynamic(0, T_DYN)
        comb.bind_dynamic(0, T_DYN)
        output.bind_dynamic(0, T_DYN)
        mhc_post(sublayer, residual, coeff, comb, output)
        return output

    @pl.jit
    def norm(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        weight: pl.Tensor[[D], pl.BF16],
        output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        x.bind_dynamic(0, T_DYN)
        output.bind_dynamic(0, T_DYN)
        normalize(x, weight, output, num_tokens)
        return output

    @pl.jit
    def routing(
        x: pl.Tensor[[32, D], pl.BF16],
        gate: pl.Tensor[[N_EXPERTS, D], pl.FP32],
        bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        indices: pl.Out[pl.Tensor[[32, TOPK], pl.INT32]],
        weights: pl.Out[pl.Tensor[[32 * TOPK, AUX_WIDTH], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        routes = pl.create_tensor([32 * TOPK, ROUTE_WIDTH], dtype=pl.INT32)
        route(x, gate, bias, indices, weights, routes, num_tokens)
        return indices, weights

    @pl.jit
    def routed(
        x: pl.Tensor[[32, D], pl.BF16],
        w1: pl.Tensor[[MOE_INTER, D // 2], pl.UINT8],
        s1: pl.Tensor[[MOE_INTER, D // 32], pl.FP8E8M0],
        w2: pl.Tensor[[D, MOE_INTER // 2], pl.UINT8],
        s2: pl.Tensor[[D, MOE_INTER // 32], pl.FP8E8M0],
        w3: pl.Tensor[[MOE_INTER, D // 2], pl.UINT8],
        s3: pl.Tensor[[MOE_INTER, D // 32], pl.FP8E8M0],
        weights: pl.Tensor[[32, AUX_WIDTH], pl.FP32],
        output: pl.Out[pl.Tensor[[32, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        gate = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        up = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        hidden = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        routed_up(x, w1, s1, gate, num_tokens)
        routed_up(x, w3, s3, up, num_tokens)
        swiglu(gate, up, weights, hidden, num_tokens)
        routed_down(hidden, w2, s2, output, num_tokens)
        return output

    @pl.jit
    def shared(
        x: pl.Tensor[[32, D], pl.BF16],
        w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        s1: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
        s2: pl.Tensor[[MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        s3: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        weights: pl.Tensor[[32, AUX_WIDTH], pl.FP32],
        output: pl.Out[pl.Tensor[[32, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        gate = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        up = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        hidden = pl.create_tensor([32, MOE_INTER], dtype=pl.BF16)
        shared_up(x, w1, s1, gate, num_tokens)
        shared_up(x, w3, s3, up, num_tokens)
        swiglu(gate, up, weights, hidden, num_tokens)
        shared_down(hidden, w2, s2, output, num_tokens)
        return output

    return dict(mixes=mixes, pre=pre, post=post, norm=norm, routing=routing, routed=routed, shared=shared)
