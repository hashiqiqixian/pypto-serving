# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in A5 TP1 backbone; Engram, embeddings and output projection retain the bridge.

The host schedules bounded local expert tiles. This is a correctness bring-up
path, with explicit activation transfers, not a fused or distributed executor.
"""

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch

from .numerics import ModelMath
from .native_weights import pack_native_matrix


class NativeModelMath(ModelMath):
    """Library arithmetic sharing the attention worker, with bounded resident weights."""

    def __init__(self, config, ops, attention, kernels, *, aux_width, weight_budget_bytes=256 << 20):
        super().__init__(config, ops)
        self.attention = attention
        self.kernels, self.aux_width = kernels, aux_width
        self.weight_budget_bytes = weight_budget_bytes
        self.weights = OrderedDict()
        self.weight_bytes = 0
        self.handles = {}
        self.calls = {}
        self.closed = False

    @staticmethod
    def _host(x, dtype):
        return x.detach().to(device="cpu", dtype=dtype).contiguous()

    def _run(self, mode, *args):
        if self.closed:
            raise RuntimeError("native backbone is closed")
        if mode not in self.handles:
            import pypto.language as pl
            config = replace(self.attention.run_config,
                             save_kernels_dir=str(Path(self.attention.artifact_dir.name) / ("backbone_" + mode)))
            scalars = {"num_tokens": pl.RUNTIME} if mode in ("norm", "routing", "routed", "shared") else {}
            self.handles[mode] = self.attention.worker.register(self.kernels[mode].compile(config=config, **scalars))
        self.handles[mode](*args)
        self.calls[mode] = self.calls.get(mode, 0) + 1

    def _resident(self, key, loader):
        if self.closed:
            raise RuntimeError("native backbone is closed")
        if key in self.weights:
            self.weights.move_to_end(key)
            return self.weights[key]
        values = loader()
        size = sum(x.numel() * x.element_size() for x in values)
        if size > self.weight_budget_bytes:
            raise ValueError("native backbone weight bundle exceeds resident budget")
        worker = self.attention.worker
        while self.weights and self.weight_bytes + size > self.weight_budget_bytes:
            _, old = self.weights.popitem(last=False)
            for value in old:
                worker.free_tensor(value)
                self.weight_bytes -= value.nbytes
        tensors = []
        try:
            for value in values:
                tensors.append(worker.alloc_tensor(value.shape, value.dtype, init=value.contiguous()))
        except BaseException:
            for tensor in tensors:
                worker.free_tensor(tensor)
            raise
        self.weights[key] = tuple(tensors)
        self.weight_bytes += size
        return self.weights[key]

    def _dense(self, names, dtype=torch.float32):
        return self._resident((tuple(names), dtype), lambda: [self._host(self.ops.weight(n), dtype) for n in names])

    def hc_mixes(self, x, prefix):
        host = self._host(x, torch.float32)
        pre = torch.empty((len(x), self.hc), dtype=torch.float32)
        post, comb = torch.empty_like(pre), torch.empty((len(x), self.hc, self.hc), dtype=torch.float32)
        weights = self._dense([prefix + suffix for suffix in ("_fn", "_scale", "_base")])
        self._run("mixes", host, *weights, pre, post, comb)
        return tuple(value.to(x.device) for value in (pre, post, comb))

    def hc_pre(self, x, pre):
        output = torch.empty((len(x), x.shape[-1]), dtype=torch.bfloat16)
        self._run("pre", self._host(x, torch.float32), self._host(pre, torch.float32), output)
        return output.to(x.device)

    def hc_post(self, x, residual, post, comb):
        output = torch.empty(residual.shape, dtype=torch.float32)
        self._run("post", self._host(x, torch.bfloat16), self._host(residual, torch.float32),
                  self._host(post, torch.float32), self._host(comb, torch.float32), output)
        return output.to(device=residual.device, dtype=residual.dtype)

    def normalize(self, x, name):
        output = torch.empty(x.shape, dtype=torch.bfloat16)
        self._run("norm", self._host(x, torch.bfloat16), *self._dense([name], torch.bfloat16), output, len(x))
        return output.to(x.device)

    def _expert(self, x, prefix, route_weight, *, shared):
        """The routed library entry computes at most 32 rows; never truncate a prefill."""
        if not 0 < len(x) <= 32:
            raise ValueError("native expert tile must contain 1..32 rows")
        def load():
            values = []
            for name in ("w1", "w2", "w3"):
                source = prefix + "." + name
                expected = "fp8" if shared else "fp4"
                if self.ops.weights.matrix_format(source) != expected:
                    raise ValueError(f"native {prefix} requires {expected} weights")
                values.extend(pack_native_matrix(self.ops.weights, source))
            return values
        weights = self._resident(("expert", prefix), load)
        padded = torch.zeros((32, x.shape[-1]), dtype=torch.bfloat16)
        padded[:len(x)].copy_(x)
        coefficients = torch.zeros((32, self.aux_width), dtype=torch.float32)
        coefficients[:len(x), 0] = route_weight
        output = torch.empty_like(padded)
        self._run("shared" if shared else "routed", padded, *weights, coefficients, output, len(x))
        return output[:len(x)].clone()

    def moe(self, x, prefix, image_mask=None, *, draft=False):
        if image_mask is not None or draft:
            raise ValueError("native TP1 backbone supports text target inference only")
        flat = self._host(x.reshape(-1, x.shape[-1]), torch.bfloat16)
        topk = int(self.text["num_experts_per_tok"])
        count = int(self.text["n_routed_experts"])
        output = torch.empty_like(flat)
        for start in range(0, len(flat), 32):
            rows = flat[start:start + 32]
            padded = torch.zeros((32, flat.shape[-1]), dtype=torch.bfloat16)
            padded[:len(rows)].copy_(rows)
            indices = torch.empty((32, topk), dtype=torch.int32)
            weights = torch.empty((32 * topk, self.aux_width), dtype=torch.float32)
            gate = self._dense([prefix + ".gate.weight", prefix + ".gate.bias"])
            self._run("routing", padded, *gate, indices, weights, len(rows))
            indices = indices[:len(rows)].long()
            weights = weights[:len(rows) * topk, 0].reshape(len(rows), topk)
            if bool(((indices < 0) | (indices >= count)).any()) or not bool(torch.isfinite(weights).all()):
                raise ValueError("native router produced invalid expert IDs or weights")
            ordered = indices.sort(-1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                raise ValueError("native router selected the same expert twice for one token")
            # Preserve per-token top-k order when combining, independent of expert dispatch order.
            routed = torch.empty((len(rows), topk, flat.shape[-1]), dtype=torch.bfloat16)
            for expert in indices.unique(sorted=True).tolist():
                token, slot = torch.where(indices == expert)
                result = self._expert(rows[token], f"{prefix}.experts.{expert}", weights[token, slot], shared=False)
                routed[token, slot] = result
            combined = routed.float().sum(1)
            combined += self._expert(rows, prefix + ".shared_experts", torch.ones(len(rows)), shared=True).float()
            output[start:start + len(rows)].copy_(combined)
        return output.reshape(x.shape).to(x.device)

    def block(self, x, pre_mix, prefix, attention, *, image_mask=None, draft=False):
        if image_mask is not None or draft:
            raise ValueError("native TP1 backbone supports text target inference only")
        attn_pre, post, comb = self.hc_mixes(x, prefix + ".hc_attn")
        collapsed = self.normalize(self.hc_pre(x, pre_mix), prefix + ".attn_norm.weight")
        x = self.hc_post(attention(collapsed), x, post, comb)
        ffn_pre, post, comb = self.hc_mixes(x, prefix + ".hc_ffn")
        collapsed = self.normalize(self.hc_pre(x, attn_pre), prefix + ".ffn_norm.weight")
        x = self.hc_post(self.moe(collapsed, prefix + ".ffn"), x, post, comb)
        return x, ffn_pre

    def diagnostics(self):
        return {"provider": "pypto-lib-a5-tp1-backbone", "weight_bytes": self.weight_bytes,
                "kernel_calls": dict(self.calls), "engram": "reference", "expert_tile_rows": 32,
                "combine": "host-fp32", "embedding_and_head_projection": "reference"}

    def close(self):
        if self.closed:
            return
        for values in self.weights.values():
            for value in values:
                self.attention.worker.free_tensor(value)
        self.weights.clear()
        self.handles.clear()
        self.kernels.clear()
        self.weight_bytes = 0
        self.closed = True


def create_backend(*, config, runtime, cache_layouts, weight_loader, device_ids, platform="a5",
                   pypto_build_dir=None, use_compile_cache=False):
    """CLI factory for the first, text-only native-backbone bring-up stage."""
    from .native_attention import _load_library, create_backend as create_attention_backend
    if platform != "a5" or len(device_ids) != 1:
        raise ValueError("native V4.1 backbone requires A5 TP1")
    if runtime.num_speculative_tokens:
        raise ValueError("native V4.1 backbone does not yet support DSpark")
    library = _load_library()
    constants = library.load_kernel_configuration(tp_size=1, ep_size=2)
    reference = constants.FLASH
    for field in ("hc_mult", "hc_sinkhorn_iters", "hc_eps", "moe_intermediate_size", "n_routed_experts",
                  "n_shared_experts",
                  "num_experts_per_tok", "swiglu_limit", "routed_scaling_factor"):
        if config.text_config[field] != getattr(reference, field):
            raise ValueError(f"native V4.1 backbone requires Flash {field}")
    if (config.text_config["scoring_func"] != "sqrtsoftplus" or not config.text_config["norm_topk_prob"]
            or config.text_config.get("gate_temp", 1) != reference.gate_temperature):
        raise ValueError("native V4.1 backbone requires the Flash routing formula")
    from .native_backbone_kernels import load_kernels
    kernels = load_kernels()
    backend = create_attention_backend(config=config, runtime=runtime, cache_layouts=cache_layouts,
                                       weight_loader=weight_loader, device_ids=device_ids, platform=platform,
                                       pypto_build_dir=pypto_build_dir, use_compile_cache=use_compile_cache)
    try:
        backend.model_provider = NativeModelMath(config, backend.ops, backend.attention_provider, kernels,
                                                aux_width=constants.AUX_WIDTH)
        backend.math = backend.model_provider
        return backend
    except BaseException:
        backend.close()
        raise
