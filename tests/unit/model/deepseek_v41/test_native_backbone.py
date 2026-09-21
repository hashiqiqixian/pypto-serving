# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host composition/recovery checks; CPU dispatch does not validate native kernel numerics."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
_spec = importlib.util.spec_from_file_location("_backbone_fixtures", Path(__file__).with_name("test_backend.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
modules, checkpoint, factory = _fixtures.modules, _fixtures.checkpoint, _fixtures.factory


def install_cpu_dispatch(harness, modules, monkeypatch):
    """Replace only native dispatch with independent existing arithmetic on real small weights."""
    native = importlib.import_module("pypto_serving.model.deepseek_v41.native_backbone")
    oracle = modules.numerics.ModelMath(harness.config, harness.ops)
    model = native.NativeModelMath(harness.config, harness.ops, None, {}, aux_width=16)
    calls = []

    def dense(names, dtype=torch.float32):
        return [harness.ops.weight(n).to(dtype) for n in names]

    def run(mode, *args):
        calls.append(mode)
        if mode == "mixes":
            x, function, scale, base, pre, post, comb = args
            flat = x.flatten(-2).float()
            raw = (flat @ function.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + oracle.eps)
            h, eps = oracle.hc, oracle.text["hc_eps"]
            pre.copy_(torch.sigmoid(raw[:, :h] * scale[0] + base[:h]) + eps)
            post.copy_(2 * torch.sigmoid(raw[:, h:2*h] * scale[1] + base[h:2*h]))
            mix = (raw[:, 2*h:] * scale[2] + base[2*h:]).unflatten(-1, (h, h)).softmax(-1) + eps
            mix /= mix.sum(-2, keepdim=True) + eps
            for _ in range(oracle.text["hc_sinkhorn_iters"] - 1):
                mix /= mix.sum(-1, keepdim=True) + eps
                mix /= mix.sum(-2, keepdim=True) + eps
            comb.copy_(mix)
        elif mode == "pre":
            x, coeff, output = args
            output.copy_(oracle.hc_pre(x, coeff))
        elif mode == "post":
            x, residual, coeff, comb, output = args
            output.copy_(oracle.hc_post(x, residual, coeff, comb))
        elif mode == "norm":
            x, weight, output, count = args
            output.copy_(modules.numerics.rms_norm(x, weight, oracle.eps))
        elif mode == "routing":
            x, gate, bias, indices, weights, count = args
            scores = torch.nn.functional.softplus(x[:count].float() @ gate.T).sqrt()
            selected = (scores + bias).topk(indices.shape[-1], -1).indices
            probability = scores.gather(-1, selected)
            probability /= probability.sum(-1, keepdim=True) + 1e-20
            probability *= oracle.text["routed_scaling_factor"]
            indices[:count].copy_(selected)
            weights[:count * indices.shape[-1], 0].copy_(probability.flatten())
        else:
            raise AssertionError(mode)

    def expert(x, prefix, weight, *, shared):
        assert 0 < len(x) <= 32
        calls.append(("shared" if shared else "routed", len(x)))
        return oracle.expert(x, prefix, None if shared else weight[:, None])

    monkeypatch.setattr(model, "_dense", dense)
    monkeypatch.setattr(model, "_run", run)
    monkeypatch.setattr(model, "_expert", expert)
    harness.backend.math = model
    harness.backend.model_provider = model
    return model, calls, oracle


def test_text_prefill_decode_keeps_engram_and_delayed_mixes(factory, modules, monkeypatch):
    reference, actual = factory(speculative=False), factory(speculative=False)
    native, calls, _ = install_cpu_dispatch(actual, modules, monkeypatch)
    for start, tokens in ((0, (3, 9, 12)), (3, (4, 18)), (5, (23,))):
        mode = "decode" if start == 5 else "prefill"
        expected = reference.execute(reference.work(tokens=tokens, start=start, mode=mode))[0]
        result = actual.execute(actual.work(tokens=tokens, start=start, mode=mode))[0]
        _fixtures.assert_logits(result, expected)
        if start == 3:
            reference.runner.finalize_prefill(["a"])
            actual.runner.finalize_prefill(["a"])
    assert {"mixes", "pre", "post", "norm", "routing"} <= set(c for c in calls if isinstance(c, str))
    assert native.engram.__func__ is modules.numerics.ModelMath.engram
    assert actual.runner.position("a") == 6
    assert actual.backend.diagnostics()["native_backbone"]["engram"] == "reference"


def test_moe_prefill_visits_every_row_across_expert_tiles(factory, modules, monkeypatch):
    harness = factory(speculative=False)
    native, calls, oracle = install_cpu_dispatch(harness, modules, monkeypatch)
    x = torch.randn((65, 32), generator=torch.Generator().manual_seed(37)).bfloat16()
    result = native.moe(x, "layers.0.ffn")
    expected = oracle.moe(x, "layers.0.ffn")
    torch.testing.assert_close(result, expected, atol=.004, rtol=.004)
    assert [c[1] for c in calls if isinstance(c, tuple) and c[0] == "shared"] == [32, 32, 1]


def test_native_failure_restores_cache_and_engram_before_retry(factory, modules, monkeypatch):
    harness = factory(speculative=False)
    native, _, _ = install_cpu_dispatch(harness, modules, monkeypatch)
    harness.execute(harness.work(tokens=(3,)))
    before = _fixtures.page_snapshot(harness.backend)
    reference = factory(speculative=False)
    reference.execute(reference.work(tokens=(3,)))
    expected = reference.execute(reference.work(tokens=(9, 12), start=1))[0]
    dispatch = native._run

    def fail(mode, *args):
        if mode == "routing":
            raise RuntimeError("injected native router failure after attention write")
        return dispatch(mode, *args)

    monkeypatch.setattr(native, "_run", fail)
    with pytest.raises(RuntimeError, match="injected native"):
        harness.execute(harness.work(tokens=(9, 12), start=1))
    assert harness.runner.position("a") == 1
    _fixtures.assert_pages(harness.backend, before)
    monkeypatch.setattr(native, "_run", dispatch)
    result = harness.execute(harness.work(tokens=(9, 12), start=1))[0]
    _fixtures.assert_logits(result, expected)


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_range", "nonfinite"])
def test_invalid_native_routes_fail_before_expert_dispatch(factory, modules, monkeypatch, corruption):
    harness = factory(speculative=False)
    native, _, _ = install_cpu_dispatch(harness, modules, monkeypatch)
    original = native._run

    def corrupt(mode, *args):
        original(mode, *args)
        if mode == "routing":
            indices, weights = args[3:5]
            if corruption == "duplicate":
                indices[0, 1] = indices[0, 0]
            elif corruption == "out_of_range":
                indices[0, 0] = harness.config.text_config["n_routed_experts"]
            else:
                weights[0, 0] = float("nan")

    monkeypatch.setattr(native, "_run", corrupt)
    monkeypatch.setattr(native, "_expert", lambda *a, **kw: pytest.fail("invalid routes must not dispatch"))
    with pytest.raises(ValueError, match="native router"):
        native.moe(torch.ones((1, 32), dtype=torch.bfloat16), "layers.0.ffn")


def test_resident_budget_eviction_and_partial_upload_cleanup(factory, modules):
    native = importlib.import_module("pypto_serving.model.deepseek_v41.native_backbone")
    harness = factory(speculative=False)

    class Worker:
        def __init__(self):
            self.live = {}
            self.allocations = 0
            self.fail_at = None

        def alloc_tensor(self, shape, dtype, init):
            self.allocations += 1
            if self.allocations == self.fail_at:
                raise RuntimeError("upload failure")
            tensor = SimpleNamespace(nbytes=init.numel() * init.element_size())
            self.live[id(tensor)] = tensor
            return tensor

        def free_tensor(self, tensor):
            del self.live[id(tensor)]

    worker = Worker()
    model = native.NativeModelMath(harness.config, harness.ops, SimpleNamespace(worker=worker), {},
                                   aux_width=16, weight_budget_bytes=16)
    load = lambda: [torch.ones(4)]
    first = model._resident("a", load)
    assert model._resident("a", lambda: pytest.fail("must reuse weights")) is first
    model._resident("b", load)
    assert len(worker.live) == 1 and "a" not in model.weights
    worker.fail_at = worker.allocations + 2
    with pytest.raises(RuntimeError, match="upload failure"):
        model._resident("c", lambda: [torch.ones(2), torch.ones(2)])
    assert not worker.live and model.weight_bytes == 0
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model._resident("d", load)


@pytest.mark.parametrize("platform,devices,speculation", [("a2a3", [0], 0), ("a5", [0, 1], 0), ("a5", [0], 5)])
def test_factory_rejects_unsupported_execution_before_loading(modules, monkeypatch, platform, devices, speculation):
    native = importlib.import_module("pypto_serving.model.deepseek_v41.native_backbone")
    attention = importlib.import_module("pypto_serving.model.deepseek_v41.native_attention")
    monkeypatch.setattr(attention, "_load_library", lambda: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="TP1|DSpark"):
        native.create_backend(config=None, runtime=SimpleNamespace(num_speculative_tokens=speculation),
                              cache_layouts=(), weight_loader=None, device_ids=devices, platform=platform)
