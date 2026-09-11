# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small real tensor CPU oracles; no NPU or complete checkpoint is required."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
F = torch.nn.functional
ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vision = load_module("v41_vision_test", ROOT / "pypto_serving/model/deepseek_v41/vision.py")
SMALL = vision.VisionConfig(
    8, 6, 2, 2, 12, patch_size=2, downsample_ratio=2, max_image_tokens=40, min_pixels=0
)
GRID_CASES = json.loads((ROOT / "tests/fixtures/deepseek_v41/vision_grid_golden.json").read_text())["cases"]


class TensorOps:
    """Actual CPU linear algebra over deterministic tensors, with recorded matrix extents."""

    def __init__(self, config, dtype=torch.float32):
        generator = torch.Generator().manual_seed(512)
        self.weights = {
            name: (torch.randn(shape, generator=generator) * 0.1).to(dtype)
            for name, shape in vision.vision_weight_shapes(config).items()
        }
        for name in self.weights:
            if "norm" in name:
                self.weights[name].add_(1)
        self.matmul_shapes = []

    def weight(self, name):
        return self.weights[name]

    def linear(self, x, name, *, bias=False):
        return F.linear(x, self.weights[name + ".weight"], self.weights[name + ".bias"] if bias else None)

    def matmul(self, a, b):
        self.matmul_shapes.append((a.shape, b.shape))
        return a @ b


def sdpa_pixel_unshuffle_oracle(patches, n_h, n_w, config, ops):
    """Independent SDPA and pixel-unshuffle operators for the reference's mathematical graph."""

    def norm(x, key):
        return F.rms_norm(x.float(), (x.shape[-1],), ops.weight(key).float(), eps=1e-6).to(x.dtype)

    x = ops.linear(patches.flatten(1), "vision.patch_embed.proj", bias=True)
    head_dim = config.hidden_size // config.num_attention_heads
    frequencies = torch.tensor(
        [1 / config.rope_theta ** (2 * i / (head_dim // 2)) for i in range(head_dim // 4)]
    )
    coordinates = torch.tensor([(row, col) for row in range(n_h) for col in range(n_w)])
    angle = (coordinates[:, :, None] * frequencies).reshape(n_h * n_w, 1, head_dim // 2)
    for layer in range(config.num_hidden_layers):
        prefix = f"vision.blocks.{layer}"
        qkv = ops.linear(norm(x, prefix + ".norm1.weight"), prefix + ".attn.wqkv", bias=True)
        q, k, v = qkv.reshape(n_h * n_w, 3, config.num_attention_heads, head_dim).unbind(1)

        def rotate(value):
            first, second = value.float().chunk(2, -1)
            complex_value = torch.complex(first, second) * torch.polar(torch.ones_like(angle), angle)
            return torch.cat((complex_value.real, complex_value.imag), -1).to(value.dtype)

        output = (
            F.scaled_dot_product_attention(
                rotate(q).transpose(0, 1), rotate(k).transpose(0, 1), v.transpose(0, 1)
            )
            .transpose(0, 1)
            .flatten(1)
        )
        x = x + ops.linear(output, prefix + ".attn.wo", bias=True)
        gate, up = ops.linear(norm(x, prefix + ".norm2.weight"), prefix + ".mlp.w1").chunk(2, -1)
        x = x + ops.linear(F.silu(gate) * up, prefix + ".mlp.w2")
    r = config.downsample_ratio
    image = norm(x, "vision.norm.weight").T.reshape(1, config.hidden_size, n_h, n_w)
    image = F.pad(image, (0, -n_w % r, 0, -n_h % r))
    packed = F.pixel_unshuffle(image, r).flatten(2).squeeze(0).T
    return ops.linear(F.gelu(ops.linear(packed, "aligner.w1", bias=True)), "aligner.w2", bias=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_h,n_w", [(2, 2), (3, 5)])
def test_vit_and_aligner_against_independent_tensor_operators(dtype, n_h, n_w):
    ops = TensorOps(SMALL, dtype)
    patches = torch.linspace(-1, 1, n_h * n_w * 12).reshape(-1, 3, 2, 2).to(dtype)
    expected = sdpa_pixel_unshuffle_oracle(patches, n_h, n_w, SMALL, ops)
    actual = vision.VisionTower(SMALL, ops, query_chunk_size=3).encode_image(patches, n_h, n_w)
    torch.testing.assert_close(actual, expected, atol=3e-3 if dtype == torch.bfloat16 else 2e-6, rtol=0)
    assert actual.shape == (((n_h + 1) // 2) * ((n_w + 1) // 2), SMALL.text_hidden_size)
    assert all(a[-2] <= 3 for a, _ in ops.matmul_shapes)


@pytest.mark.parametrize("case", GRID_CASES)
def test_official_resize_plan_golden(case):
    config = replace(
        SMALL,
        patch_size=14,
        downsample_ratio=3,
        **{key: case[key] for key in ("min_pixels", "max_wh_ratio", "max_image_tokens")},
    )
    assert vision.plan_image_grid(case["width"], case["height"], config) == tuple(case["expected"])


def image_bytes(width=4, height=4):
    Image = pytest.importorskip("PIL.Image")
    source = Image.new("RGB", (width, height))
    source.putdata([(row * 30, col * 40, 255) for row in range(height) for col in range(width)])
    stream = io.BytesIO()
    source.save(stream, format="PNG")
    return stream.getvalue()


def test_patch_pixels_and_span_expansion_are_ordered_and_request_owned():
    data = image_bytes()
    patches, n_h, n_w, grid_h, grid_w = vision.load_image(data, SMALL)
    assert (n_h, n_w, grid_h, grid_w) == (2, 2, 1, 1)
    expected = torch.tensor([[[0, 0], [30, 30]], [[0, 40], [0, 40]], [[255, 255], [255, 255]]])
    torch.testing.assert_close(patches[0], ((expected.float() / 255 - 0.5) / 0.5).bfloat16(), atol=0, rtol=0)
    prompt = [0, SMALL.image_token_id, 7, SMALL.image_token_id, 8]
    result = vision.prepare_image_inputs(prompt, [data, {"data": base64.b64encode(data).decode()}], SMALL)
    assert result.tokens == (0,) + (SMALL.image_token_id,) * 4 + (7,) + (SMALL.image_token_id,) * 4 + (8,)
    assert result.token_types == (-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1)
    assert [image.start for image in result.images] == [1, 6]
    assert result.engram_mask == tuple(kind == -1 for kind in result.token_types)
    assert result.images[0].patches.data_ptr() != result.images[1].patches.data_ptr()
    assert prompt == [0, SMALL.image_token_id, 7, SMALL.image_token_id, 8]


def test_image_resource_resolution_and_budgets():
    data = b"abcd"
    assert vision.load_image_bytes({"source": {"data": base64.b64encode(data).decode()}}) == data
    assert vision.load_image_bytes({"url": "data:image/png;base64,YWJjZA=="}) == data
    seen = []
    assert (
        vision.load_image_bytes(
            {"url": "https://example.test/a.png"}, read_resource=lambda url: seen.append(url) or data
        )
        == data
    )
    assert seen == ["https://example.test/a.png"]
    for record in (
        {"url": "C:/private.png"},
        {"url": "https://example.test/a"},
        {"url": "data:image/png,abc"},
    ):
        with pytest.raises(ValueError):
            vision.load_image_bytes(record)
    with pytest.raises(ValueError, match="budget"):
        vision.load_image_bytes(data, max_bytes=3)
    with pytest.raises(ValueError, match="placeholder"):
        vision.prepare_image_inputs([SMALL.image_token_id], [], SMALL)
    with pytest.raises(ValueError, match="sequence"):
        vision.prepare_image_inputs([1, 2, 3], [], SMALL, max_sequence_tokens=2)


def test_merge_uses_learned_delimiters_and_is_atomic_on_failure():
    ops = TensorOps(SMALL)
    tower = vision.VisionTower(SMALL, ops)
    patches = torch.linspace(-1, 1, 4 * 12).reshape(4, 3, 2, 2)
    image = vision.ImageInput(1, patches, 2, 2, vision.image_token_types(1, 1))
    h = torch.ones(2, 7, SMALL.text_hidden_size)
    before = h.clone()
    result = tower.merge_embeddings(h, [[image], []])
    torch.testing.assert_close(result[0, 1], ops.weight("image_start"))
    torch.testing.assert_close(result[0, 2], tower.encode_image(patches, 2, 2)[0])
    torch.testing.assert_close(result[0, 3], ops.weight("image_newline"))
    torch.testing.assert_close(result[0, 4], ops.weight("image_end"))
    torch.testing.assert_close(result[1], before[1])
    torch.testing.assert_close(h, before)
    with pytest.raises(ValueError, match="ordered"):
        tower.merge_embeddings(h, [[image, replace(image, start=2)], []])
    with pytest.raises(ValueError, match="first chunk"):
        tower.merge_embeddings(h, [[image], []], start_pos=1)
    del ops.weights["vision.blocks.1.mlp.w2.weight"]
    with pytest.raises(KeyError):
        tower.merge_embeddings(h, [[image], []])
    torch.testing.assert_close(h, before)


def test_optional_pinned_official_vit_numerical_oracle():
    path = ROOT / "artifacts/deepseek-v41-p0/reference/inference__vision.py"
    if not path.exists():
        pytest.skip("optional downloaded pinned reference is absent; independent tensor oracle runs above")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c"
    )
    official = load_module("official_v41_vision_oracle", path)
    args = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=12,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        dim=6,
    )
    vit, aligner = official.ViT(args), official.Aligner(args)
    ops = TensorOps(SMALL)
    vit.load_state_dict(
        {
            name.removeprefix("vision."): weight
            for name, weight in ops.weights.items()
            if name.startswith("vision.")
        }
    )
    aligner.load_state_dict(
        {
            name.removeprefix("aligner."): weight
            for name, weight in ops.weights.items()
            if name.startswith("aligner.")
        }
    )
    patches = torch.linspace(-1, 1, 15 * 12).reshape(15, 3, 2, 2)
    with torch.inference_mode():
        expected = aligner(vit(patches, 3, 5), 3, 5)
        actual = vision.VisionTower(SMALL, ops, query_chunk_size=3).encode_image(patches, 3, 5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
