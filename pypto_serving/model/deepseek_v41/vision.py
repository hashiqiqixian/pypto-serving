# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 image preprocessing, bidirectional ViT, aligner, and first-chunk embedding merge.

Semantics follow DeepSeek-V4.1-Flash revision dba1be0a40aa45a94ad051997016db3960a90277:
inference/vision.py SHA256 5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c;
inference/image_processor.py SHA256 482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272.
See encoding.LICENSE for the upstream MIT notice. The implementation is stateless;
the caller owns per-request inputs and commits returned embeddings only on success.
Torch tensor operations provide CPU execution; injected linear/matmul operations
are the device integration boundary, not a claim of A5 kernel validation.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import io
import math
from typing import Protocol

import torch
import torch.nn.functional as F


TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
MAX_WIRE_IMAGES = 8
MAX_WIRE_PATCH_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    text_hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None
    image_token_id: int = 129264

    def __post_init__(self):
        for name in (
            "hidden_size",
            "text_hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "patch_size",
            "downsample_ratio",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads or self.hidden_size // self.num_attention_heads % 4:
            raise ValueError("vision head dimension must be divisible by four for two-axis RoPE")
        if type(self.max_image_tokens) is not int or self.max_image_tokens < 4:
            raise ValueError("max_image_tokens must be at least four")
        if type(self.min_pixels) is not int or self.min_pixels < 0:
            raise ValueError("min_pixels must be nonnegative")
        if type(self.image_token_id) is not int or self.image_token_id < 0:
            raise ValueError("image_token_id must be nonnegative")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if self.max_wh_ratio is not None and (not math.isfinite(self.max_wh_ratio) or self.max_wh_ratio <= 0):
            raise ValueError("max_wh_ratio must be finite and positive")

    @classmethod
    def from_config(cls, config: Mapping) -> VisionConfig:
        vision = dict(config["vision_config"])
        vision.pop("model_type", None)
        return cls(
            **vision,
            text_hidden_size=config["text_config"]["hidden_size"],
            image_token_id=config.get("image_token_id", 129264),
        )


@dataclass(frozen=True)
class ImageInput:
    """One owned image, with its absolute span offset in a single request."""

    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


@dataclass(frozen=True)
class VisionInputs:
    tokens: tuple[int, ...]
    token_types: tuple[int, ...]
    images: tuple[ImageInput, ...]

    @property
    def engram_mask(self) -> tuple[bool, ...]:
        return tuple(kind == TEXT for kind in self.token_types)


def plan_image_grid(width: int, height: int, config: VisionConfig) -> tuple[int, int, int, int]:
    """Return LLM height/width and resized pixel height/width, including delimiter budget."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image width and height must be positive integers")
    if config.max_wh_ratio is not None:
        width = min(width, height * config.max_wh_ratio)
    if width * height < config.min_pixels:
        ratio = math.sqrt(config.min_pixels / (width * height))
        width, height = int(width * ratio), int(height * ratio)
    p, r, budget = config.patch_size, config.downsample_ratio, config.max_image_tokens
    out_w, out_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    grid_h, grid_w = math.ceil(out_h / p / r), math.ceil(out_w / p / r)
    if grid_h * (grid_w + 1) + 2 > budget:
        aspect, cell = height / width, p * r
        columns = math.sqrt((budget - 2) / aspect + 0.25) - 0.5
        rows = columns * aspect
        if columns < 1:
            out_h, out_w = (budget - 2) // 2 * cell, cell
        elif rows < 1:
            out_h, out_w = cell, (budget - 3) * cell
        else:
            scale = min(math.floor(columns) * cell / width, math.floor(rows) * cell / height)
            out_h, out_w = math.floor(height * scale / p) * p, math.floor(width * scale / p) * p
        grid_h, grid_w = math.ceil(out_h / p / r), math.ceil(out_w / p / r)
    if min(grid_h, grid_w) < 1 or grid_h * (grid_w + 1) + 2 > budget:
        raise ValueError("image resize plan does not fit the token budget")
    return grid_h, grid_w, out_h, out_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    if min(n_llm_h, n_llm_w) <= 0:
        raise ValueError("image token grid must be positive")
    return torch.tensor(
        [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END], dtype=torch.int64
    )


def load_image_bytes(
    record: bytes | Mapping,
    *,
    read_resource: Callable[[str], bytes] | None = None,
    max_bytes: int = 32 * 1024 * 1024,
) -> bytes:
    """Decode official record formats; URL/path access is owned by the injected resolver."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if isinstance(record, bytes):
        data = record
    elif isinstance(record, Mapping):
        source = record.get("source")
        if isinstance(source, Mapping) and not isinstance(record.get("data"), (bytes, str)):
            return load_image_bytes(source, read_resource=read_resource, max_bytes=max_bytes)
        raw = record.get("data")
        if isinstance(raw, bytes):
            data = raw
        elif isinstance(raw, str):
            if len(raw) > 4 * math.ceil(max_bytes / 3):
                raise ValueError("encoded image exceeds the byte budget")
            data = base64.b64decode(raw, validate=True)
        elif isinstance(record.get("url"), str):
            url = record["url"]
            if url.startswith("data:"):
                header, separator, payload = url.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError("image data URLs must use base64")
                return load_image_bytes({"data": payload}, max_bytes=max_bytes)
            if read_resource is None:
                raise ValueError("image URL/path input requires an explicit resource resolver")
            data = read_resource(url)
        else:
            raise ValueError("image record requires bytes, base64 data, source, or URL")
    else:
        raise ValueError("image input must be bytes or a source record")
    if not isinstance(data, bytes) or not data or len(data) > max_bytes:
        raise ValueError("image must contain nonempty bytes within the byte budget")
    return data


def load_image(
    record: bytes | Mapping,
    config: VisionConfig,
    *,
    read_resource=None,
    max_bytes: int = 32 * 1024 * 1024,
    max_source_pixels: int = 64 * 1024 * 1024,
):
    """PIL padding/resize, [-1,1] BF16 normalization, and row-major NCHW patch extraction."""
    import numpy as np
    from PIL import Image, ImageOps

    data = load_image_bytes(record, read_resource=read_resource, max_bytes=max_bytes)
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > max_source_pixels:
                raise ValueError("decoded image exceeds the source pixel budget")
            image = source.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("image cannot be decoded within the source limits") from exc
    grid_h, grid_w, height, width = plan_image_grid(image.width, image.height, config)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    p = config.patch_size
    n_h, n_w = height // p, width // p
    patches = pixels.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(n_h * n_w, 3, p, p)
    return patches, n_h, n_w, grid_h, grid_w


def prepare_image_inputs(
    prompt_tokens: Sequence[int],
    images: Sequence[bytes | Mapping],
    config: VisionConfig,
    *,
    read_resource=None,
    max_sequence_tokens=1048576,
) -> VisionInputs:
    """Expand already-tokenized image placeholders; image records follow prompt order."""
    if any(type(token) is not int or token < 0 for token in prompt_tokens):
        raise ValueError("prompt tokens must be nonnegative integers")
    if sum(token == config.image_token_id for token in prompt_tokens) != len(images):
        raise ValueError("image placeholder count must match the number of image records")
    tokens, types, inputs = [], [], []
    image_iter = iter(images)
    for token in prompt_tokens:
        if token == config.image_token_id:
            patches, n_h, n_w, grid_h, grid_w = load_image(
                next(image_iter), config, read_resource=read_resource
            )
            span_types = image_token_types(grid_h, grid_w)
            inputs.append(ImageInput(len(tokens), patches, n_h, n_w, span_types))
            tokens.extend([token] * len(span_types))
            types.extend(span_types.tolist())
        else:
            tokens.append(token)
            types.append(TEXT)
        if len(tokens) > max_sequence_tokens:
            raise ValueError("expanded image/text prompt exceeds the sequence token budget")
    return VisionInputs(tuple(tokens), tuple(types), tuple(inputs))


def to_wire(inputs: VisionInputs) -> dict:
    """IPC-safe CPU patch buffers. Register this once per request, never on each decode."""
    images = []
    total = 0
    if not 0 < len(inputs.images) <= MAX_WIRE_IMAGES:
        raise ValueError("image count is outside the supported per-request range")
    for image in inputs.images:
        data = (
            image.patches.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        )
        total += len(data)
        if total > MAX_WIRE_PATCH_BYTES:
            raise ValueError("image patch buffers exceed the per-request byte budget")
        images.append(
            {"start": image.start, "n_vit_h": image.n_vit_h, "n_vit_w": image.n_vit_w, "patches_bf16": data}
        )
    return {
        "version": 1,
        "model_type": "deepseek_v41",
        "tokens": list(inputs.tokens),
        "token_types": list(inputs.token_types),
        "images": images,
        "first_chunk_end": max(image.start + len(image.types) for image in inputs.images),
    }


def from_wire(payload: Mapping, config: VisionConfig) -> VisionInputs:
    """Validate bounded transport metadata before allocating owned CPU tensors."""
    allowed = {"version", "model_type", "tokens", "token_types", "images", "first_chunk_end"}
    if (
        not isinstance(payload, Mapping)
        or set(payload) != allowed
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["model_type"] != "deepseek_v41"
    ):
        raise ValueError("unsupported V4.1 multimodal wire format")
    tokens, types, records = payload["tokens"], payload["token_types"], payload["images"]
    if (
        not isinstance(tokens, list)
        or not isinstance(types, list)
        or not 0 < len(tokens) <= 1048576
        or len(types) != len(tokens)
    ):
        raise ValueError("multimodal tokens/types must cover a bounded nonempty prompt")
    if any(type(token) is not int or token < 0 for token in tokens) or any(
        type(kind) is not int or kind not in (TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END)
        for kind in types
    ):
        raise ValueError("invalid multimodal token IDs or token types")
    if any(token == config.image_token_id and kind == TEXT for token, kind in zip(tokens, types)):
        raise ValueError("image token IDs must belong to a declared image span")
    if not isinstance(records, list) or not 0 < len(records) <= MAX_WIRE_IMAGES:
        raise ValueError("image count is outside the supported per-request range")
    plans, previous_end, total = [], 0, 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"start", "n_vit_h", "n_vit_w", "patches_bf16"}:
            raise ValueError("invalid image wire record")
        start, n_h, n_w = record["start"], record["n_vit_h"], record["n_vit_w"]
        if any(type(value) is not int or value < 1 for value in (n_h, n_w)) or type(start) is not int:
            raise ValueError("invalid image span/grid dimensions")
        grid_h, grid_w = math.ceil(n_h / config.downsample_ratio), math.ceil(n_w / config.downsample_ratio)
        count = grid_h * (grid_w + 1) + 2
        if count > config.max_image_tokens or start < previous_end or start + count > len(tokens):
            raise ValueError("image grid exceeds its token budget or span bounds")
        expected = image_token_types(grid_h, grid_w)
        if types[start : start + count] != expected.tolist() or any(
            kind != TEXT for kind in types[previous_end:start]
        ):
            raise ValueError("multimodal token types disagree with image spans")
        if any(token != config.image_token_id for token in tokens[start : start + count]):
            raise ValueError("image span must contain the configured image token ID")
        data = record["patches_bf16"]
        byte_count = n_h * n_w * 3 * config.patch_size**2 * 2
        if not isinstance(data, bytes) or len(data) != byte_count:
            raise ValueError("BF16 patch buffer byte length does not match its declared grid")
        total += byte_count
        if total > MAX_WIRE_PATCH_BYTES:
            raise ValueError("image patch buffers exceed the per-request byte budget")
        previous_end = start + count
        plans.append((record, expected))
    if payload["first_chunk_end"] != previous_end or any(kind != TEXT for kind in types[previous_end:]):
        raise ValueError("multimodal first chunk boundary or trailing token types are invalid")
    inputs = []
    for record, expected in plans:
        n_h, n_w = record["n_vit_h"], record["n_vit_w"]
        patches = torch.frombuffer(bytearray(record["patches_bf16"]), dtype=torch.bfloat16).reshape(
            n_h * n_w, 3, config.patch_size, config.patch_size
        )
        if not bool(torch.isfinite(patches).all()) or bool((patches.abs() > 1).any()):
            raise ValueError("preprocessed image patches must be finite normalized pixels in [-1, 1]")
        inputs.append(ImageInput(record["start"], patches, n_h, n_w, expected))
    return VisionInputs(tuple(tokens), tuple(types), tuple(inputs))


class VisionOps(Protocol):
    def weight(self, name: str) -> torch.Tensor: ...
    def linear(self, x: torch.Tensor, name: str, *, bias: bool = False) -> torch.Tensor: ...
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def vision_weight_shapes(config: VisionConfig) -> dict[str, tuple[int, ...]]:
    d, t, m = config.hidden_size, config.text_hidden_size, config.intermediate_size
    shapes = {
        "vision.patch_embed.proj.weight": (d, 3 * config.patch_size**2),
        "vision.patch_embed.proj.bias": (d,),
        "vision.norm.weight": (d,),
        "aligner.w1.weight": (t, d * config.downsample_ratio**2),
        "aligner.w1.bias": (t,),
        "aligner.w2.weight": (t, t),
        "aligner.w2.bias": (t,),
        "image_start": (t,),
        "image_end": (t,),
        "image_newline": (t,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"vision.blocks.{layer}"
        for suffix, shape in {
            "norm1.weight": (d,),
            "norm2.weight": (d,),
            "attn.wqkv.weight": (3 * d, d),
            "attn.wqkv.bias": (3 * d,),
            "attn.wo.weight": (d, d),
            "attn.wo.bias": (d,),
            "mlp.w1.weight": (2 * m, d),
            "mlp.w2.weight": (d, m),
        }.items():
            shapes[f"{prefix}.{suffix}"] = shape
    return shapes


class VisionTower:
    def __init__(self, config: VisionConfig, ops: VisionOps, *, query_chunk_size: int = 128):
        if type(query_chunk_size) is not int or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.config, self.ops = config, ops
        self.query_chunk_size = query_chunk_size

    def _norm(self, x, name):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (value * self.ops.weight(name)).to(x.dtype)

    def _rotary(self, x, cos, sin):
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        config, ops = self.config, self.ops
        n, d = n_vit_h * n_vit_w, config.hidden_size
        if min(n_vit_h, n_vit_w) < 1 or patches.shape != (n, 3, config.patch_size, config.patch_size):
            raise ValueError("image patch tensor does not match its declared grid")
        grid_h, grid_w = (
            math.ceil(n_vit_h / config.downsample_ratio),
            math.ceil(n_vit_w / config.downsample_ratio),
        )
        if grid_h * (grid_w + 1) + 2 > config.max_image_tokens:
            raise ValueError("image patch grid exceeds the configured token budget")
        x = ops.linear(patches.flatten(1), "vision.patch_embed.proj", bias=True)
        head_dim = d // config.num_attention_heads
        rope_dim = head_dim // 2
        inv = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim)
        )
        row = torch.arange(n_vit_h, device=x.device).unsqueeze(1).expand(n_vit_h, n_vit_w)
        col = torch.arange(n_vit_w, device=x.device).unsqueeze(0).expand(n_vit_h, n_vit_w)
        angles = (torch.stack((row, col), dim=-1).reshape(n, 2, 1).float() * inv).flatten(1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        for layer in range(config.num_hidden_layers):
            prefix = f"vision.blocks.{layer}"
            qkv = ops.linear(self._norm(x, f"{prefix}.norm1.weight"), f"{prefix}.attn.wqkv", bias=True)
            q, k, v = (part.reshape(n, config.num_attention_heads, head_dim) for part in qkv.chunk(3, dim=-1))
            q, k = self._rotary(q, cos, sin).transpose(0, 1), self._rotary(k, cos, sin).transpose(0, 1)
            # Query tiling retains full bidirectional keys while bounding the score buffer.
            chunks = []
            for start in range(0, n, self.query_chunk_size):
                scores = ops.matmul(
                    q[:, start : start + self.query_chunk_size].float(), k.float().transpose(-2, -1)
                ) / math.sqrt(head_dim)
                output = ops.matmul(scores.softmax(-1), v.transpose(0, 1).float()).to(v.dtype)
                chunks.append(output)
            output = torch.cat(chunks, dim=1).transpose(0, 1).reshape(n, d)
            x = x + ops.linear(output, f"{prefix}.attn.wo", bias=True)
            gate, up = ops.linear(self._norm(x, f"{prefix}.norm2.weight"), f"{prefix}.mlp.w1").chunk(
                2, dim=-1
            )
            x = x + ops.linear(F.silu(gate) * up, f"{prefix}.mlp.w2")
        x = self._norm(x, "vision.norm.weight").reshape(n_vit_h, n_vit_w, d).permute(2, 0, 1)
        r = config.downsample_ratio
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return ops.linear(F.gelu(ops.linear(x, "aligner.w1", bias=True)), "aligner.w2", bias=True)

    def merge_embeddings(
        self, h: torch.Tensor, images: Sequence[Sequence[ImageInput] | None], *, start_pos: int = 0
    ) -> torch.Tensor:
        """Validate all spans first, then construct new embeddings; inputs stay unchanged on failure."""
        if start_pos != 0:
            raise ValueError("image spans must be entirely prefilled in the first chunk")
        if h.ndim != 3 or h.shape[-1] != self.config.text_hidden_size or len(images) != h.shape[0]:
            raise ValueError("image batch must match [batch, sequence, text_hidden_size] embeddings")
        for sample in images:
            previous_end = 0
            for image in sample or ():
                expected = image_token_types(
                    math.ceil(image.n_vit_h / self.config.downsample_ratio),
                    math.ceil(image.n_vit_w / self.config.downsample_ratio),
                )
                if (
                    image.types.dtype != torch.int64
                    or image.types.shape != expected.shape
                    or not torch.equal(image.types.cpu(), expected)
                ):
                    raise ValueError("image token types do not match the aligner grid")
                if image.start < previous_end or image.start + len(expected) > h.shape[1]:
                    raise ValueError(
                        "image spans must be ordered, disjoint, and fully inside the first chunk"
                    )
                previous_end = image.start + len(expected)
        merged = h.clone()
        for batch, sample in enumerate(images):
            for image in sample or ():
                types = image.types.to(h.device)
                span = merged[batch, image.start : image.start + len(types)]
                for kind, name in (
                    (IMAGE_START, "image_start"),
                    (IMAGE_END, "image_end"),
                    (IMAGE_NEW_LINE, "image_newline"),
                ):
                    span[types == kind] = self.ops.weight(name).to(device=h.device, dtype=h.dtype)
                features = self.encode_image(image.patches.to(h.device), image.n_vit_h, image.n_vit_w)
                span[types == IMAGE] = features.to(h.dtype)
        return merged
