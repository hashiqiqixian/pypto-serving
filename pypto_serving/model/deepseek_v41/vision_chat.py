# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded OpenAI data-image content normalization for the V4.1 serving tokenizer."""

from __future__ import annotations

from collections.abc import Mapping
import io

from pypto_serving.model.tokenizer import PreparedPrompt

from .encoding import IMAGE_TOKEN
from .vision import MAX_WIRE_IMAGES, VisionConfig, load_image_bytes, prepare_image_inputs, to_wire


MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_PIXELS = 16 * 1024 * 1024


def prepare_chat_prompt(adapter, messages, **kwargs):
    """Preserve official block separators; never resolve URLs or local server paths."""
    if kwargs.get("tokenize", False):
        raise ValueError("multimodal chat requires tokenize=False so image data stays attached to the prompt")
    normalized, images, total = [], [], 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("chat messages must be objects")
        record = dict(message)
        content = record.get("content")
        if isinstance(content, list):
            text = []
            for block in content:
                if not isinstance(block, Mapping):
                    raise ValueError("chat content blocks must be objects")
                if block.get("type") == "text" and set(block) <= {"type", "text"}:
                    value = block.get("text")
                    if not isinstance(value, str) or IMAGE_TOKEN in value:
                        raise ValueError("text blocks must be strings without raw image placeholder tokens")
                    text.append(value)
                elif block.get("type") == "image_url" and set(block) == {"type", "image_url"}:
                    source = block["image_url"]
                    if not isinstance(source, Mapping) or not set(source) <= {"url", "detail"}:
                        raise ValueError("image_url must be an object with url and optional detail")
                    if source.get("detail", "auto") not in ("auto", "low", "high"):
                        raise ValueError("unsupported image detail value")
                    url = source.get("url")
                    if not isinstance(url, str) or not url.startswith("data:image/"):
                        raise ValueError("V4.1 serving accepts only data:image/...;base64 image URLs")
                    data = load_image_bytes({"url": url}, max_bytes=MAX_IMAGE_BYTES)
                    total += len(data)
                    if total > MAX_REQUEST_IMAGE_BYTES or len(images) >= MAX_WIRE_IMAGES:
                        raise ValueError("request exceeds the bounded image count or byte budget")
                    from PIL import Image

                    try:
                        with Image.open(io.BytesIO(data)) as image:
                            if image.width * image.height > MAX_SOURCE_PIXELS:
                                raise ValueError("image exceeds the decoded pixel budget")
                    except (OSError, Image.DecompressionBombError) as exc:
                        raise ValueError("inline image cannot be decoded within the image limits") from exc
                    images.append(data)
                    text.append(IMAGE_TOKEN)
                else:
                    raise ValueError("V4.1 content blocks support only text and inline image_url")
            record["content"] = "\n\n".join(text)
        elif not isinstance(content, str) or IMAGE_TOKEN in content:
            raise ValueError("message text must be a string without raw image placeholders")
        normalized.append(record)
    prompt = adapter._render_text(normalized, bool(images), **kwargs)
    if not images:
        return prompt
    if adapter.raw_config is None:
        raise ValueError("multimodal tokenizer requires the local V4.1 model config")
    config = VisionConfig.from_config(adapter.raw_config)
    prepared = prepare_image_inputs(adapter.encode(prompt), images, config)
    return PreparedPrompt(prompt, list(prepared.tokens), to_wire(prepared))
