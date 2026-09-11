# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Register V4.1 metadata and lazy weights with the existing serving model loader."""

from pathlib import Path

import torch

from pypto_serving.config.types import LayerSpec, LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel
from pypto_serving.model.model_family import is_deepseek_v41_config, read_model_config
from pypto_serving.model.model_loader import ModelLoadRequest, SafetensorsDirectoryLoader
from pypto_serving.model.tokenizer import load_tokenizer

from .checkpoint import read_index
from .cache import configure_v41_runtime
from .config import DeepSeekV41Config
from .weight_spec import backbone_weight_specs


class DeepSeekV41DirectoryLoader(SafetensorsDirectoryLoader):
    """Load only V4.1 metadata; an executor must materialize and validate its weights."""

    format_names = ("deepseek_v41", "deepseek-v41", "dsv41")

    def _recognises(self, model_path: Path) -> bool:
        return is_deepseek_v41_config(read_model_config(model_path))

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        root = Path(request.model_dir)
        parsed = DeepSeekV41Config.from_json(root / "config.json")
        raw = read_model_config(root)
        index = read_index(root / "model.safetensors.index.json")
        weight_map = index["weight_map"]
        expected = backbone_weight_specs(raw)
        missing = expected.keys() - weight_map.keys()
        unexpected = [
            name
            for name in weight_map.keys() - expected.keys()
            if not name.startswith(("vision.", "aligner.", "image_", "mtp."))
        ]
        if missing or unexpected:
            raise ValueError(
                f"V4.1 weight index mismatch: missing={sorted(missing)[:8]}, "
                f"unrecognized={sorted(unexpected)[:8]}"
            )
        for filename in set(weight_map.values()):
            path = (root / filename).resolve()
            if path.parent != root.resolve() or not path.is_file():
                raise FileNotFoundError(
                    f"V4.1 checkpoint shard missing or outside model directory: {filename}"
                )
        tokenizer = load_tokenizer(
            root, trust_remote_code=bool(request.loader_options.get("trust_remote_code", False))
        )
        text = parsed.text_config
        config = ModelConfig(
            model_id=request.model_id,
            architecture="DeepseekV41ForCausalLM",
            vocab_size=parsed.vocab_size,
            hidden_size=parsed.hidden_size,
            intermediate_size=parsed.moe_intermediate_size,
            num_hidden_layers=parsed.num_hidden_layers,
            num_attention_heads=parsed.num_attention_heads,
            num_key_value_heads=parsed.num_key_value_heads,
            head_dim=parsed.head_dim,
            max_position_embeddings=parsed.max_position_embeddings,
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=float(text["rope_theta"]),
            bos_token_id=raw.get("bos_token_id", tokenizer.bos_token_id),
            eos_token_id=raw.get("eos_token_id", tokenizer.eos_token_id),
            pad_token_id=raw.get("pad_token_id", tokenizer.pad_token_id),
            torch_dtype="bfloat16",
        )
        runtime = request.runtime_config or RuntimeConfig(
            max_seq_len=min(8192, parsed.max_position_embeddings)
        )
        if runtime.max_seq_len > parsed.max_position_embeddings or runtime.max_seq_len <= 0:
            raise ValueError("V4.1 runtime max_seq_len must fit checkpoint positions")
        runtime = configure_v41_runtime(parsed, runtime)
        placeholder = torch.empty(0, parsed.hidden_size, dtype=torch.bfloat16)
        model = RuntimeModel(
            config,
            runtime,
            placeholder,
            torch.empty(0, dtype=torch.bfloat16),
            placeholder,
            extra={
                "family": "deepseek_v41",
                "checkpoint_format": "fp8-fp4-ue8m0",
                "config_data": raw,
                "v41_config": parsed,
                "weight_map": weight_map,
                "model_dir": str(root),
                "weights_materialized": False,
            },
        )
        layers = [
            LayerSpec(
                i,
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
            for i in range(config.num_hidden_layers)
        ]
        return LoadedModel(request.model_id, str(root), config, tokenizer, layers, model)
