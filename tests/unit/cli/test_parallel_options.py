# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import pypto_serving.cli.main as cli


def _parse_cli_args(argv: list[str]):
    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize(("role", "specified", "expected"), [
    (None, None, 16), ("prefill", None, 16), ("decode", None, 72),
    (None, 32, 32), ("prefill", 8, 8), ("decode", 16, 16),
])
def test_request_capacity_defaults_reach_scheduler_and_cache(tmp_path, role, specified, expected):
    model = tmp_path / "model"
    model.mkdir()
    argv = ["--model", str(model)]
    if role is not None:
        argv += ["--pd-role", role]
    if specified is not None:
        argv += ["--max-num-seqs", str(specified)]
    config = cli.build_serving_engine_config(_parse_cli_args(argv))
    assert config.max_num_running_reqs == expected
    assert config.runtime_config.max_batch_size == expected


def test_cli_keeps_generic_model_chunk_sizes_unrestricted():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size("qwen", args.long_prefill_token_threshold)
    assert args.long_prefill_token_threshold == 128


def test_cli_keeps_dspark_chunk_sizes_under_its_kernel_contract():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size(
        "deepseek_v4",
        args.long_prefill_token_threshold,
        variant="dspark",
    )
    assert args.long_prefill_token_threshold == 128


def test_build_serving_engine_config_rejects_unsupported_deepseek_chunk_size(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = _parse_cli_args(
        ["--model", str(model_dir), "--long-prefill-token-threshold", "3072"]
    )

    with pytest.raises(
        ValueError,
        match="--long-prefill-token-threshold must be one of",
    ):
        cli.build_serving_engine_config(args)


def test_build_serving_engine_config_uses_parallel_config_for_devices(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.device_id == 0
    assert config.device_ids == ()
    assert config.worker_device_ids() == (0, 1)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.replica_device_groups == ((0, 1), (2, 3))


def test_build_serving_engine_config_rejects_invalid_parallel_topology(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="number of devices"):
        cli.build_serving_engine_config(args)


@pytest.mark.parametrize("draft_tokens", [0, 5])
@pytest.mark.parametrize("dp", [4, 8])
def test_v41_checkpoint_cli_keeps_its_speculative_runtime_contract(tmp_path, monkeypatch, draft_tokens, dp):
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.model.deepseek_v41 import npu_runner

    model_dir = tmp_path / "v41"
    model_dir.mkdir()
    raw = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120,
            "num_hidden_layers": 40, "num_attention_heads": 64, "head_dim": 512,
            "max_position_embeddings": 1048576, "hc_mult": 4, "rms_norm_eps": 1e-6,
            "compress_ratios": [0] * 40, "kv_source_layer_ids": [], "index_source_layer_ids": [],
            "n_routed_experts": 384, "dspark_n_routed_experts": 128,
            "num_experts_per_tok": 6, "dspark_num_experts_per_tok": 3,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    groups = (KVCacheGroupSpec("window", tuple(range(40)), KVCacheSpec(128, 128), 4,
                               num_partitions=dp),)

    def load_bindings(path, max_seq_len, *, dp_size):
        assert path == str(model_dir.resolve())
        assert max_seq_len == 512 and dp_size == dp
        return SimpleNamespace(
            require=lambda layers, placement: None,
            segment_topology=SimpleNamespace(capacity=72), cache_groups=groups,
        )

    monkeypatch.setattr(npu_runner, "load_composite_bindings", load_bindings)
    argv = [
        "--model", str(model_dir), "--platform", "a5", "--tp", "1", "--dp", str(dp), "--ep", str(dp),
        "--devices", ",".join(str(device) for device in range(dp)), "--block-size", "128",
        "--max-model-len", "512", "--max-num-seqs", "1", "--max-num-batched-tokens", "128",
    ]
    if draft_tokens:
        argv += ["--speculative-config", json.dumps({"method": "dspark", "num_speculative_tokens": draft_tokens})]
    config = cli.build_serving_engine_config(_parse_cli_args(argv))
    runtime = config.runtime_config
    assert runtime.num_speculative_tokens == draft_tokens
    assert runtime.kv_cache_groups == groups
    assert runtime.max_num_batched_tokens == 72
    assert runtime.supports_chunked_prefill_with_speculation is (draft_tokens == 0)
    assert config.worker_device_ids() == tuple(range(dp))
    assert config.executor_cls == cli._executor_cls_for_model_family("deepseek_v41")


def test_v4_checkpoint_cli_still_rejects_v41_draft_length(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v4"}', encoding="utf-8")
    args = _parse_cli_args([
        "--model", str(tmp_path), "--platform", "a5", "--tp", "4", "--dp", "2", "--ep", "8",
        "--devices", "0,1,2,3,4,5,6,7", "--block-size", "32",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
    ])
    with pytest.raises(ValueError, match="DeepSeek V4 method='dspark' speculation is fixed at"):
        cli.build_serving_engine_config(args)


def test_generate_config_rejects_wrong_value_types():
    for invalid in (
        {"stream": "false"},
        {"ignore_eos": "true"},
        {"stop": "END"},
        {"stop": [1, 2]},
        {"max_new_tokens": 2.5},
        {"max_new_tokens": True},
        {"max_new_tokens": 0},
        {"temperature": -0.1},
        {"temperature": "0"},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": "5"},
    ):
        with pytest.raises(ValueError, match="--generate-config"):
            cli._build_generate_config(invalid)


def test_generate_config_accepts_valid_values():
    config = cli._build_generate_config(
        {
            "max_new_tokens": 8,
            "temperature": 0.2,
            "top_p": 1,
            "top_k": None,
            "stop": ["END"],
            "stream": True,
            "ignore_eos": False,
        }
    )

    assert config.max_new_tokens == 8
    assert config.temperature == 0.2
    assert config.top_p == 1
    assert config.top_k is None
    assert config.stop == ("END",)
    assert config.stream is True
    assert config.ignore_eos is False


@pytest.mark.parametrize("options", [None, {"max_new_tokens": 8}])
def test_generate_config_defaults_to_greedy(options):
    config = cli._build_generate_config(options)

    assert config.temperature == 0.0


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_dspark_topology_accepts_the_8_card_and_16_card_worlds(tp):
    # Attention TP determines cache partitions; expert parallelism spans the world.
    for ep in (8, 16):
        dp = ep // tp
        devices = ",".join(str(device) for device in range(ep))
        args = _parse_cli_args(
            [
                "--model", "model",
                "--dp", str(dp),
                "--ep", str(ep),
                "--tp", str(tp),
                "--devices", devices,
                "--block-size", "32",
                "--max-num-seqs", "8",
                "--max-model-len", "1024",
            ]
        )

        cli._validate_dspark_topology(args)


@pytest.mark.parametrize("requested", [None, "", "diagnostic-output"])
def test_worker_only_exports_explicit_build_directory(monkeypatch, requested):
    import os
    from pathlib import Path
    from types import SimpleNamespace

    from pypto_serving.serving.server.serving_worker import WorkerProcess

    if requested is None:
        monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    else:
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", requested)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.config = SimpleNamespace(dp_rank=2)
    directory = worker._configure_pypto_build_dir((3, 4))
    assert directory == (Path(requested) / "serving_dp2_d3_4" if requested else None)
    assert os.environ.get("PYPTO_PROG_BUILD_DIR") == (str(directory) if requested else requested)
