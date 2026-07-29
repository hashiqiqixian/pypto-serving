# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import json
import stat
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor

import pypto_serving.cli.main as cli
from pypto_serving.config.types import DecodeBatch, PrefillBatch, RuntimeConfig
from pypto_serving.model import model_loader
from pypto_serving.model import tokenizer as tokenizer_module
from pypto_serving.model.deepseek import npu_executor, weight_loader
from pypto_serving.model.deepseek.npu_runner import (
    DEEPSEEK_V4_LM_HEAD_TP_SIZE,
    DeepSeekV4CacheLayout,
    DeepSeekV4CacheMetadataBuilder,
    DeepSeekV4CompiledKernels,
    DeepSeekV4L3Callable,
    DeepSeekV4ModelRunner,
    accept_mtp_tokens,
    build_deepseek_v4_cache_group_specs,
    build_deepseek_v4_layer_plan,
    deepseek_v4_cache_blocks_for_slots,
    deepseek_v4_decode_layout,
)
from pypto_serving.model.deepseek.weight_loader import (
    DEEPSEEK_V4_PACKED_FORMAT,
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
    deepseek_v4_layer_core_weight_names,
    deepseek_v4_packed_weights_path,
    deepseek_v4_hadamard_idx,
    deepseek_v4_local_expert_ids,
    deepseek_v4_routed_expert_weight_names,
    deepseek_v4_startup_weight_names,
    pack_deepseek_v4_layer_weights,
)
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.tools import prepack_deepseek_v4


def test_deepseek_kernel_dir_uses_v4_flash_variant(tmp_path):
    kernel_dir = tmp_path / "models" / "deepseek_v4_flash_mtp"
    kernel_dir.mkdir(parents=True)

    assert npu_executor._find_pypto_lib_deepseek_v4_dir(str(tmp_path)) == kernel_dir
    assert npu_executor._is_deepseek_v4_module_file(kernel_dir / "decode_fwd.py", kernel_dir)


def test_accept_mtp_tokens_commits_second_main_token_only_on_draft_match():
    accepted = accept_mtp_tokens(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
        torch.tensor([[11], [99]], dtype=torch.long),
    )

    assert accepted == [[11, 12], [21]]


def test_accept_mtp_tokens_commits_matching_prefix_and_one_target_token():
    accepted = accept_mtp_tokens(
        torch.tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [11, 12, 13],
                [21, 99, 23],
                [99, 32, 33],
            ],
            dtype=torch.long,
        ),
    )

    assert accepted == [[11, 12, 13, 14], [21, 22], [31]]


def test_deepseek_mtp_proposer_reuses_recurrent_hidden_for_configured_depth(monkeypatch):
    runner, _model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 3
    runner._mtp_request_states["req-a"] = SimpleNamespace(
        draft_token_id=10,
        draft_pre_hc_hidden=torch.zeros((4, 1)),
        draft_position=129,
    )
    calls = []

    monkeypatch.setattr(runner, "_initialize_mtp_drafts", lambda batch: None)

    def fake_step(model, batch, token_ids, previous_hidden, positions):
        calls.append((token_ids.tolist(), positions.tolist(), previous_hidden.clone()))
        return token_ids + 1, previous_hidden + 1

    monkeypatch.setattr(runner, "_run_mtp_token_step", fake_step)
    drafts = runner._propose_mtp_tokens(
        _model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[9]], dtype=torch.long),
            hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([129], dtype=torch.int32),
        )
    )

    assert drafts.tolist() == [[10, 11, 12]]
    assert [positions for _, positions, _ in calls] == [[129], [130]]
    torch.testing.assert_close(calls[1][2], torch.ones((1, 4, 1)))


def test_deepseek_mtp_token_step_runs_one_request_per_rank_wave(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    layout = deepseek_v4_decode_layout(3)
    runner._compiled.layout = layout
    buffers = SimpleNamespace(
        decode_input_ids=torch.empty((layout.ranks, layout.decode_tokens), dtype=torch.long),
        decode_position_ids=torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.int32,
        ),
        decode_accepted_counts=torch.empty(
            (layout.ranks, layout.decode_batch),
            dtype=torch.int32,
        ),
        decode_tail_slot_ids=torch.empty(
            (layout.ranks, layout.decode_batch),
            dtype=torch.int32,
        ),
        decode_pre_hc_out=torch.empty(
            (layout.ranks, layout.decode_tokens, 4, 1),
            dtype=torch.float32,
        ),
        decode_sampled_ids=torch.empty(
            (layout.ranks, layout.decode_tokens, 8),
            dtype=torch.int32,
        ),
        decode_logit_row_indices=torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.int32,
        ),
    )
    runner._mtp_buffers = buffers
    indices_by_rank = ((0, 1),) + ((),) * (layout.ranks - 1)
    assignment = SimpleNamespace(
        ranks=(0, 0),
        local_rows=(0, 1),
        per_rank_counts=(2,) + (0,) * (layout.ranks - 1),
        indices_by_rank=indices_by_rank,
    )
    monkeypatch.setattr(runner, "_decode_assignment", lambda batch: assignment)
    prepared_request_ids = []

    def fake_prepare(model, batch, **kwargs):
        prepared_request_ids.append(tuple(batch.request_ids))
        return SimpleNamespace(ranks=(0,), local_rows=(0,))

    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_stage_decode_inputs", lambda prepared: prepared)
    monkeypatch.setattr(runner, "_mtp_decode_args", lambda: ())
    monkeypatch.setattr(runner, "_require_mtp_decode_callable", lambda: object())
    runner._mtp_request_states = {
        "req-a": SimpleNamespace(tail_rank=0, tail_slot_id=0),
        "req-b": SimpleNamespace(tail_rank=0, tail_slot_id=1),
    }
    uploaded = []
    monkeypatch.setattr(
        runner,
        "_write_mtp_tail_hidden",
        lambda state, rank, hidden: uploaded.append((state.tail_slot_id, hidden.clone())),
    )
    active_tokens = []

    def fake_run_l3(callable_spec, *args):
        active_tokens.append(int(args[-1]))
        call_index = len(active_tokens)
        buffers.decode_sampled_ids[0, 0, 0] = 10 + 10 * (call_index - 1)
        buffers.decode_pre_hc_out[0, 0].fill_(call_index)

    monkeypatch.setattr(runner, "_run_l3", fake_run_l3)
    batch = DecodeBatch(
        request_ids=["req-a", "req-b"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([129, 130], dtype=torch.int32),
        block_ids_by_group=[{"ori": [7]}, {"ori": [8]}],
    )
    token_ids = torch.tensor([10, 20], dtype=torch.long)
    previous_hidden = torch.stack(
        (torch.full((4, 1), 3.0), torch.full((4, 1), 4.0))
    )
    positions = torch.tensor([129, 130], dtype=torch.int32)
    next_tokens, next_hidden = runner._run_mtp_token_step(
        model,
        batch,
        token_ids,
        previous_hidden,
        positions,
    )

    assert active_tokens == [1, 1]
    assert prepared_request_ids == [("req-a",), ("req-b",)]
    assert [slot for slot, _hidden in uploaded] == [0, 1]
    assert next_tokens.tolist() == [10, 20]
    torch.testing.assert_close(next_hidden[:, 0, 0], torch.tensor([1.0, 2.0]))


def test_deepseek_mtp_target_verification_chunks_arbitrary_depth(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(9)
    runner._compiled.num_speculative_tokens = 9
    prepared_chunks = []

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        chunk = SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
            positions=positions,
            active_width=active_width,
        )
        prepared_chunks.append(chunk)
        return chunk

    def fake_execute(model, prepared, *, active_seq):
        ranks = tuple(range(len(prepared.request_ids)))
        logits = torch.zeros((8, active_seq, 128), dtype=torch.float32)
        sampled_ids = torch.zeros((8, active_seq, 8), dtype=torch.int32)
        pre_hc = torch.zeros((8, active_seq, 4, 1), dtype=torch.float32)
        for row, token_row in enumerate(prepared.token_rows):
            predictions = token_row[:active_seq] + 1
            logits[row, torch.arange(active_seq), predictions] = 1
            sampled_ids[row, :, 0] = predictions
            pre_hc[row, :, 0, 0] = torch.arange(active_seq)
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=ranks, local_rows=(0,) * len(ranks)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=pre_hc,
        )

    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )
    verification = runner._verify_mtp_drafts(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[9], [19]], dtype=torch.long),
            hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([100, 100], dtype=torch.int32),
            cache_partitions=[0, 1],
        ),
        torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                [20, 21, 99, 23, 24, 25, 26, 27, 28],
            ],
            dtype=torch.long,
        ),
    )

    assert verification.accepted_token_ids == [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22],
    ]
    assert verification.tail_token_ids.tolist() == [19, 22]
    assert verification.tail_positions.tolist() == [109, 102]
    assert [chunk.active_width for chunk in prepared_chunks] == [8, 2]
    assert prepared_chunks[1].request_ids == ("req-a",)


def test_deepseek_mtp_partial_target_chunk_waves_requests_on_same_rank(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(2)
    runner._compiled.num_speculative_tokens = 2
    prepared_request_ids = []

    def fake_assignment(batch):
        count = len(batch.request_ids)
        return SimpleNamespace(
            ranks=(0,) * count,
            local_rows=tuple(range(count)),
            per_rank_counts=(count,) + (0,) * 7,
            indices_by_rank=(tuple(range(count)),) + ((),) * 7,
        )

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        prepared_request_ids.append(tuple(batch.request_ids))
        return SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
        )

    def fake_execute(model, prepared, *, active_seq):
        predictions = prepared.token_rows + 1
        logits = torch.zeros((8, 4, 128), dtype=torch.float32)
        sampled_ids = torch.zeros((8, 4, 8), dtype=torch.int32)
        logits[0, torch.arange(active_seq), predictions[0, :active_seq]] = 1
        sampled_ids[0, :active_seq, 0] = predictions[0, :active_seq]
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=(0,), local_rows=(0,)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=torch.zeros((8, 4, 4, 1), dtype=torch.float32),
        )

    monkeypatch.setattr(runner, "_decode_assignment", fake_assignment)
    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )

    verification = runner._verify_mtp_drafts(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[9], [19]], dtype=torch.long),
            hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([100, 100], dtype=torch.int32),
        ),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
    )

    assert prepared_request_ids == [("req-a",), ("req-b",)]
    assert verification.accepted_token_ids == [[10, 11, 12], [20, 21, 22]]


@pytest.mark.parametrize(
    ("num_speculative_tokens", "decode_seq", "decode_batch"),
    [(0, 1, 8), (1, 2, 4), (2, 4, 2), (3, 4, 2), (4, 8, 1), (32, 8, 1)],
)
def test_deepseek_mtp_depth_selects_fixed_eight_row_layout(
    num_speculative_tokens,
    decode_seq,
    decode_batch,
):
    layout = deepseek_v4_decode_layout(num_speculative_tokens)

    assert layout.decode_seq == decode_seq
    assert layout.decode_batch == decode_batch
    assert layout.decode_tokens == 8


def test_deepseek_mtp_draft_depth_is_capped_by_remaining_context():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 9
    model.runtime = replace(model.runtime, max_seq_len=130)
    batch = DecodeBatch(
        request_ids=["req-a", "req-b"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([128, 125], dtype=torch.int32),
    )

    assert runner._mtp_draft_count(model, batch) == 2


def test_deepseek_mtp_corrects_async_lengths_from_committed_tokens():
    runner, _model = _runner_for_prepared_inputs()
    runner._mtp_request_states = {
        "continued": SimpleNamespace(
            proposed_tokens=3,
            prompt_len=100,
            committed_count=5,
        ),
        "first-step": SimpleNamespace(
            proposed_tokens=0,
            prompt_len=40,
            committed_count=0,
        ),
    }
    batch = DecodeBatch(
        request_ids=["continued", "first-step"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([999, 41], dtype=torch.int32),
    )

    assert runner._correct_mtp_seq_lens(batch).tolist() == [106, 41]


def test_deepseek_mtp_exhausted_context_verifies_single_target_column(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(9)
    runner._compiled.num_speculative_tokens = 9
    model.runtime = replace(model.runtime, max_seq_len=130)
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[9]], dtype=torch.long),
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([130], dtype=torch.int32),
        cache_partitions=[0],
    )
    prepared_chunks = []

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        chunk = SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
            positions=positions,
            active_width=active_width,
        )
        prepared_chunks.append(chunk)
        return chunk

    def fake_execute(model, prepared, *, active_seq):
        logits = torch.zeros((8, active_seq, 128), dtype=torch.float32)
        logits[0, 0, 10] = 1
        sampled_ids = torch.zeros((8, active_seq, 8), dtype=torch.int32)
        sampled_ids[0, 0, 0] = 10
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=(0,), local_rows=(0,)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=torch.zeros((8, active_seq, 4, 1), dtype=torch.float32),
        )

    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )

    num_drafts = runner._mtp_draft_count(model, batch)
    drafts = runner._propose_mtp_tokens(model, batch, num_drafts=num_drafts)
    verification = runner._verify_mtp_drafts(model, batch, drafts)

    assert num_drafts == 0
    assert drafts.shape == (1, 0)
    assert verification.accepted_token_ids == [[10]]
    assert verification.tail_token_ids.tolist() == [10]
    assert verification.tail_positions.tolist() == [130]
    assert len(prepared_chunks) == 1
    assert prepared_chunks[0].active_width == 1
    assert prepared_chunks[0].token_rows.tolist() == [[9] * 8]


def test_cli_selects_deepseek_executor_and_configures_mtp_depth(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3,4,5,6,7",
            "--dp", "8",
            "--ep", "8",
            "--tp", "1",
            "--block-size", "128",
            "--max-model-len", "260",
            "--dtype", "int8",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":4}',
            "--max-num-seqs", "8",
            "--use-compile-cache",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.device_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert config.parallel_config.replica_device_groups == ((0, 1, 2, 3, 4, 5, 6, 7),)
    assert config.runtime_config.page_size == 128
    assert config.runtime_config.weight_dtype == "int8"
    assert config.enable_prefix_cache is False
    assert config.executor_kwargs["num_speculative_tokens"] == 4
    assert config.runtime_config.num_speculative_tokens == 4
    assert config.max_num_running_reqs == 8
    assert config.executor_kwargs["use_compile_cache"] is True


def test_cli_keeps_deepseek_autoregressive_decode_when_mtp_is_disabled(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--dp",
            "8",
            "--ep",
            "8",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_kwargs["num_speculative_tokens"] == 0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"method": "draft_model", "num_speculative_tokens": 3}, "method='mtp'"),
        ({"method": "mtp"}, "requires num_speculative_tokens"),
        ({"method": "mtp", "num_speculative_tokens": 0}, "must be positive"),
    ],
)
def test_cli_rejects_invalid_deepseek_speculative_config(config, message):
    args = SimpleNamespace(
        speculative_config=config,
        num_speculative_tokens=None,
        enable_mtp=None,
    )

    with pytest.raises(ValueError, match=message):
        cli._resolve_num_speculative_tokens(args)


def test_cli_rejects_speculative_config_with_deprecated_alias():
    args = SimpleNamespace(
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        num_speculative_tokens=2,
        enable_mtp=None,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        cli._resolve_num_speculative_tokens(args)


def test_tokenizer_falls_back_when_deepseek_config_fails_strict_validation(tmp_path, monkeypatch):
    class StrictDataclassFieldValidationError(Exception):
        pass

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise StrictDataclassFieldValidationError("attention_dropout expected float")

    sentinel = object()
    fake_transformers = type(
        "FakeTransformers",
        (),
        {"AutoTokenizer": AutoTokenizer, "PreTrainedTokenizerFast": object},
    )
    fake_hub_errors = ModuleType("huggingface_hub.errors")
    fake_hub_errors.StrictDataclassFieldValidationError = StrictDataclassFieldValidationError
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_hub_errors)
    monkeypatch.setattr(tokenizer_module, "_load_fast_tokenizer_from_file", lambda *args: sentinel)
    (tmp_path / "tokenizer.json").touch()

    adapter = tokenizer_module.TransformersTokenizerAdapter.from_pretrained(str(tmp_path))

    assert adapter.tokenizer is sentinel


def test_deepseek_compile_attaches_lazy_weight_store_without_opening_shards(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(
        model_loader,
        "load_tokenizer",
        lambda *args, **kwargs: _Tokenizer(),
    )
    opened: list[Path] = []

    def _fail_open(path: Path, device: str):
        opened.append(path)
        raise AssertionError(f"unexpected safetensors open on {device}: {path}")

    monkeypatch.setattr(weight_loader, "_default_safe_open", _fail_open)
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(
            page_size=128,
            max_batch_size=4,
            max_seq_len=256,
            weight_dtype="int8",
        ),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(platform="a2a3sim", device_ids=tuple(range(8)))

    compiled = executor._compile_model(loaded.runtime_model)

    assert opened == []
    assert isinstance(compiled.weight_store, DeepSeekV4WeightStore)
    assert compiled.weight_store.filename_for("head.weight") == "model-00001-of-00001.safetensors"
    assert compiled.weight_store.device == "cpu"
    assert compiled.layer_plan[0].attention_kind == "swa"
    assert compiled.layer_plan[2].attention_kind == "csa"
    assert compiled.layer_plan[2].include_tid2eid is True
    assert compiled.layer_plan[3].attention_kind == "hca"
    assert compiled.layer_plan[3].include_gate_bias is True


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_deepseek_compile_selects_mtp_programs(
    tmp_path,
    monkeypatch,
    num_speculative_tokens,
):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(model_loader, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    monkeypatch.setattr(DeepSeekV4WeightStore, "validate_mtp_startup_contract", lambda *args, **kwargs: None)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(
            page_size=128,
            max_batch_size=4,
            max_seq_len=256,
            weight_dtype="int8",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )

    prefill_fwd = SimpleNamespace(l3_prefill_fwd=object())
    decode_fwd = SimpleNamespace(l3_decode_fwd=object())
    decode_fwd_mtp = SimpleNamespace(l3_decode_fwd_mtp=object())
    decode_mtp = SimpleNamespace(l3_mtp_decode_layer=object())
    prefill_mtp = SimpleNamespace(l3_mtp_prefill_fwd=object())
    compile_calls: list[tuple[str, object, frozenset[str] | None]] = []

    def _fake_compile(self, name, jit_fn, *, layout, runtime_scalar_names=None):
        assert layout == deepseek_v4_decode_layout(num_speculative_tokens)
        compile_calls.append((name, jit_fn, runtime_scalar_names))
        return DeepSeekV4L3Callable(compiled=object(), name=name)

    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_load_kernel_modules",
        lambda self, layout: {
            "config": object(),
            "prefill_fwd": prefill_fwd,
            "decode_fwd": decode_fwd,
            "decode_fwd_mtp": decode_fwd_mtp,
            "decode_mtp": decode_mtp,
            "prefill_mtp": prefill_mtp,
            "utils": object(),
        },
    )
    monkeypatch.setattr(npu_executor.DeepSeekV4PyptoExecutor, "_compile_l3_callable", _fake_compile)
    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_build_rope_tables",
        lambda self, utils_module, config_module: (torch.empty(1), torch.empty(1)),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        compile_kernels=True,
        num_speculative_tokens=num_speculative_tokens,
    )

    compiled = executor._compile_model(loaded.runtime_model)

    expected_calls = [
        ("deepseek_v4_prefill", prefill_fwd.l3_prefill_fwd, None),
    ]
    if num_speculative_tokens == 1:
        expected_calls.append((
            "deepseek_v4_decode_mtp_fused",
            decode_fwd_mtp.l3_decode_fwd_mtp,
            frozenset({"mtp_num_tokens"}),
        ))
    else:
        expected_calls.append(("deepseek_v4_decode", decode_fwd.l3_decode_fwd, None))
    expected_calls.extend([
        ("deepseek_v4_mtp_prefill", prefill_mtp.l3_mtp_prefill_fwd, frozenset({"num_tokens"})),
    ])
    if num_speculative_tokens > 1:
        expected_calls.append(
            ("deepseek_v4_mtp_decode", decode_mtp.l3_mtp_decode_layer, frozenset({"num_tokens"}))
        )
    assert compile_calls == expected_calls
    assert compiled.prefill is not None
    assert compiled.decode is not None
    assert compiled.mtp_prefill is not None
    assert (compiled.mtp_decode is None) == (num_speculative_tokens == 1)


def test_deepseek_l3_compile_preserves_runtime_scalars_with_meta_tensors():
    """Runtime scalars become meta-tensor/ctypes compile args forwarded to the compiler."""
    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(self, name, jit_fn, compile_args, *, use_cache=False):
            captured["name"] = name
            captured["compile_args"] = compile_args
            return "compiled"

    def _kernel(x, num_tokens):
        pass

    _kernel.__annotations__ = {
        "x": SimpleNamespace(shape=(2, 3), dtype="fp32"),
        "num_tokens": SimpleNamespace(dtype="int32"),
    }

    class _JitFunction:
        def __init__(self) -> None:
            self._func = _kernel

    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiler = _FakeCompiler()
    executor._use_compile_cache = False

    compiled = executor._compile_l3_callable(
        "deepseek_v4_mtp_prefill",
        _JitFunction(),
        layout=DeepSeekV4CacheLayout(),
        runtime_scalar_names=frozenset({"num_tokens"}),
    )

    assert compiled == "compiled"
    assert captured["name"] == "deepseek_v4_mtp_prefill"
    args = captured["compile_args"]
    assert len(args) == 2
    assert isinstance(args[0], torch.Tensor)
    assert args[0].device.type == "meta"
    assert args[0].shape == (2, 3)
    assert args[0].dtype == torch.float32
    assert isinstance(args[1], ctypes.c_int32)
    assert args[1].value == 0


def test_deepseek_compile_l3_callable_threads_use_cache_to_compiler():
    """_compile_l3_callable forwards use_compile_cache to the shared compiler."""
    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(self, name, jit_fn, compile_args, *, use_cache=False):
            captured["name"] = name
            captured["use_cache"] = use_cache
            return "compiled"

    def _kernel(x: tuple[int, int]):
        pass

    class _JitFunction:
        _func = _kernel

    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiler = _FakeCompiler()
    executor._use_compile_cache = True

    compiled = executor._compile_l3_callable(
        "deepseek_v4_decode", _JitFunction(), layout=DeepSeekV4CacheLayout()
    )

    assert compiled == "compiled"
    assert captured["name"] == "deepseek_v4_decode"
    assert captured["use_cache"] is True


def test_deepseek_weight_store_reads_real_safetensors_by_name(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "head.weight": torch.ones(2, 2),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "head.weight": "global.safetensors",
        },
    )

    loaded = store.load_tensor("embed.weight")

    assert loaded.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_deepseek_weight_store_maps_valid_prepacked_sidecar(tmp_path, monkeypatch, caplog):
    from safetensors.torch import save_file

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    params = {
        "ranks": 2,
        "n_routed_experts": 4,
        "compress_ratios": (4,),
        "num_hash_layers": 1,
    }
    fingerprint = store.packed_stacked_layer_weights_fingerprint(**params)
    expected = {
        name: torch.arange(2, dtype=torch.float32).reshape(2, 1)
        for name in weight_loader._DEEPSEEK_V4_PACKED_WEIGHT_NAMES
    }
    save_file(
        expected,
        str(deepseek_v4_packed_weights_path(tmp_path, ranks=2)),
        metadata={
            "format": DEEPSEEK_V4_PACKED_FORMAT,
            "source_fingerprint": fingerprint,
        },
    )
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda path: 1.0)
    monkeypatch.setattr(
        store,
        "load_packed_layer_weights",
        lambda *args, **kwargs: pytest.fail("valid sidecar must skip checkpoint packing"),
    )

    packed = store.load_stacked_layer_weights(**params)

    assert packed.tensors.keys() == expected.keys()
    assert all(torch.equal(packed.tensors[name], tensor) for name, tensor in expected.items())

    shard_path.write_bytes(b"changed-source-checkpoint")
    caplog.set_level("WARNING")
    assert store.load_prepacked_stacked_layer_weights(**params) is None
    assert "Ignoring stale DeepSeekV4 packed weights sidecar" in caplog.text


def test_deepseek_weight_store_ignores_prepacked_sidecar_with_wrong_tensor_names(
    tmp_path,
    monkeypatch,
    caplog,
):
    from safetensors.torch import save_file

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    params = {
        "ranks": 2,
        "n_routed_experts": 4,
        "compress_ratios": (4,),
        "num_hash_layers": 1,
    }
    save_file(
        {"unexpected": torch.zeros((2, 1))},
        str(deepseek_v4_packed_weights_path(tmp_path, ranks=2)),
        metadata={
            "format": DEEPSEEK_V4_PACKED_FORMAT,
            "source_fingerprint": store.packed_stacked_layer_weights_fingerprint(**params),
        },
    )
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda path: 1.0)
    caplog.set_level("WARNING")

    assert store.load_prepacked_stacked_layer_weights(**params) is None
    assert "invalid tensor names" in caplog.text
    assert "unexpected" in caplog.text


def test_deepseek_weight_store_skips_cold_prepacked_sidecar(tmp_path, monkeypatch, caplog):
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    deepseek_v4_packed_weights_path(tmp_path, ranks=2).write_bytes(b"not-opened")
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda path: 0.5)
    caplog.set_level("INFO")

    packed = store.load_prepacked_stacked_layer_weights(
        ranks=2,
        n_routed_experts=4,
        compress_ratios=(4,),
        num_hash_layers=1,
    )

    assert packed is None
    assert "Skipping cold DeepSeekV4 packed weights sidecar" in caplog.text


def test_deepseek_page_cache_probe_open_failure_falls_back(tmp_path, monkeypatch, caplog):
    packed_path = tmp_path / "packed.safetensors"

    def deny_open(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(weight_loader.os, "open", deny_open)
    caplog.set_level("WARNING")

    assert weight_loader._sample_file_page_cache_residency(packed_path) is None
    assert "Could not inspect page-cache residency" in caplog.text


def test_deepseek_prepack_fingerprints_before_packing_and_preserves_shard_mode(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "compress_ratios": [4],
                "n_routed_experts": 4,
                "num_hash_layers": 1,
            }
        )
    )
    shard_path = model_dir / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"checkpoint")
    shard_path.chmod(0o640)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"source.weight": shard_path.name}})
    )
    events: list[str] = []

    class FakeStore:
        def __init__(self, *, model_dir, weight_map):
            pass

        def packed_stacked_layer_weights_fingerprint(self, **kwargs):
            events.append("fingerprint")
            return "source-fingerprint"

        def load_stacked_layer_weights(self, **kwargs):
            assert kwargs["use_prepacked"] is False
            events.append("pack")
            return DeepSeekV4StackedLayerWeights(tensors={"weight": torch.zeros((1, 1))})

    monkeypatch.setattr(prepack_deepseek_v4, "DeepSeekV4WeightStore", FakeStore)
    output = model_dir / "packed.safetensors"

    prepack_deepseek_v4.build_sidecar(
        model_dir,
        ranks=1,
        output=output,
        force=False,
    )

    assert events == ["fingerprint", "pack"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def test_deepseek_executor_lazily_loads_and_caches_embeddings(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"embed.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4)},
        str(tmp_path / "embed.safetensors"),
    )
    open_count = 0
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"embed.weight": "embed.safetensors"},
    )
    original_open = store._safe_open_fn

    def _counting_open(path: Path, device: str):
        nonlocal open_count
        open_count += 1
        return original_open(path, device)

    store._safe_open_fn = _counting_open
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiled = {
        "dsv4": DeepSeekV4CompiledKernels(
            layout=DeepSeekV4CacheLayout(),
            model_dir=str(tmp_path),
            weight_map=store.weight_map,
            weight_store=store,
            compress_ratios=tuple([0] * 44),
            layer_plan=build_deepseek_v4_layer_plan(
                compress_ratios=tuple([0] * 44),
                num_hidden_layers=43,
                num_hash_layers=3,
            ),
            kernel_dir=str(tmp_path),
        )
    }
    executor._embedding_cache = {}
    model = _runtime_model_for_embeddings()

    first = executor.lookup_embeddings(model, torch.tensor([1, 3], dtype=torch.long))
    second = executor.lookup_embeddings(model, torch.tensor([[2, 4]], dtype=torch.long))
    runner = DeepSeekV4ModelRunner(compiled=executor._compiled["dsv4"])
    runner_rows = runner._embedding_rows(torch.tensor([0, 5]), torch.float32)

    assert first.tolist() == [[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]
    assert second.shape == (1, 2, 4)
    assert second[0, 1].tolist() == [16.0, 17.0, 18.0, 19.0]
    assert runner_rows.tolist() == [[0.0, 1.0, 2.0, 3.0], [20.0, 21.0, 22.0, 23.0]]
    assert open_count == 1


def test_deepseek_weight_store_loads_rank_local_experts(tmp_path):
    core_names = deepseek_v4_layer_core_weight_names(0, include_tid2eid=True)
    local_experts = deepseek_v4_local_expert_ids(rank=1, ranks=4, n_routed_experts=8)
    expert_names = deepseek_v4_routed_expert_weight_names(0, local_experts)
    weight_map = {name: "layer.safetensors" for name in (*core_names, *expert_names)}
    (tmp_path / "layer.safetensors").touch()
    reads: list[str] = []

    class _Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append(name)
            return torch.tensor([len(reads)])

    store = DeepSeekV4WeightStore(
        model_dir=tmp_path, weight_map=weight_map, safe_open_fn=lambda path, device: _Reader()
    )

    loaded = store.load_rank_layer_weights(
        0,
        rank=1,
        ranks=4,
        n_routed_experts=8,
        include_tid2eid=True,
    )

    assert local_experts == (2, 3)
    assert set(loaded) == set(weight_map)
    assert all(".experts.2." in name or ".experts.3." in name for name in expert_names)
    assert not any(".experts.0." in name or ".experts.1." in name for name in loaded)


def test_deepseek_weight_store_packs_lm_head_into_8_tp_shards(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "norm.weight": torch.arange(4, dtype=torch.float32),
            "head.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4) + 100,
            "hc_head_fn": torch.zeros((4, 16), dtype=torch.float32),
            "hc_head_scale": torch.ones((1,), dtype=torch.float32),
            "hc_head_base": torch.zeros((4,), dtype=torch.float32),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "norm.weight": "global.safetensors",
            "head.weight": "global.safetensors",
            "hc_head_fn": "global.safetensors",
            "hc_head_scale": "global.safetensors",
            "hc_head_base": "global.safetensors",
        },
    )

    global_weights = store.load_packed_global_weights(ranks=8)

    assert global_weights.lm_head_layout.vocab_per_rank == 2
    assert global_weights.lm_head_layout.padded_vocab_per_rank == 512
    assert global_weights.lm_head_weight.shape == (8, 512, 4)
    assert global_weights.lm_head_weight[0, :2].tolist() == [
        [100.0, 101.0, 102.0, 103.0],
        [104.0, 105.0, 106.0, 107.0],
    ]
    assert global_weights.lm_head_weight[1, :2].tolist() == [
        [108.0, 109.0, 110.0, 111.0],
        [112.0, 113.0, 114.0, 115.0],
    ]
    assert torch.count_nonzero(global_weights.lm_head_weight[:, 2:]) == 0


def test_deepseek_layer_packer_transposes_and_stacks_rank_local_experts():
    raw = _synthetic_layer_raw(layer_id=0, n_experts=4)

    packed = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
    )

    assert packed.tensors["wq_a"].shape == (2, 4, 2)
    assert packed.tensors["wq_a"][0].tolist() == raw["layers.0.attn.wq_a.weight"].t().tolist()
    assert packed.tensors["wo_a"].shape == (2, 8, 2, 4)
    assert packed.tensors["csa_cmp_wkv"].shape == (2, 2, 4)
    assert packed.tensors["csa_cmp_wkv"][0].tolist() == raw["layers.0.attn.compressor.wkv.weight"].tolist()
    assert packed.tensors["csa_inner_wkv"].shape == (2, 2, 4)
    assert (
        packed.tensors["csa_inner_wkv"][0].tolist()
        == raw["layers.0.attn.indexer.compressor.wkv.weight"].tolist()
    )
    assert packed.tensors["hca_cmp_wkv"].shape == (2, 512, 4096)
    assert torch.count_nonzero(packed.tensors["hca_cmp_wkv"]) == 0
    assert packed.tensors["gate_bias"].shape == (2, 4)
    assert packed.tensors["tid2eid"].shape == (2, 129280, 6)
    assert packed.tensors["routed_w1"].shape == (2, 2, 2, 4)
    assert packed.tensors["routed_w1"][0, 0].tolist() == raw["layers.0.ffn.experts.0.w1.weight"].tolist()
    assert packed.tensors["routed_w1"][1, 0].tolist() == raw["layers.0.ffn.experts.2.w1.weight"].tolist()
    assert torch.equal(packed.tensors["csa_hadamard_idx"][0], deepseek_v4_hadamard_idx())

    destination_storage = {
        name: torch.empty(
            (int(tensor.shape[0]), int(tensor.shape[1]) * 2, *tensor.shape[2:]),
            dtype=tensor.dtype,
        )
        for name, tensor in packed.tensors.items()
    }
    destinations = {
        name: storage[:, int(packed.tensors[name].shape[1]) :]
        for name, storage in destination_storage.items()
    }
    assert not all(destination.is_contiguous() for destination in destinations.values())
    direct = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
        destinations=destinations,
    )

    assert direct.tensors.keys() == packed.tensors.keys()
    for name, expected in packed.tensors.items():
        assert direct.tensors[name] is destinations[name]
        assert torch.equal(direct.tensors[name], expected), name


def test_deepseek_stacked_weight_loader_packs_subsequent_layers_into_final_slices(monkeypatch):
    def packed_layer(layer_id: int) -> weight_loader.DeepSeekV4PackedLayerWeights:
        tensors = {"fwd": torch.full((2, 2), layer_id, dtype=torch.int8)}
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES)
            }
        )
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + 12 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES)
            }
        )
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)

    layers = [packed_layer(layer_id) for layer_id in range(3)]
    direct_flags: list[bool] = []
    store = DeepSeekV4WeightStore(model_dir=".", weight_map={})

    def fake_load(layer_id: int, **kwargs):
        destinations = kwargs.get("destinations")
        direct_flags.append(destinations is not None)
        packed = layers[layer_id]
        if destinations is None:
            return packed
        for name, destination in destinations.items():
            destination.copy_(packed.tensors[name])
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=destinations)

    monkeypatch.setattr(store, "load_packed_layer_weights", fake_load)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("torch.cat must not be used"))

    stacked = store.load_stacked_layer_weights(
        ranks=2,
        n_routed_experts=4,
        compress_ratios=(0, 4, 128),
        num_hash_layers=1,
    )

    assert direct_flags == [False, True, True]
    assert stacked.tensors["fwd"].tolist() == [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]
    for name in weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[1].tensors[name])
    for name in weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[2].tensors[name])
    assert all(tensor.is_contiguous() for tensor in stacked.tensors.values())


def test_deepseek_worker_registers_main_and_mtp_weights_for_inheritance(monkeypatch):
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)
    compiled_program = object()
    captured = {}

    class FakeDistributedWorker:
        def __init__(
            self,
            compiled,
            *,
            persistent,
            reset_persistent_windows,
            inherited_host_tensors,
        ):
            captured["compiled"] = compiled
            captured["persistent"] = persistent
            captured["reset_persistent_windows"] = reset_persistent_windows
            captured["inherited"] = inherited_host_tensors

    monkeypatch.setattr("pypto.runtime.DistributedWorker", FakeDistributedWorker)
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._l3_worker = None
    runner._stacked_host_weights = {"main": main_weight}
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._compiled = type(
        "Compiled",
        (),
        {
            "l3_callables": lambda _self: (
                DeepSeekV4L3Callable(
                    compiled_program,
                    "decode",
                ),
            )
        },
    )()
    runner._assert_l3_shared_buffers_preallocated = lambda: None

    worker = runner._shared_l3_worker()

    assert isinstance(worker, FakeDistributedWorker)
    assert captured["compiled"] == [compiled_program]
    assert captured["persistent"] is True
    assert captured["reset_persistent_windows"] is False
    assert captured["inherited"] == [main_weight, mtp_weight]


def test_deepseek_resident_upload_releases_inherited_host_references():
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)

    class FakeWorker:
        def __init__(self):
            self.released = False

        def alloc_stacked_tensor(self, tensor):
            return tensor

        def free_stacked_tensor(self, _tensor):
            pass

        def release_inherited_host_tensor_refs(self):
            self.released = True

    worker = FakeWorker()
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._stacked_host_weights = {"main": main_weight}
    runner._stacked_device_weights = None
    runner._global_weights = None
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._mtp_device_weights = None
    runner._compiled = SimpleNamespace(
        prepacked_layer_weights=DeepSeekV4StackedLayerWeights(tensors={"main": main_weight})
    )
    runner._shared_l3_worker = lambda: worker

    runner._materialize_resident_weights()

    assert worker.released
    assert runner._compiled.prepacked_layer_weights is None
    assert runner._stacked_host_weights is None
    assert not runner._mtp_buffers.weights


def test_deepseek_cache_metadata_maps_scheduler_block_ids():
    metadata = DeepSeekV4CacheMetadataBuilder(layout=DeepSeekV4CacheLayout())

    table = metadata.block_table_from_ids([[64, 65]], max_blocks=4)
    assert table.tolist() == [[64, 65, 0, 0]]

    cmp_mapping = metadata.slot_mapping_from_ids(
        [[64]],
        [[0, 4, 256]],
        block_size=128,
        compress_ratio=4,
    )
    base = 64 * 128
    assert cmp_mapping.tolist() == [[base, base + 1, base + 64]]

    hca_state_mapping = metadata.slot_mapping_from_ids(
        [[64]],
        [[0, 128, 256]],
        block_size=8,
        compress_ratio=128,
    )
    assert hca_state_mapping.tolist() == [[64 * 8, 64 * 8 + 1, 64 * 8 + 2]]


def test_deepseek_cache_group_specs_leave_physical_capacity_for_runtime_sizing():
    compress_ratios = (0, 0, *([4] * 21), *([128] * 20))
    specs = build_deepseek_v4_cache_group_specs(43, compress_ratios, decode_batch=8)
    by_name = {spec.name: spec for spec in specs}

    assert all(spec.num_blocks is None for spec in specs)
    assert by_name["ori"].spec.page_size_bytes == 43 * 128 * 512 * 2
    assert by_name["idx"].spec.page_size_bytes == 21 * 128 * (128 + 4)
    assert by_name["hca_state"].spec.page_size_bytes == 20 * 8 * 1024 * 4
    assert deepseek_v4_cache_blocks_for_slots(specs, 3) == {
        "ori": 96,
        "cmp": 24,
        "idx": 48,
        "hca_state": 48,
        "csa_state": 48,
        "csa_inner_state": 48,
    }


def test_deepseek_cache_sizing_uses_limiting_rank_post_weight_budget(monkeypatch):
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            device_id=2,
            device_ids=(2, 5),
        )
    )
    runner._cache_group_specs = build_deepseek_v4_cache_group_specs(
        43,
        runner._compiled.compress_ratios,
        decode_batch=layout.decode_batch,
    )
    memory = {
        "npu:2": (5_000_000_000, 10_000_000_000),
        "npu:5": (4_000_000_000, 10_000_000_000),
    }
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda device: memory[device])
    runtime = RuntimeConfig(npu_memory_utilization=0.8)

    bytes_per_slot = sum(
        spec.max_blocks_per_seq * spec.spec.page_size_bytes for spec in runner._cache_group_specs
    )
    scratch_bytes = sum(layout.decode_batch * spec.spec.page_size_bytes for spec in runner._cache_group_specs)
    expected = max((2_000_000_000 - scratch_bytes) // bytes_per_slot, 1)

    assert runner._compute_kv_cache_capacity_slots(runtime) == expected


def test_deepseek_cache_allocation_halves_all_groups_together_on_oom(monkeypatch):
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
        )
    )
    runner._cache_group_specs = build_deepseek_v4_cache_group_specs(
        43,
        runner._compiled.compress_ratios,
        decode_batch=layout.decode_batch,
    )
    attempts = []

    def allocate_main_cache():
        slots = runner._cache_group_num_blocks["ori"] // 32
        attempts.append(slots)
        if slots > 2:
            raise MemoryError("synthetic OOM")
        return object()

    monkeypatch.setattr(runner, "_materialize_decode_device_cache", allocate_main_cache)
    monkeypatch.setattr(runner, "_materialize_mtp_device_kv_cache", lambda: None)

    assert runner._alloc_kv_cache_with_retry(8) == 2
    assert attempts == [8, 4, 2]
    assert runner._cache_group_num_blocks == deepseek_v4_cache_blocks_for_slots(
        runner._cache_group_specs,
        2,
    )


def test_deepseek_device_cache_allocates_runtime_sized_rank_shards():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        block_size=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
        )
    )
    runner._cache_group_num_blocks = {
        name: 2 for name in ("ori", "cmp", "idx", "hca_state", "csa_state", "csa_inner_state")
    }

    class FakeWorker:
        def __init__(self):
            self.allocations = []
            self.frees = []

        def alloc_tensor(self, shape, dtype, *, worker_id=0):
            tensor = DeviceTensor(
                0x1000 + len(self.allocations) * 0x100,
                tuple(shape),
                dtype,
            )
            self.allocations.append((worker_id, tensor))
            return tensor

        def free_tensor(self, tensor, *, worker_id=0):
            self.frees.append((worker_id, tensor))

        def free_stacked_tensor(self, stacked):
            for tensor, worker_id in zip(
                stacked.shards,
                stacked.worker_ids,
                strict=True,
            ):
                self.free_tensor(tensor, worker_id=worker_id)

    worker = FakeWorker()
    runner._l3_worker = worker

    cache = runner._materialize_decode_device_cache()

    assert cache.kv_cache.full_shape == (2, 43 * 3, 1, 1, 512)
    assert cache.cmp_kv.full_shape == (2, 43 * 3, 1, 1, 512)
    assert cache.idx_kv_cache.full_shape == (2, 21 * 3, 1, 1, 128)
    assert cache.hca_compress_state.full_shape == (2, 20 * 3, 8, 1024)
    assert len(worker.allocations) == 14
    assert {worker_id for worker_id, _tensor in worker.allocations} == {0, 1}

    runner._free_device_caches()
    assert len(worker.frees) == 14


def _grouped_cache_rows(count: int) -> list[dict[str, list[int]]]:
    names = ("ori", "cmp", "idx", "hca_state", "csa_state", "csa_inner_state")
    return [
        {name: [request_index * len(names) + group_index] for group_index, name in enumerate(names)}
        for request_index in range(count)
    ]


def test_deepseek_prepare_prefill_inputs_maps_chunk_metadata():
    runner, model = _runner_for_prepared_inputs()
    layout = runner._compiled.layout
    embeddings = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

    prepared = runner.prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([10, 11, 12], dtype=torch.long),
            input_embeddings=embeddings,
            seq_lens=[129],
            chunk_lens=[3],
            chunk_offsets=[0],
            chunk_starts=[126],
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    assert prepared.request_ids == ("req-a",)
    assert prepared.ranks == (0,)
    assert prepared.actual_tokens == (3,)
    assert prepared.x_hc.shape == (8, 128, 4, 4)
    assert prepared.x_hc.dtype == torch.float32
    assert prepared.ori_block_table.shape == (8, 128)
    assert prepared.ori_block_table[0, :4].tolist() == [0, 0, 0, 0]
    assert prepared.cmp_block_table.shape == (8, 32)
    assert prepared.idx_block_table.shape == (8, 64)
    assert prepared.position_ids.shape == (8, 128)
    assert prepared.position_ids[0, :4].tolist() == [126, 127, 128, 129]
    assert prepared.input_ids[0, :4].tolist() == [10, 11, 12, 10]
    assert prepared.ori_slot_mapping.shape == (8, 128)
    assert prepared.ori_slot_mapping[0, :4].tolist() == [126, 127, 0, -1]
    assert prepared.hca_cmp_slot_mapping.shape == (8, 128)
    assert prepared.hca_cmp_slot_mapping[0, :3].tolist() == [-1, 128, -1]
    assert prepared.hca_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_cmp_slot_mapping.shape == (8, 128)
    assert prepared.csa_cmp_slot_mapping[0, :3].tolist() == [-1, 159, -1]
    assert prepared.csa_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_idx_slot_mapping.shape == (8, 128)
    assert prepared.csa_idx_slot_mapping[0, :3].tolist() == [-1, 287, -1]
    assert prepared.csa_idx_slot_mapping[0, 3].item() == -1
    assert prepared.hca_state_slot_mapping.shape == (8, 128)
    assert prepared.hca_state_slot_mapping[0, :4].tolist() == [
        30,
        31,
        24,
        -1,
    ]
    assert prepared.csa_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_state_slot_mapping[0, :4].tolist() == [
        18,
        19,
        16,
        -1,
    ]
    assert prepared.csa_inner_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_inner_state_slot_mapping[0, :4].tolist() == [
        22,
        23,
        20,
        -1,
    ]
    assert prepared.num_tokens_per_owner.tolist() == [3, 0, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [2, -1, -1, -1, -1, -1, -1, -1]


def test_deepseek_prepare_decode_inputs_accepts_device_embedding_batch():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )
    assert prepared.x_hc is None
    assert prepared.input_ids[0, :2].tolist() == [5, 5]


def test_deepseek_prepare_mtp_target_inputs_limits_partial_chunk_rows():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_mtp_target_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([129], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
        token_rows=torch.tensor([[5, 5]], dtype=torch.long),
        positions=((128, 128),),
        active_width=1,
    )

    assert prepared.input_ids[0, :2].tolist() == [5, 5]
    assert prepared.position_ids[0, :2].tolist() == [128, 128]
    assert prepared.num_tokens_per_owner.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [0, -1, -1, -1, -1, -1, -1, -1]


def test_deepseek_stage_decode_inputs_uses_shared_buffers():
    runner, model = _runner_for_prepared_inputs()
    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    staged = runner._stage_decode_inputs(prepared)

    assert staged.x_hc is None
    assert runner._decode_buffers is not None
    for name in (
        "input_ids",
        "position_ids",
        "kv_seq_lens",
        "block_table",
        "cmp_block_table",
        "idx_block_table",
        "hca_compress_state_block_table",
        "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "block_counts",
    ):
        assert getattr(staged, name).is_shared()


def test_deepseek_prepare_decode_inputs_reuses_static_metadata():
    runner, model = _runner_for_prepared_inputs()
    original_metadata = runner.cache_metadata

    class CountingMetadata:
        def __init__(self):
            self.ring_table_calls = 0

        def ring_block_table_from_ids(self, *args, **kwargs):
            self.ring_table_calls += 1
            return original_metadata.ring_block_table_from_ids(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(original_metadata, name)

    counting_metadata = CountingMetadata()
    runner.cache_metadata = counting_metadata

    def prepare(seq_len, grouped_rows):
        return runner.prepare_decode_inputs(
            model,
            DecodeBatch(
                request_ids=["req-a"],
                token_ids=torch.tensor([[5]], dtype=torch.long),
                hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                block_ids_by_group=grouped_rows,
                cache_partitions=[0],
            ),
        )

    first = prepare(128, _grouped_cache_rows(1))
    first_ring_table_calls = counting_metadata.ring_table_calls
    assert first_ring_table_calls == 3 * runner._compiled.layout.ranks
    assert first.block_table.is_shared()

    second = prepare(129, _grouped_cache_rows(1))
    assert counting_metadata.ring_table_calls == first_ring_table_calls
    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert second.position_ids[0, 0].item() == 128

    changed_rows = _grouped_cache_rows(1)
    changed_rows[0]["ori"] = [10]
    prepare(130, changed_rows)
    assert counting_metadata.ring_table_calls == first_ring_table_calls + 3


def test_deepseek_run_decode_dispatches_active_token_count():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode")
    captured: dict[str, object] = {}

    def fake_stage(inputs):
        captured["prepared"] = inputs
        return inputs

    def fake_decode_fwd_args(inputs, pre_hc_hidden_out, hidden_out, logits, sampled_ids):
        captured["num_tokens_per_owner"] = inputs.num_tokens_per_owner
        return (pre_hc_hidden_out, hidden_out, logits, sampled_ids)

    def fake_run_l3(_callable, *args):
        args[-2].fill_(1)

    hidden_out = torch.empty(
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        model.config.hidden_size,
        dtype=torch.bfloat16,
    )
    runner._ensure_l3_shared_buffers = lambda _model: None
    runner._stage_decode_inputs = fake_stage
    runner._require_decode_buffers = lambda: SimpleNamespace(
        sampled_ids=torch.empty(
            runner._compiled.layout.ranks,
            runner._compiled.layout.decode_tokens,
            8,
            dtype=torch.int32,
        ),
    )
    runner._materialize_main_pre_hc_device = lambda _hidden_size: torch.empty(
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        runner._compiled.layout.hc_mult,
        model.config.hidden_size,
        dtype=torch.float32,
    )
    runner._require_decode_output_buffer = lambda _hidden_size: hidden_out
    runner._require_decode_logits_buffer = lambda _vocab_size: torch.empty(
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        model.config.vocab_size,
        dtype=torch.float32,
    )
    runner._decode_fwd_args = fake_decode_fwd_args
    runner._run_l3 = fake_run_l3

    result = runner.run_decode(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    assert captured["num_tokens_per_owner"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert result.logits.shape == (1, model.config.vocab_size)


def test_deepseek_mtp_decode_fuses_main_verify_and_draft_into_one_dispatch():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 1
    runner._compiled.layout = deepseek_v4_decode_layout(1)
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode_mtp_fused")
    runner._decode_flow = runner._run_mtp_decode
    layout = runner._compiled.layout
    main_sampled_ids = torch.zeros(
        layout.ranks,
        layout.decode_tokens,
        8,
        dtype=torch.int32,
    )
    mtp_buffers = SimpleNamespace(
        decode_accepted_counts=torch.ones(
            layout.ranks,
            layout.decode_batch,
            dtype=torch.int32,
        ),
        decode_input_ids=torch.zeros(
            layout.ranks,
            layout.decode_tokens,
            dtype=torch.long,
        ),
        decode_position_ids=torch.zeros(
            layout.ranks,
            layout.decode_tokens,
            dtype=torch.int32,
        ),
        decode_sampled_ids=torch.zeros(
            layout.ranks,
            layout.decode_batch,
            8,
            dtype=torch.int32,
        ),
    )
    state = SimpleNamespace(
        draft_token_id=5,
        tail_token_id=3,
        tail_slot_id=0,
        tail_position=126,
        proposed_tokens=0,
        accepted_tokens=0,
        committed_count=0,
    )
    runner._mtp_request_states["req-a"] = state
    staged = SimpleNamespace(
        request_ids=("req-a",),
        ranks=(0,),
        local_rows=(0,),
        actual_batch=1,
        per_rank_counts=(1,) + (0,) * (layout.ranks - 1),
    )
    dispatches = []
    prepared_rows = []

    runner._ensure_l3_shared_buffers = lambda _model: None

    def fake_prepare(_model, speculative_batch, *, token_rows, positions, active_width):
        prepared_rows.append(
            (
                speculative_batch.seq_lens.tolist(),
                token_rows.tolist(),
                positions,
                active_width,
            )
        )
        return staged

    runner.prepare_mtp_target_inputs = fake_prepare
    runner._stage_decode_inputs = lambda prepared: prepared
    runner._stage_fused_mtp_metadata = lambda _inputs: layout.decode_seq
    runner._require_decode_buffers = lambda: SimpleNamespace(sampled_ids=main_sampled_ids)
    runner._require_mtp_buffers = lambda: mtp_buffers
    runner._require_decode_output_buffer = lambda _hidden_size: torch.empty(0)
    runner._materialize_main_pre_hc_device = lambda _hidden_size: torch.empty(0)
    runner._require_decode_logits_buffer = lambda _vocab_size: torch.empty(0)
    runner._decode_fwd_args = lambda *_args: ()
    runner._fused_mtp_decode_args = lambda _main_args, _active_tokens: ("fused",)

    def fake_run_l3(callable_spec, *args):
        dispatches.append((callable_spec.name, args))
        main_sampled_ids[0, 0, 0] = 5
        main_sampled_ids[0, 1, 0] = 9
        mtp_buffers.decode_accepted_counts[0, 0] = 2
        mtp_buffers.decode_input_ids[0, 1] = 9
        mtp_buffers.decode_position_ids[0, 1] = 128
        mtp_buffers.decode_sampled_ids[0, 0, 0] = 7

    runner._run_l3 = fake_run_l3

    result = runner.run_decode(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[3]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([128], dtype=torch.int32),
            allow_device_greedy_sampling=True,
        ),
    )

    assert dispatches == [("decode_mtp_fused", ("fused",))]
    assert prepared_rows == [([129], [[3, 5]], ((127, 128),), 2)]
    assert result.accepted_token_ids == [[5, 9]]
    assert state.draft_token_id == 7
    assert state.tail_token_id == 9
    assert state.tail_position == 128
    assert state.proposed_tokens == 1
    assert state.accepted_tokens == 1
    assert state.committed_count == 2


def test_deepseek_prefill_staging_keeps_worker_resident_cache_tensors_out():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        block_size=1,
        prefill_ori_max_blocks=1,
        decode_ori_max_blocks=1,
        ori_table_max_blocks=1,
        cmp_max_blocks=1,
        idx_max_blocks=1,
        hca_state_max_blocks=1,
        csa_state_max_blocks=1,
        csa_inner_state_max_blocks=1,
        c128_state_block_size=1,
        c4_state_block_size=1,
        prefill_cmp_max_blocks=1,
        prefill_idx_max_blocks=1,
        prefill_hca_state_max_blocks=1,
        prefill_csa_state_max_blocks=1,
        prefill_csa_inner_state_max_blocks=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            freqs_cos=torch.empty((1, 1), dtype=torch.bfloat16),
            freqs_sin=torch.empty((1, 1), dtype=torch.bfloat16),
        )
    )
    prefill = runner._ensure_prefill_fwd_buffers(hidden_size=1)

    assert not hasattr(runner, "_decode_work_cache")
    for name in (
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
        "hca_compress_state",
        "csa_compress_state",
        "csa_inner_compress_state",
    ):
        assert name not in prefill.tensors


def test_deepseek_mtp_prefill_and_decode_reuse_same_kv_cache():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        block_size=1,
        prefill_ori_max_blocks=1,
        decode_ori_max_blocks=1,
        sliding_window=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            mtp_prefill=DeepSeekV4L3Callable(compiled=object(), name="mtp_prefill"),
            mtp_decode=DeepSeekV4L3Callable(compiled=object(), name="mtp_decode"),
        )
    )
    weight = torch.arange(2, dtype=torch.float32)
    runner.load_mtp_weights = lambda: weight_loader.DeepSeekV4MtpWeights(tensors={"weight": weight})

    buffers = runner._ensure_mtp_buffers(hidden_size=1)

    assert buffers is not None
    assert buffers.weights["weight"] is weight
    assert not buffers.weights["weight"].is_shared()
    assert buffers.prefill_kv_cache is buffers.decode_kv_cache


def test_deepseek_mtp_prefill_outputs_allocate_empty_rank_shards_once():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        prefill_seq=3,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = []

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(
                0x1000 + len(self.allocations) * 0x100000,
                tuple(shape),
                dtype,
            )
            self.allocations.append((tuple(shape), dtype, init, worker_id))
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

    worker = FakeWorker()
    runner._l3_worker = worker

    first = runner._materialize_mtp_prefill_device_outputs(hidden_size=5)
    second = runner._materialize_mtp_prefill_device_outputs(hidden_size=5)

    assert second is first
    assert first.hidden_out.full_shape == (2, 3, 5)
    assert first.pre_hc_hidden_out.full_shape == (2, 3, 4, 5)
    assert first.logits.full_shape == (2, 8, 129280)
    assert all(shard.dtype == torch.bfloat16 for shard in first.hidden_out.shards)
    assert all(shard.dtype == torch.float32 for shard in first.pre_hc_hidden_out.shards)
    assert all(shard.dtype == torch.float32 for shard in first.logits.shards)
    assert len(worker.allocations) == 6
    assert all(init is None for _shape, _dtype, init, _worker_id in worker.allocations)
    assert [worker_id for _shape, _dtype, _init, worker_id in worker.allocations] == [0, 1] * 3


def test_deepseek_mtp_prefill_reads_only_owner_logits_row():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = []
            self.copies = []

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(
                0x10000000 + len(self.allocations) * 0x1000000,
                tuple(shape),
                dtype,
            )
            self.allocations.append((tensor, init, worker_id))
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

        def copy_from(self, dst, src, nbytes, *, worker_id=0):
            self.copies.append((dst, src, nbytes, worker_id))

    worker = FakeWorker()
    runner._l3_worker = worker
    runner._mtp_buffers = SimpleNamespace(
        prefill_hidden_in=torch.empty((layout.ranks, layout.prefill_seq, 5), dtype=torch.bfloat16),
        prefill_logits=torch.empty(
            (layout.ranks, 8, 129280),
            dtype=torch.float32,
        ).share_memory_(),
    )
    outputs = runner._materialize_mtp_prefill_device_outputs(hidden_size=5)

    host_row = runner._read_mtp_prefill_logits(owner_rank=1)

    assert host_row.data_ptr() == runner._mtp_buffers.prefill_logits[1, 0].data_ptr()
    assert worker.copies == [
        (
            host_row.data_ptr(),
            outputs.logits.shards[1].data_ptr,
            129280 * torch.float32.itemsize,
            1,
        )
    ]


def test_deepseek_mtp_prefill_args_use_device_outputs():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = 0

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(0x1000 + self.allocations * 0x100000, tuple(shape), dtype)
            self.allocations += 1
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

    runner._l3_worker = FakeWorker()
    outputs = runner._materialize_mtp_prefill_device_outputs(hidden_size=1)
    value = object()
    runner._mtp_buffers = SimpleNamespace(
        weights={},
        prefill_hidden_in=torch.empty((1, 1, 1), dtype=torch.bfloat16),
        prefill_prev_hidden_in=value,
        prefill_block_table=value,
        prefill_slot_mapping=value,
        prefill_position_ids=value,
        prefill_input_ids=value,
        prefill_logit_row_indices=value,
    )
    runner._mtp_device_weights = {}
    runner._materialize_mtp_device_kv_cache = lambda: value
    runner._static_freqs_cos_tensor = lambda: value
    runner._static_freqs_sin_tensor = lambda: value
    runner._static_lm_head_weight_tensor = lambda: value
    runner._mark_resident_args = lambda values, _policy: values
    runner._ordered_layer_args = lambda values, _order: values

    named = runner._mtp_prefill_args()

    assert named["hidden_out"] is outputs.hidden_out
    assert named["pre_hc_hidden_out"] is outputs.pre_hc_hidden_out
    assert named["logits"] is outputs.logits


def test_deepseek_release_finished_requests_discards_mtp_state():
    runner, _model = _runner_for_prepared_inputs()
    runner._mtp_request_states = {
        "req-a": SimpleNamespace(proposed_tokens=0, tail_rank=1, tail_slot_id=2),
        "req-b": SimpleNamespace(proposed_tokens=0, tail_rank=None, tail_slot_id=None),
    }
    runner._mtp_free_tail_slots = [[] for _ in range(runner._compiled.layout.ranks)]

    runner.release_finished_requests(["req-a"])

    assert runner._mtp_request_states == {
        "req-b": SimpleNamespace(proposed_tokens=0, tail_rank=None, tail_slot_id=None),
    }
    assert runner._mtp_free_tail_slots[1] == [2]


def _write_deepseek_model_dir(tmp_path: Path, *, quant_method: str = "compressed-tensors") -> Path:
    model_dir = tmp_path / "dsv4-flash-w8a8"
    model_dir.mkdir()
    compress_ratios = _deepseek_flash_compress_ratios()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "torch_dtype": "bfloat16",
        "compress_ratios": compress_ratios,
        "quantization_config": {
            "quant_method": quant_method,
            "format": "int-quantized",
            "quantization_status": "compressed",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_names = deepseek_v4_startup_weight_names(
        43,
        n_routed_experts=256,
        compress_ratios=compress_ratios,
        num_hash_layers=3,
    )
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weight_names}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return model_dir


def _deepseek_flash_compress_ratios() -> list[int]:
    return [0, 0, *(4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)), 0]


def _write_deepseek_kernel_dir(
    tmp_path: Path,
    *,
    lm_head_tp_size: int,
    use_config_constant: bool = False,
    block_size: int = 128,
    hca_state_blocks: int = 2048,
    csa_state_blocks: int = 4096,
    csa_inner_state_blocks: int = 4096,
) -> Path:
    kernel_dir = tmp_path / f"deepseek-v4-kernels-tp{lm_head_tp_size}"
    kernel_dir.mkdir()
    (kernel_dir / "prefill_hca.py").write_text(
        "\n".join(
            [
                "HCA_STATE_BLOCK_NUM = 64",
                f"HCA_STATE_MAX_BLOCKS = {hca_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_csa.py").write_text(
        "\n".join(
            [
                "CSA_STATE_BLOCK_NUM = 65",
                f"CSA_STATE_MAX_BLOCKS = {csa_state_blocks}",
                "INNER_STATE_BLOCK_NUM = 65",
                f"INNER_STATE_MAX_BLOCKS = {csa_inner_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_layer.py").write_text("")
    (kernel_dir / "prefill_fwd.py").write_text("")
    (kernel_dir / "prefill_mtp.py").write_text("")
    (kernel_dir / "decode_layer.py").write_text("")
    (kernel_dir / "decode_fwd.py").write_text("")
    (kernel_dir / "decode_fwd_mtp.py").write_text("")
    (kernel_dir / "decode_mtp.py").write_text("")
    (kernel_dir / "config.py").write_text(
        "\n".join(
            [
                f"BLOCK_SIZE = {block_size}",
                "DECODE_BATCH = 4",
                "DECODE_SEQ = 2",
                "DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ",
                "PREFILL_BATCH = 1",
                "PREFILL_SEQ = 128",
                "KV_ORI_MAX_BLOCKS = 128",
                "KV_ORI_TABLE_MAX_BLOCKS = 128",
                "KV_CMP_MAX_BLOCKS = 32",
                "IDX_CACHE_MAX_BLOCKS = 64",
                "ORI_KV_BLOCK_NUM = 128",
                "HCA_STATE_PHYSICAL_BLOCKS = 64",
                "CSA_STATE_PHYSICAL_BLOCKS = 65",
                "CSA_INNER_STATE_PHYSICAL_BLOCKS = 65",
                "PREFILL_ORI_MAX_BLOCKS = 128",
                "PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS",
                "PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS",
                "EP_WORLD_SIZE = 8",
                f"LM_HEAD_TP_SIZE = {lm_head_tp_size}",
                "",
            ]
        )
    )
    if use_config_constant:
        (kernel_dir / "lm_head.py").write_text("TP_SIZE = LM_HEAD_TP_SIZE\n")
    else:
        (kernel_dir / "lm_head.py").write_text(f"TP_SIZE = {lm_head_tp_size}\n")
    return kernel_dir


def _synthetic_layer_raw(*, layer_id: int, n_experts: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer_id}"
    raw = {
        f"{prefix}.hc_attn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_attn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_attn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.attn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.attn.wq_a.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.wkv.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        f"{prefix}.attn.q_norm.weight": torch.arange(2, dtype=torch.bfloat16),
        f"{prefix}.attn.kv_norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.attn_sink": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.wo_a.weight": torch.arange(64, dtype=torch.bfloat16).reshape(16, 4),
        f"{prefix}.attn.wo_b.weight": torch.arange(64, dtype=torch.int8).reshape(4, 16),
        f"{prefix}.attn.wo_b.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.hc_ffn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_ffn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_ffn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.ffn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.ffn.gate.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        f"{prefix}.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w1.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w1.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w2.weight": torch.arange(8, dtype=torch.int8).reshape(4, 2),
        f"{prefix}.ffn.shared_experts.w2.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w3.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w3.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.compressor.norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.indexer.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.indexer.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.indexer.compressor.norm.weight": torch.arange(2, dtype=torch.bfloat16),
    }
    for expert_id in range(n_experts):
        base = expert_id * 10
        raw.update(
            {
                f"{prefix}.ffn.experts.{expert_id}.w1.weight": torch.full((2, 4), base, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w1.scale": torch.full((2,), base + 1, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w2.weight": torch.full((4, 2), base + 2, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w2.scale": torch.full((4,), base + 3, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w3.weight": torch.full((2, 4), base + 4, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w3.scale": torch.full((2,), base + 5, dtype=torch.float32),
            }
        )
    return raw


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = None

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return ""


def _runtime_model_for_embeddings():
    from pypto_serving.config.types import ModelConfig, RuntimeModel

    config = ModelConfig(
        model_id="dsv4",
        architecture="DeepseekV4ForCausalLM",
        vocab_size=6,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(page_size=128, max_batch_size=1, max_seq_len=260, weight_dtype="int8")
    placeholder = torch.empty(0, config.hidden_size)
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=placeholder,
        final_norm_weight=torch.empty(0),
        lm_head=placeholder,
        layers=[],
    )


def _runner_for_prepared_inputs() -> tuple[DeepSeekV4ModelRunner, object]:
    model = _runtime_model_for_embeddings()
    compiled = DeepSeekV4CompiledKernels(
        # Exercise the production MTP tile while keeping the tiny host-side model.
        layout=DeepSeekV4CacheLayout(decode_batch=4, decode_seq=2, decode_tokens=8),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=tuple([0] * 44),
        layer_plan=build_deepseek_v4_layer_plan(
            compress_ratios=tuple([0] * 44),
            num_hidden_layers=43,
            num_hash_layers=3,
        ),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner.init_kv_cache("dsv4", model.config, model.runtime)
    return runner, model


def test_deepseek_static_lm_head_weight_replicates_one_vocab_shard_per_dp_rank():
    layout = DeepSeekV4CacheLayout()
    tp_size = DEEPSEEK_V4_LM_HEAD_TP_SIZE
    # The regression only shows up when the DP world spans more than one TP group.
    assert layout.ranks > tp_size
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    vocab_per_rank = 2
    hidden_size = 3
    # Shard s carries the constant s + 1 so every rank's copy is identifiable.
    packed = torch.stack(
        [
            torch.full((vocab_per_rank, hidden_size), float(shard + 1), dtype=torch.bfloat16)
            for shard in range(tp_size)
        ]
    )
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.empty(0),
        lm_head_weight=packed,
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=tp_size,
            vocab_size=tp_size * vocab_per_rank,
            hidden_size=hidden_size,
            vocab_per_rank=vocab_per_rank,
            padded_vocab_per_rank=vocab_per_rank,
        ),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
    )

    replicated = runner._static_lm_head_weight_tensor()

    # Every card holds a full shard, not just the first TP group: ranks 4..7 must
    # repeat shards 0..3 instead of reading whatever the kernel maps past rank 3.
    assert replicated.shape == (layout.ranks, vocab_per_rank, hidden_size)
    for rank in range(layout.ranks):
        assert torch.equal(replicated[rank], packed[rank % tp_size]), f"rank {rank} shard mismatch"
    assert [float(replicated[rank, 0, 0]) for rank in range(layout.ranks)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
