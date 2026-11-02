# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the production DSpark payload builder without compiling a device program."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pypto_serving.model.deepseek_v41 import npu_runner
from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.deepseek_v41.dspark_resources import (
    DSparkResourceLayout,
    V41DSparkLoweringProvider,
    _DRAFT_MODEL_KEYS,
    _MARKOV_MODEL_KEYS,
    allocate_dspark_host_resources,
)
from pypto_serving.model.deepseek_v41.dspark_runtime import (
    DECODE_STAGE_ORDER,
    PREFILL_STAGE_ORDER,
    V41DSparkRuntime,
)
from pypto_serving.model.deepseek_v41.npu_runner import (
    ResidentDSparkBuffers,
    ResidentDSparkCache,
    SegmentTopology,
    V41ModelRunner,
    V41ResidentResources,
)
from pypto_serving.model.deepseek_v41.task_args import V41CacheGroups


@pytest.fixture
def production(monkeypatch):
    host = allocate_dspark_host_resources(DSparkResourceLayout(
        dp_size=4, max_seq_len=256, vocab_size=8, max_prefill_tokens=32))
    resident = V41ResidentResources.__new__(V41ResidentResources)
    resident.dspark_host_resources = host
    resident.dspark = V41DSparkRuntime(resident, max_seq_len=256)
    resident.ledger = SimpleNamespace(owners={})
    resident._dspark_generations = {}
    resident._dspark_drafter_epochs = {}
    resident._decode_fwd = None
    resident.dspark_embedding = SimpleNamespace(tensor=object())
    resident.dspark_cache = SimpleNamespace(window_cache=object(), window_cache_scale=object())
    packed = {name: object() for name in _DRAFT_MODEL_KEYS | _MARKOV_MODEL_KEYS}
    resident.dspark_weights = SimpleNamespace(tensors=packed, full=packed.__getitem__)
    resident.final = SimpleNamespace(norm_weight=object(), head_weight=object())
    resident.prefill_host = SimpleNamespace(fields={
        "dspark_target_hidden": torch.ones(4, 32, 15360, dtype=torch.bfloat16),
    })
    resident.decode_host = SimpleNamespace(
        topology=SegmentTopology.for_decode(dp=4),
        fields={"engram_lookup": torch.zeros(4, 2, 80, 1, dtype=torch.bfloat16)},
    )
    resident.cache = SimpleNamespace(
        plan=SimpleNamespace(family_groups=V41CacheGroups("window", "c2a", "c1a")),
        reset_request=lambda owner: None,
    )
    engram_steps = []

    def stage_lookup(target, step, topology):
        engram_steps.append(step)
        target.zero_()
        offsets = [0] * topology.dp
        for request in step.requests:
            start = offsets[request.partition]
            count = len(request.token_ids)
            target[request.partition, :, start:start + count, 0] = torch.tensor(request.token_ids)
            offsets[request.partition] += count

    resident.engram_host = SimpleNamespace(stage_lookup=stage_lookup)
    arguments = {
        "final_norm_weight": resident.final.norm_weight,
        "lm_head_weight": resident.final.head_weight,
        "logit_row_indices": torch.full((4, 16), 77, dtype=torch.int32),
        "logits": torch.ones(4, 16, 8),
        "sampled_ids": torch.ones(4, 16, 8, dtype=torch.int32),
        "attention_num_tokens": torch.full((4,), 99, dtype=torch.int32),
        "moe_num_tokens": torch.full((4,), 99, dtype=torch.int32),
        "state_block_table": object(), "c1a_index_block_table": object(),
        "c2a_index_block_table": object(), "engram_lookup": resident.decode_host.fields["engram_lookup"],
    }
    activations = []

    def activate():
        activations.append(True)
        resident._decode_fwd = SimpleNamespace(arguments=arguments)

    monkeypatch.setattr(resident, "_activate_decode", activate)
    buffers = ResidentDSparkBuffers.__new__(ResidentDSparkBuffers)
    buffers.host = host
    buffers.state = host.state
    buffers._copy = lambda source, destination: None
    resident.dspark_buffers = buffers
    provider = V41DSparkLoweringProvider(
        host, payload_builder=resident._dspark_payload,
        state_resolver=resident._resolve_dspark_state,
        target_param_names=(*arguments, "x_hc", "pre_mix"),
        draft_param_names=tuple(_DRAFT_MODEL_KEYS),
        markov_param_names=tuple(_MARKOV_MODEL_KEYS),
    )
    return SimpleNamespace(resident=resident, host=host, provider=provider,
                           activations=activations, arguments=arguments, engram_steps=engram_steps)


def _batch(ids=("a",), ranks=(1,)):
    return SimpleNamespace(
        request_ids=ids, cache_partitions=ranks, token_ids=[11 + index for index in range(len(ids))],
        seq_lens=[33] * len(ids), prompt_lens=[32] * len(ids),
        # These are valid target-pool IDs, deliberately outside the draft pool.
        block_ids_by_group=[{"window": (100 + 2 * index, 101 + 2 * index),
                             "c2a": (200 + index,), "c1a": (300 + 2 * index, 301 + 2 * index)}
                            for index in range(len(ids))],
    )


def _reserve(production, request_id, rank, slot, *, bootstrapped=True):
    production.resident.ledger.owners[request_id] = SimpleNamespace(partition=rank, slot=slot)
    if bootstrapped:
        production.resident.dspark.record_bootstrap(
            request_id, slot=slot, generation=1, prompt_length=32, anchor=32,
            draft_ids=(21, 22, 23, 24, 25))


def test_production_decode_binds_six_rows_counts_outputs_and_current_tables(production):
    for request_id, rank, slot in (("a", 1, 7), ("b", 1, 2), ("c", 3, 4)):
        _reserve(production, request_id, rank, slot)
    batch = _batch(("a", "b", "c"), (1, 1, 3))
    lowered = production.provider.lower_decode(model=object(), batch=batch).lowered
    target, work = lowered.target_model, lowered.workspace

    assert production.activations == [True]
    assert target["attention_num_tokens"].tolist() == [0, 12, 0, 6]
    assert target["moe_num_tokens"].tolist() == [0, 12, 0, 6]
    assert target["logit_row_indices"][1].tolist() == [5, 11] + [-1] * 14
    assert target["logit_row_indices"][0].tolist() == [-1] * 16
    assert target["state_block_table"][1, :2, 0].tolist() == [7, 2]
    assert target["state_block_table"].shape == (4, 12, 1)
    assert target["c2a_index_block_table"] is lowered.cache_tables["target_c2a"]
    assert target["c1a_index_block_table"] is lowered.cache_tables["target_c1a"]
    assert target["logits"] is production.arguments["logits"]
    assert target["sampled_ids"] is production.arguments["sampled_ids"]
    assert target["final_norm_weight"] is production.resident.final.norm_weight
    assert lowered.markov_model["final_norm_weight"] is production.resident.dspark_weights.full(
        "final_norm_weight")
    assert lowered.markov_model["final_norm_weight"] is not target["final_norm_weight"]
    assert lowered.markov_model["lm_head_weight"] is production.resident.final.head_weight
    assert torch.count_nonzero(target["logits"]) == 0
    assert bool((target["sampled_ids"] == -1).all())
    assert work["x_hc"].shape[1] == 80
    assert work["target_hidden"].shape[1] == 72
    assert work["draft_head_hidden"].shape[1:3] == (12, 5)
    assert work["target_positions"][1, 0].tolist() == [32, 33, 34, 35, 36, 37]
    assert [request.start for request in production.engram_steps[-1].requests] == [32, 32, 32]
    assert [request.token_ids for request in production.engram_steps[-1].requests] == [
        (11, 21, 22, 23, 24, 25), (12, 21, 22, 23, 24, 25), (13, 21, 22, 23, 24, 25)]
    assert target["engram_lookup"][1, 0, :12, 0].tolist() == [11, 21, 22, 23, 24, 25,
                                                            12, 21, 22, 23, 24, 25]


def test_draft_pages_follow_stable_slots_across_batch_reordering(production):
    _reserve(production, "a", 1, 7)
    _reserve(production, "b", 1, 2)
    first = production.provider.lower_decode(model=object(), batch=_batch(("a", "b"), (1, 1))).lowered
    draft = first.cache_tables["draft"]
    assert draft[1, 0, :2].tolist() == [[14, 15], [4, 5]]
    assert first.cache_tables["target_window"][1, 0].tolist() == [100, 101]
    assert int(draft.max()) < 12 * 2
    second = production.provider.lower_decode(model=object(), batch=_batch(("b",), (1,))).lowered
    assert second.cache_tables["draft"][1, 0, 0].tolist() == [4, 5]
    assert bool((second.cache_tables["draft"][1, :, 1:] == -1).all())


def _selected_lib_orchestrators(production):
    root = Path(os.environ.get("PYPTO_LIB_ROOT", Path(__file__).resolve().parents[4] / "pypto-lib"))
    model = root / "models" / "deepseek_v4_1_flash"
    if not (model / "decode_fwd_dspark.py").is_file():
        pytest.skip("selected pypto-lib source is unavailable")
    functions = {}
    for filename in ("decode_fwd", "decode_fwd_dspark", "dspark_state", "dspark_target_sample",
                     "dspark_drafter", "dspark_markov"):
        source = ast.parse((model / (filename + ".py")).read_text(encoding="utf-8-sig"))
        functions.update({node.name: node for node in source.body if isinstance(node, ast.FunctionDef)})
    namespace = {"CHUNKS": 5}
    for name, function in functions.items():
        if name.startswith("l3_"):
            namespace[name] = SimpleNamespace(
                param_names=tuple(arg.arg for arg in function.args.args),
                _func=SimpleNamespace(__name__=name))
    target_names = namespace["l3_decode_fwd"].param_names
    for name in target_names:
        production.arguments.setdefault(name, object())
    production.provider._target_param_names = frozenset(target_names)
    production.provider._draft_param_names = frozenset(namespace["l3_dspark_drafter"].param_names)
    production.provider._markov_param_names = frozenset(namespace["l3_markov_sample_from_hidden"].param_names)
    # Execute only the unmodified, dependency-free host orchestration. Device
    # programs remain signature objects; no PyPTO or model kernels are imported.
    host_code = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        functions["_run"], functions["decode_fwd_dspark"], functions["prefill_bootstrap_dspark"],
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(host_code), "selected_lib_host_orchestrator", "exec"), namespace)
    return namespace


def test_production_payload_runs_selected_lib_host_orchestrator(production):
    """Use the actual lib host composition and every stage's declared keys."""
    namespace = _selected_lib_orchestrators(production)
    _reserve(production, "a", 1, 7)
    lowered = production.provider.lower_decode(model=object(), batch=_batch()).lowered
    called = []
    namespace["decode_fwd_dspark"](
        lambda program, arguments: called.append(program._func.__name__),
        lowered.state, lowered.workspace, lowered.target_model, lowered.draft_model,
        lowered.markov_model, lowered.cache_tables, lowered.rope_tables,
        attention_epoch=1, moe_epoch=1, logits_epoch=1)
    assert called[2] == "l3_decode_fwd"
    assert called[-3:] == ["l3_dspark_drafter", "l3_markov_sample_from_hidden", "l3_commit_drafts"]
    assert len(called) == 15


def test_persistent_drafter_epochs_follow_actual_dispatches(production, monkeypatch):
    namespace = _selected_lib_orchestrators(production)
    resident = production.resident
    resident.closed = resident.failed = False
    resident.run_config = object()
    resident.dspark_programs, resident.dspark_param_names = {}, {}
    compiled_names = {}
    for stage in set(PREFILL_STAGE_ORDER + DECODE_STAGE_ORDER):
        entry = {"target_decode": "l3_decode_fwd",
                 "markov_sample": "l3_markov_sample_from_hidden"}.get(stage, "l3_" + stage)
        if stage.startswith("sample_target_chunk_"):
            entry = "l3_sample_target_chunk"
        program = SimpleNamespace(compiled=object())
        resident.dspark_programs[stage] = program
        resident.dspark_param_names[stage] = namespace[entry].param_names
        compiled_names[program.compiled] = stage
    original_import = npu_runner.importlib.import_module

    def import_module(name):
        if name == "models.deepseek_v4_1_flash.decode_fwd_dspark":
            return SimpleNamespace(**namespace)
        if name == "models.deepseek_v4_1_flash.dspark_drafter":
            return SimpleNamespace(DSPARK_CHUNK_REQUESTS=16, DSPARK_DRAFT_LAYERS=3)
        return original_import(name)

    monkeypatch.setattr(npu_runner.importlib, "import_module", import_module)
    drafter_calls, logits_epochs = [], []

    def run(compiled, *values, config):
        stage = compiled_names[compiled]
        arguments = dict(zip(resident.dspark_param_names[stage], values, strict=True))
        if stage == "dspark_drafter":
            drafter_calls.append(arguments)
        if stage == "markov_sample":
            logits_epochs.append(arguments["logits_epoch"])

    resident.worker = SimpleNamespace(run=run)
    _reserve(production, "a", 1, 7, bootstrapped=False)
    dispatch = production.provider.lower_prefill(
        model=object(), batch=_batch(), request_ids=("a",), sampled_token_ids=(123,))
    resident.run_dspark_prefill(
        dispatch.request_id, slot=dispatch.slot, generation=dispatch.generation,
        prompt_length=dispatch.prompt_length, anchor=dispatch.anchor, stages=(), lowered=dispatch.lowered)
    lowered = production.provider.lower_decode(model=object(), batch=_batch()).lowered
    production.provider.lower_decode(model=object(), batch=_batch())
    assert [call["moe_epoch"] for call in drafter_calls] == [1]
    resident._run_dspark_orchestrator(lowered, phase="decode")
    # Reusing a lowered payload is a new dispatch and needs new completion values.
    resident._run_dspark_orchestrator(lowered, phase="decode")
    assert [call["attention_epoch"] for call in drafter_calls] == [1, 4, 7]
    assert [call["moe_epoch"] for call in drafter_calls] == [1, 4, 7]
    assert logits_epochs == [1, 1, 1]

    original = resident.dspark_programs["dspark_drafter"]
    another = SimpleNamespace(compiled=object())
    compiled_names[another.compiled] = "dspark_drafter"
    resident.dspark_programs["dspark_drafter"] = another
    resident._run_dspark_orchestrator(lowered, phase="decode")
    assert drafter_calls[-1]["moe_epoch"] == 1
    resident.dspark_programs["dspark_drafter"] = original
    two_chunks = dict(drafter_calls[-1], head_hidden=SimpleNamespace(shape=(4, 17, 5, 5120)))
    resident.dispatch_dspark("dspark_drafter", two_chunks)
    resident.dispatch_dspark("dspark_drafter", two_chunks)
    assert [call["moe_epoch"] for call in drafter_calls[-2:]] == [10, 16]


@pytest.mark.parametrize("prompt_length", (1, 8, 32))
def test_production_bootstrap_reads_compact_sample_slot(production, prompt_length):
    _reserve(production, "a", 1, 7, bootstrapped=False)
    batch = _batch()
    batch.prompt_lens = [prompt_length]
    lowered = production.provider.lower_prefill(
        model=object(), batch=batch, request_ids=("a",), sampled_token_ids=(123,)).lowered
    workspace = lowered.workspace
    sampled_row = int(workspace["sampled_rows"][1, 0])
    assert sampled_row == 0
    assert int(workspace["prefill_sampled_ids"][1, sampled_row, 0]) == 123
    assert int(workspace["prompt_lengths"][1, 0]) == prompt_length
    assert bool((workspace["sampled_rows"][0] == -1).all())
    assert bool((workspace["prefill_request_ids"][1, prompt_length:] == -1).all())
    assert production.activations == []


def test_release_invalidates_draft_state_before_reusing_slot_pages(production):
    _reserve(production, "a", 1, 7)
    production.provider.lower_decode(model=object(), batch=_batch())
    host = production.host
    host.state.tokens[1, 7].fill_(42)
    host.state.meta[1, 7, 0] = 1
    production.resident.reset_request("a", production.resident.ledger.owners.pop("a"))
    assert "a" not in production.resident.dspark.states.requests
    assert bool((host.state.tokens[1, 7] == 0).all())
    assert int(host.state.meta[1, 7, 0]) == 0
    assert int(host.state.meta[1, 7, 1]) == 2
    assert bool((host.state.slot_ids[1] == -1).all())
    assert bool((host.cache_tables["draft"][1] == -1).all())
    _reserve(production, "new", 1, 7, bootstrapped=False)
    dispatch = production.provider.lower_prefill(
        model=object(), batch=_batch(("new",), (1,)),
        request_ids=("new",), sampled_token_ids=(123,))
    assert dispatch.generation == 2
    assert dispatch.lowered.cache_tables["draft"][1, 0, 0].tolist() == [14, 15]


def test_private_draft_cache_reserves_all_stable_slots(monkeypatch):
    shapes = []

    def allocate(worker, shape, dtype, *, stacked):
        shapes.append(shape)
        return object()

    monkeypatch.setattr(npu_runner, "alloc_device_buffer", allocate)
    ResidentDSparkCache(object(), 4, 256)
    assert shapes == [(4, 3, 24, 128, 1, 512), (4, 3, 24, 128, 1, 16)]


def test_runner_shares_its_request_ledger_with_dspark_resources():
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.preflight = lambda: None
    runner.ledger = None
    runner.runtime = SimpleNamespace(max_batch_size=12, max_seq_len=256)
    runner.plan = SimpleNamespace(placement=SimpleNamespace(dp_size=4))
    runner.resources = SimpleNamespace(dspark=object())
    assert runner._request_ledger() is runner.resources.ledger


def test_production_registers_each_compiled_object_once():
    prefill = L3Callable(object(), "prefill")
    decode = L3Callable(object(), "decode")
    final = L3Callable(object(), "final")
    draft = L3Callable(object(), "draft")
    decode_alias = L3Callable(decode.compiled, "target_decode")
    assert L3Callable.__hash__ is None
    namespace = dict(prefill=prefill, decode=decode, final=final, dspark_programs={
        "target_decode": decode,
        "dspark_drafter": draft,
        "decode_alias": decode_alias,
        "drafter_alias": draft,
    })

    # Run the production registration assembly without checkpoint staging or
    # device compilation. Its input handles are the real, unhashable class.
    source = ast.parse(Path(npu_runner.__file__).read_text(encoding="utf-8"))
    factory = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name == "make_production_bindings")
    allocate = next(node for node in factory.body if isinstance(node, ast.FunctionDef)
                    and node.name == "allocate")
    end = next(index for index, node in enumerate(allocate.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "programs"
                       for target in node.targets))
    start = end
    while start and isinstance(allocate.body[start - 1], (ast.Assign, ast.For)):
        start -= 1
    assembly = ast.Module(body=allocate.body[start:end + 1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(assembly), "production_program_registration", "exec"), namespace)

    programs = namespace["programs"]
    assert tuple(id(program) for program in programs) == tuple(map(id, (prefill, decode, final, draft)))
    assert sum(program.compiled is decode.compiled for program in programs) == 1
