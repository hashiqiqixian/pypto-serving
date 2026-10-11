# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The final lib composite must share the backbone worker and host lifetime."""

from types import MappingProxyType
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import resident_resources
from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.prefill_metadata_resources import PrefillMetadataResources
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.segment_inputs import prepare_segment_inputs
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


def test_resident_resources_register_every_phase_on_one_worker(monkeypatch):
    events = []
    shared = torch.empty(1).share_memory_()
    program = object()
    worker = SimpleNamespace(close=lambda: events.append("worker.close"))
    bank = SimpleNamespace(tensors={"w": shared}, finish=lambda names: events.append(tuple(names)))
    decode_topology = SimpleNamespace(local_capacity=48)
    input_host = SimpleNamespace(inherited_host_tensors=(shared,), topology=decode_topology)
    prefill_host = SimpleNamespace(
        inherited_host_tensors=(shared,), stage=lambda step: (1, 0),
        arguments=lambda mode: {"mode": mode}, topology=SimpleNamespace(local_capacity=1024),
    )
    decode_host = SimpleNamespace(
        inherited_host_tensors=(shared,), stage=lambda step: (1, 0), topology=decode_topology,
        max_requests=3, window_pages=66, compressed_pages={"c2a": 34, "c1a": 66})
    final_host = SimpleNamespace(all_host_tensors=(shared,))

    def worker_factory(programs, config, inherited):
        assert programs == (program,)
        assert config == "config"
        assert len(inherited) == 8
        events.append("worker.create")
        return worker

    def resource(name):
        def create(actual_worker, *args, **kwargs):
            assert actual_worker is worker
            events.append(name + ".create")
            return SimpleNamespace(
                plan=SimpleNamespace(physical_request_slots=3,
                                     family_groups=SimpleNamespace(window="window", c2a="c2a", c1a="c1a"),
                                     physical_group_blocks={"window": 66, "c2a": 34, "c1a": 66},
                                     group_sources={"window": {"window": ()},
                                                    "c2a": {"compressed": (), "index": ()},
                                                    "c1a": {"compressed": (), "index": ()}}),
                close=lambda: events.append(name + ".close"),
                stage=lambda embeddings, step: (embeddings, step),
                run=lambda state, inputs: (state, inputs),
                reset_request=lambda owner: events.append((name, owner)),
            )
        return create

    for name, attr in (("weights", "ResidentDecodeWeights"),
                       ("cache", "DecodeCachePools"), ("input", "InputStateRuntime"),
                       ("final", "FinalOutputRuntime")):
        monkeypatch.setattr(resident_resources, attr, resource(name))
    monkeypatch.setattr(resident_resources, "pad_single_decode",
                        lambda step, embeddings, plan: (step, embeddings, None))

    owner = object()
    resources = resident_resources.V41ResidentResources(
        (program,), "config", weight_bank=bank, weight_param_names=("w",),
        cache_plan=object(), state_reset=shared, page_resets={"page": shared},
        input_host=input_host, prefill_host=prefill_host, final_host=final_host, final_program=program,
        final_param_names=("x_hc",), topology=object(), decode_input_host=input_host,
        decode_host=decode_host, worker_factory=worker_factory,
    )
    assert events[2:7] == [
        "weights.create", "input.create", "input.create", "final.create", "cache.create",
    ]
    step = SimpleNamespace(phase="prefill")
    assert resources.begin_step("embedding", step) == "embedding"
    assert resources.prefill_arguments("c1a_full", step) == ({"mode": "c1a_full"}, (1, 0))
    with pytest.raises(RuntimeError, match="already active"):
        resources.begin_step("embedding", step)
    assert resources.finish_step("state", step) == ("state", step)
    prefill = SimpleNamespace(
        plans=(object(),), attention={0: object()}, moe={0: object()},
        close=lambda: events.append("prefill.close"),
    )
    monkeypatch.setattr(resident_resources, "PrefillLayerResources", lambda *args, **kwargs: prefill)
    assert resources.bind_prefill(prefill.plans, kv_sources=(), index_sources=(),
                                  c2a_sources=(), max_seq_len=16) is prefill
    with pytest.raises(RuntimeError, match="bound once"):
        resources.bind_prefill(prefill.plans, kv_sources=(), index_sources=(),
                               c2a_sources=(), max_seq_len=16)
    decode_layers = SimpleNamespace(
        plans=(SimpleNamespace(layer_id=0),), arguments={0: {"weight": object()}},
        close=lambda: events.append("decode.close"),
    )
    monkeypatch.setattr(resident_resources, "DecodeLayerResources", lambda *args, **kwargs: decode_layers)
    assert resources.bind_decode_layers(decode_layers.plans, ("weight",),
                                        kv_sources=(), index_sources=(), c2a_sources=()) is decode_layers
    with pytest.raises(ValueError, match="active request step"):
        resources.finish_step("state", step)
    resources.begin_step("embedding", step)
    with pytest.raises(ValueError, match="active request step"):
        resources.finish_step("state", object())
    resources.reset_request(owner)
    resources.begin_step("embedding", step)
    segment = SimpleNamespace(
        worker=worker, run_chain=lambda state, plans, attention, moe, *, group_counts:
        (state, plans, attention, moe, group_counts),
    )
    assert resources.run_prefill(segment, "state", step) == (
        "state", prefill.plans, prefill.attention, prefill.moe, (1, 0))
    assert resources.finish_step("state", step) == ("state", step)
    decode_step = SimpleNamespace(phase="decode")
    assert resources.begin_step("decode_embedding", decode_step) == "decode_embedding"
    decode_segment = SimpleNamespace(
        worker=worker, run_layer=lambda state, plan, args, *, group_counts:
        (state, plan.layer_id, args["weight"], group_counts),
    )
    assert resources.run_decode_layers(decode_segment, "state", decode_step) == (
        "state", 0, decode_layers.arguments[0]["weight"], (1, 0))
    with pytest.raises(resident_resources.MissingCompositeInterface, match="both C2A and C1A"):
        resources.decode_backbone_metadata(decode_step)
    assert resources.finish_step("decode_state", decode_step) == ("decode_state", decode_step)
    resources.close()
    resources.close()
    assert events[-9:] == [
        ("cache", owner), "decode.close", "prefill.close", "cache.close",
        "final.close", "input.close", "input.close", "weights.close",
        "worker.close",
    ]


def test_resident_resources_reject_unregistered_output_before_worker():
    shared = torch.empty(1).share_memory_()
    bank = SimpleNamespace(tensors={"w": shared}, finish=lambda names: None)
    host = SimpleNamespace(inherited_host_tensors=(shared,), all_host_tensors=(shared,))
    with pytest.raises(ValueError, match="registered"):
        resident_resources.V41ResidentResources(
            (), None, weight_bank=bank, weight_param_names=("w",), cache_plan=None,
            state_reset=shared, page_resets={}, input_host=host, prefill_host=host, final_host=host,
            final_program=object(), final_param_names=(), topology=None,
            worker_factory=lambda *args: pytest.fail("worker should not start"),
        )


def test_resident_resources_release_partial_upload_on_failure(monkeypatch):
    events = []
    shared = torch.empty(1).share_memory_()
    program = object()
    bank = SimpleNamespace(tensors={"w": shared}, finish=lambda names: None)
    host = SimpleNamespace(inherited_host_tensors=(shared,), all_host_tensors=(shared,))
    worker = SimpleNamespace(close=lambda: events.append("worker"))

    def owned(name):
        def create(actual_worker, *args):
            assert actual_worker is worker
            return SimpleNamespace(close=lambda: events.append(name))
        return create

    monkeypatch.setattr(resident_resources, "ResidentDecodeWeights", owned("weights"))
    monkeypatch.setattr(resident_resources, "DecodeCachePools", owned("cache"))

    def fail_input(worker, host):
        raise RuntimeError("input allocation failed")

    monkeypatch.setattr(resident_resources, "InputStateRuntime", fail_input)
    with pytest.raises(RuntimeError, match="input allocation failed"):
        resident_resources.V41ResidentResources(
            (program,), None, weight_bank=bank, weight_param_names=("w",),
            cache_plan=None, state_reset=shared, page_resets={}, input_host=host, prefill_host=host,
            final_host=host, final_program=program, final_param_names=(),
            topology=None, worker_factory=lambda *args: worker,
        )
    assert events == ["weights", "worker"]


def test_resident_decode_padding_keeps_only_real_output_row(monkeypatch):
    shared = torch.zeros(1).share_memory_()
    topology = SegmentTopology.for_decode()
    prefill_topology = SegmentTopology(local_capacity=1024)
    table = torch.ones((16, 64), dtype=torch.float32)
    metadata = PrefillMetadataResources(
        topology, max_seq_len=16, max_requests=3, window_pages=66,
        c2a_pages=34, c1a_pages=66, rope_tables=(table, table),
        compressed_rope_tables=(table, table))
    groups = {"window": 65, "c2a": 33, "c1a": 65}
    plan = SimpleNamespace(group_blocks=groups, physical_group_blocks={name: blocks + 1
                           for name, blocks in groups.items()},
                           family_groups=SimpleNamespace(window="window", c2a="c2a", c1a="c1a"),
                           group_sources={"window": {"window": ()},
                                          "c2a": {"compressed": (), "index": ()},
                                          "c1a": {"compressed": (), "index": ()}},
                           request_slots=2, physical_request_slots=3)
    resets = []
    monkeypatch.setattr(resident_resources, "ResidentDecodeWeights",
                        lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(resident_resources, "DecodeCachePools", lambda *args:
                        SimpleNamespace(plan=plan, reset_padding=resets.append, close=lambda: None))
    monkeypatch.setattr(resident_resources, "InputStateRuntime", lambda worker, host:
                        SimpleNamespace(stage=lambda embeddings, step:
                                        (LayerState(object(), object(), "tp_local_token"),
                                         prepare_segment_inputs(embeddings, step, host.topology)),
                                        close=lambda: None))
    monkeypatch.setattr(resident_resources, "FinalOutputRuntime", lambda *args, **kwargs:
                        SimpleNamespace(run=lambda state, inputs: inputs.request_last_rows,
                                        close=lambda: None))
    program = object()
    worker = SimpleNamespace(close=lambda: None)
    host = SimpleNamespace(inherited_host_tensors=(shared,), topology=prefill_topology)
    decode_input = SimpleNamespace(inherited_host_tensors=(shared,), topology=topology)
    resources = resident_resources.V41ResidentResources(
        (program,), None, weight_bank=SimpleNamespace(tensors={"w": shared}, finish=lambda names: None),
        weight_param_names=("w",), cache_plan=plan, state_reset=shared,
        page_resets={"page": shared}, input_host=host, decode_input_host=decode_input,
        prefill_host=host, decode_host=metadata,
        final_host=SimpleNamespace(all_host_tensors=(shared,)), final_program=program,
        final_param_names=(), topology=topology, worker_factory=lambda *args: worker)
    request = RequestSlice("real", 0, 0, 2, (42,), 2,
                           MappingProxyType({name: (0,) for name in groups}))
    step = ForwardStep("decode", (request,), 4)
    state = resources.begin_step(torch.ones((1, 5120), dtype=torch.bfloat16), step)
    assert resources._group_counts == (1, 1)
    assert resets == [1]
    assert resources.finish_step(state, step) == ((0, 0),)
    resources.close()


@pytest.mark.parametrize("field,value,error", (
    ("max_requests", 2, "filler state slot"),
    ("window_pages", 65, "window metadata"),
    ("c2a_pages", 33, "c2a metadata"),
    ("c1a_pages", 65, "c1a metadata"),
))
def test_resident_rejects_decode_metadata_without_filler_capacity(monkeypatch, field, value, error):
    shared = torch.zeros(1).share_memory_()
    events = []
    worker = SimpleNamespace(close=lambda: events.append("worker"))
    program = object()
    host = SimpleNamespace(inherited_host_tensors=(shared,), all_host_tensors=(shared,),
                           topology=SimpleNamespace(local_capacity=48))
    metadata = SimpleNamespace(inherited_host_tensors=(shared,), topology=host.topology,
                               max_requests=3, window_pages=66,
                               compressed_pages={"c2a": 34, "c1a": 66})
    if field.endswith("_pages") and field != "window_pages":
        metadata.compressed_pages[field[:3]] = value
    else:
        setattr(metadata, field, value)
    plan = SimpleNamespace(physical_request_slots=3,
                           family_groups=SimpleNamespace(window="window", c2a="c2a", c1a="c1a"),
                           physical_group_blocks={"window": 66, "c2a": 34, "c1a": 66})

    def resource(name):
        return lambda *args, **kwargs: SimpleNamespace(
            plan=plan, close=lambda: events.append(name))

    monkeypatch.setattr(resident_resources, "ResidentDecodeWeights", resource("weights"))
    monkeypatch.setattr(resident_resources, "InputStateRuntime", resource("input"))
    monkeypatch.setattr(resident_resources, "FinalOutputRuntime", resource("final"))
    monkeypatch.setattr(resident_resources, "DecodeCachePools", resource("cache"))
    with pytest.raises(ValueError, match=error):
        resident_resources.V41ResidentResources(
            (program,), None, weight_bank=SimpleNamespace(
                tensors={"w": shared}, finish=lambda names: None),
            weight_param_names=("w",), cache_plan=plan, state_reset=shared,
            page_resets={}, input_host=host, decode_input_host=host,
            prefill_host=host, decode_host=metadata, final_host=host,
            final_program=program, final_param_names=(), topology=object(),
            worker_factory=lambda *args: worker)
    assert events == ["cache", "final", "input", "input", "weights", "worker"]


def test_resident_worker_sizes_cache_after_weights_and_fixed_buffers(monkeypatch):
    shared = torch.zeros(1).share_memory_()
    events = []
    program = object()
    worker = SimpleNamespace(close=lambda: events.append("worker.close"))
    host = SimpleNamespace(inherited_host_tensors=(shared,), all_host_tensors=(shared,),
                           topology=SimpleNamespace(local_capacity=48))

    def resident(name):
        def create(*args, **kwargs):
            events.append(name)
            return SimpleNamespace(close=lambda: events.append(name + ".close"))
        return create

    monkeypatch.setattr(resident_resources, "ResidentDecodeWeights", resident("weights"))
    monkeypatch.setattr(resident_resources, "InputStateRuntime", resident("input"))
    monkeypatch.setattr(resident_resources, "FinalOutputRuntime", resident("final"))
    cache = SimpleNamespace(close=lambda: events.append("cache.close"))

    def planner(actual_worker):
        assert actual_worker is worker
        events.append("size-cache")
        return 4, lambda slots: slots

    def allocate(actual_worker, slots, plan_for_slots, state_reset, page_resets):
        assert actual_worker is worker and slots == 4 and plan_for_slots(2) == 2
        events.append("allocate-cache")
        return cache

    monkeypatch.setattr(resident_resources, "allocate_cache_with_retry", allocate)
    resources = resident_resources.V41ResidentResources(
        (program,), None, weight_bank=SimpleNamespace(tensors={"w": shared}, finish=lambda names: None),
        weight_param_names=("w",), cache_plan=planner, state_reset=shared,
        page_resets={"page": shared}, input_host=host, prefill_host=host,
        final_host=host, final_program=program, final_param_names=(),
        topology=object(), worker_factory=lambda *args: worker)
    assert events == ["weights", "input", "final", "size-cache", "allocate-cache"]
    resources.close()
    assert events[-5:] == ["cache.close", "final.close", "input.close",
                           "weights.close", "worker.close"]
