"""CPU contracts for the complete V4.1 decode dispatch."""

from types import SimpleNamespace

import torch

from pypto.runtime import DeviceTensor
from pypto_serving.model.deepseek_v41.npu_runner import DecodeForwardSegment, LayerState, SegmentTopology
from pypto_serving.model.deepseek_v41.weight_loader import ResidentFullDecodeWeights


def test_weight_views_alias_one_full_rank_allocation():
    class Worker:
        def __init__(self):
            self.next_ptr = 1000
            self.copies = []
            self.freed = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 1000
            return DeviceTensor(self.next_ptr, shape, dtype)

        def copy_to_offset(self, base, offset, source, nbytes, *, worker_id):
            self.copies.append((base, offset, nbytes, worker_id))

        def free_tensor(self, tensor, *, worker_id):
            self.freed.append((tensor.data_ptr, worker_id))

    host = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3).share_memory_()
    bank = SimpleNamespace(
        tensors={"hc_attn_fn": host},
        layout=SimpleNamespace(world=2, slots=lambda name: 2),
        finish=lambda names: None,
    )
    worker = Worker()
    resident = ResidentFullDecodeWeights(worker, bank, ("hc_attn_fn",))
    full = resident.full("hc_attn_fn")
    second = resident.view("hc_attn_fn", 1)
    assert full.shape == (2, 4, 3)
    assert second.shape == (2, 2, 3)
    assert second.shards[0].data_ptr == full.shards[0].data_ptr + 2 * 3 * 4
    assert second.shards[1].data_ptr == full.shards[1].data_ptr + 2 * 3 * 4
    assert len(worker.copies) == 4
    resident.close()
    assert len(worker.freed) == 2


def test_full_decode_binds_distinct_c2a_and_c1a_metadata():
    fields = {source: object() for source in DecodeForwardSegment._METADATA.values()}
    for name in (
        "window_slots", "window_indices", "c2a_compressed_lens", "c1a_compressed_lens",
        "c2a_compressed_slots", "c1a_compressed_slots", "moe_counts",
    ):
        fields[name] = object()
    metadata = SimpleNamespace(fields=fields)
    weight = object()
    cache_tensor = object()
    weights = SimpleNamespace(layout_names=("hc_attn_fn",), full=lambda name: weight)
    cache = SimpleNamespace(_full={"window_cache_pool": cache_tensor},
                            full=lambda name: cache_tensor)
    names = (
        "x_hc", "pre_mix", "attention_num_tokens", "hc_attn_fn", "window_cache_pool",
        "moe_num_tokens", *DecodeForwardSegment._METADATA,
        "window_slots", "window_indices", "c2a_compressed_lens", "c1a_compressed_lens",
        "c2a_compressed_slots", "c1a_compressed_slots",
    )
    calls = []
    worker = SimpleNamespace(_persistent=True, _reset_persistent_windows=False,
                             run=lambda *args, **kwargs: calls.append(args))
    program = SimpleNamespace(compiled=object())
    segment = DecodeForwardSegment(worker, program, names, object(),
                                   SegmentTopology.for_decode(tp=1, dp=8),
                                   weights, cache, metadata)
    state = LayerState(object(), object(), "tp_local_token")
    assert segment.run(state, group_counts=(1,) * 8) is state
    bound = dict(zip(names, calls[0][1:]))
    assert bound["c2a_compressed_lens"] is not bound["c1a_compressed_lens"]
    assert bound["c2a_index_block_table"] is not bound["c1a_index_block_table"]
    assert bound["x_hc"] is state.residual
    assert bound["pre_mix"] is state.pre_mix
    assert bound["hc_attn_fn"] is weight
    assert worker._reset_persistent_windows is False
