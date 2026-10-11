# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A/C1A prefill must not share their compressed length or page ABI."""

import torch

from pypto_serving.model.deepseek_v41.prefill_metadata_resources import PrefillMetadataResources
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


def test_prefill_metadata_stages_distinct_compression_families_in_shared_memory():
    topology = SegmentTopology(4, 2, 2)
    positions = torch.arange(16, dtype=torch.float32).unsqueeze(1).expand(16, 2)
    tables = (positions.cos(), positions.sin())
    resources = PrefillMetadataResources(
        topology, max_seq_len=16, max_requests=2, window_pages=1,
        c2a_pages=1, c1a_pages=1, rope_tables=tables,
        compressed_rope_tables=tables,
    )
    step = ForwardStep("prefill", (
        RequestSlice("a", 0, 0, 0, (11, 12), 2,
                     {"window": (0,), "c2a": (0,), "c1a": (0,)}),
    ), 1)
    assert resources.stage(step) == (2, 0)
    c2a = resources.arguments("c2a_full")
    c1a = resources.arguments("c1a_full")
    assert c2a["compressed_lens"][0, :2].tolist() == [0, 1]
    assert c1a["compressed_lens"][0, :2].tolist() == [1, 2]
    assert c2a["compressed_lens"].data_ptr() != c1a["compressed_lens"].data_ptr()
    assert c2a["token_to_req_indices"] is c2a["request_ids"]
    assert c1a["window_slots"][0, :2].tolist() == [0, 1]
    assert c2a["freqs_cos"].shape == (8, 16, 2)
    assert c1a["compressed_rope_cos"][0, 1, 0] == tables[0][1, 0]
    assert all(tensor.is_shared() for tensor in resources.inherited_host_tensors)


def test_m0_chunk_fits_prefork_metadata_and_count_buffers():
    topology = SegmentTopology(4, 2, 1024)
    positions = torch.arange(8192, dtype=torch.float32).unsqueeze(1).expand(8192, 64)
    tables = (positions.cos(), positions.sin())
    resources = PrefillMetadataResources(
        topology, max_seq_len=8192, max_requests=1, window_pages=32,
        c2a_pages=16, c1a_pages=32, rope_tables=tables,
        compressed_rope_tables=tables)
    step = ForwardStep("prefill", (
        RequestSlice("a", 0, 0, 0, tuple(range(4096)), 0,
                     {"window": tuple(range(32)), "c2a": tuple(range(16)),
                      "c1a": tuple(range(32))}),
    ), 1)
    assert resources.stage(step) == (4096, 0)
    assert resources.fields["attention_counts"][:, 0].tolist() == [4096] * 4 + [0] * 4
    assert resources.fields["moe_counts"].tolist() == [1024] * 4 + [0] * 4
    assert resources.arguments("c2a_full")["compressed_lens"][0, 4095] == 2048
    assert resources.arguments("c1a_full")["compressed_lens"][0, 4095] == 4096


def test_decode_keeps_two_compression_lengths_and_pages_distinct():
    topology = SegmentTopology.for_decode()
    positions = torch.arange(16, dtype=torch.float32).unsqueeze(1).expand(16, 2)
    tables = (positions.cos(), positions.sin())
    resources = PrefillMetadataResources(
        topology, max_seq_len=16, max_requests=2, window_pages=2,
        c2a_pages=2, c1a_pages=2, rope_tables=tables,
        compressed_rope_tables=tables)
    step = ForwardStep("decode", (
        RequestSlice("a", 0, 0, 1, (12,), 2,
                     {"window": (0,), "c2a": (1,), "c1a": (0,)}),
    ), 1)
    assert resources.stage(step) == (1, 0)
    c2a, c1a = resources.arguments("c2a_full"), resources.arguments("c1a_full")
    assert c2a["compressed_lens"][0, 0] == 1
    assert c1a["compressed_lens"][0, 0] == 2
    assert c2a["index_block_table"][0, 0, 0] == 1
    assert c1a["index_block_table"][0, 0, 0] == 0
    assert c2a["compressed_slots"][0, 0] == 128
    assert c1a["compressed_slots"][0, 0] == 1
