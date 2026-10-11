# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pre-fork host metadata for the lib's distinct V4.1 prefill composites."""

import torch

from .compressed_metadata import prepare_compressed_metadata
from .swa_metadata import gather_swa_rope_rows, prepare_swa_window_metadata


class PrefillMetadataResources:
    """Keep C2A and C1A page/length contracts separate across one layer chain."""

    def __init__(self, topology, *, max_seq_len, max_requests, window_pages,
                 c2a_pages, c1a_pages, rope_tables, compressed_rope_tables):
        if any(type(value) is not int or value <= 0 for value in
               (max_seq_len, max_requests, window_pages, c2a_pages, c1a_pages)):
            raise ValueError("prefill metadata capacities must be positive integers")
        for pair in (rope_tables, compressed_rope_tables):
            if (len(pair) != 2 or any(not isinstance(value, torch.Tensor) or
                value.device.type != "cpu" or value.dtype != torch.float32 or
                value.ndim != 2 or value.shape[0] < max_seq_len for value in pair)
                    or pair[0].shape != pair[1].shape):
                raise ValueError("prefill requires lib FP32 RoPE tables covering the model context")
        self.topology = topology
        self.max_requests = max_requests
        self.window_pages = window_pages
        self.compressed_pages = {"c2a": c2a_pages, "c1a": c1a_pages}
        self.fields = {}

        def shared(name, shape, dtype, fill=0):
            value = torch.empty(shape, dtype=dtype).share_memory_()
            value.fill_(fill)
            self.fields[name] = value
            return value

        world, tokens, width = topology.world, topology.capacity, rope_tables[0].shape[1]
        shared("attention_counts", (world, 1), torch.int32)
        shared("moe_counts", (world,), torch.int32)
        shared("window_slots", (world, tokens), torch.int64, -1)
        shared("window_indices", (world, tokens, 128), torch.int32, -1)
        for name in ("rope_cos", "rope_sin", "compressed_rope_cos", "compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for name in ("c2a_compressed_rope_cos", "c2a_compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for prefix, pages in self.compressed_pages.items():
            for name, shape, dtype, fill in (
                ("position_ids", (world, tokens), torch.int32, 0),
                ("request_ids", (world, tokens), torch.int32, -1),
                ("query_start_loc", (world, max_requests + 1), torch.int32, 0),
                ("compressed_lens", (world, tokens), torch.int32, 0),
                ("compressed_slots", (world, tokens), torch.int64, -1),
                ("compressed_rope_positions", (world, tokens), torch.int32, -1),
                ("index_block_table", (world, max_requests, pages), torch.int32, -1),
                ("state_block_table", (world, max_requests, 1), torch.int32, -1),
            ):
                shared(prefix + "_" + name, shape, dtype, fill)
        self.tables = {}
        for prefix, pair in (("", rope_tables), ("compressed_", compressed_rope_tables)):
            for suffix, value in zip(("cos", "sin"), pair):
                target = shared(prefix + "freqs_" + suffix,
                                (world, max_seq_len, width), torch.float32)
                target.copy_(value[:max_seq_len].unsqueeze(0).expand_as(target))
            self.tables[prefix] = pair

    @property
    def inherited_host_tensors(self):
        return tuple(self.fields.values())

    def _compressed_rope_rows(self, positions):
        cos, sin = self.tables["compressed_"]
        active = positions >= 0
        if bool((positions[active] >= cos.shape[0]).any()):
            raise ValueError("compressed RoPE position exceeds the lib table")
        result_cos = torch.ones((*positions.shape, cos.shape[1]), dtype=torch.float32)
        result_sin = torch.zeros_like(result_cos)
        result_cos[active] = cos[positions[active].long()]
        result_sin[active] = sin[positions[active].long()]
        return result_cos, result_sin

    def stage(self, step):
        swa = prepare_swa_window_metadata(
            step, self.topology, cache_pages=self.window_pages)
        self.fields["window_slots"].copy_(swa.window_slots)
        self.fields["window_indices"].copy_(swa.window_indices)
        cos, sin = gather_swa_rope_rows(
            swa, self.tables[""], max_prepare_bytes=64 << 20)
        self.fields["rope_cos"].copy_(cos)
        self.fields["rope_sin"].copy_(sin)
        attention, moe = self.topology.counts(swa.group_counts)
        self.fields["attention_counts"][:, 0].copy_(torch.tensor(attention, dtype=torch.int32))
        self.fields["moe_counts"].copy_(torch.tensor(moe, dtype=torch.int32))
        for prefix, ratio in (("c2a", 2), ("c1a", 1)):
            metadata = prepare_compressed_metadata(
                step, self.topology, ratio=ratio, compressed_group=prefix,
                cache_pages=self.compressed_pages[prefix], max_requests=self.max_requests,
                state_blocks=self.max_requests,
            )
            if metadata.group_counts != swa.group_counts:
                raise ValueError("SWA and compressed metadata token layouts disagree")
            for name in ("position_ids", "request_ids", "query_start_loc", "compressed_lens",
                         "compressed_slots", "compressed_rope_positions", "state_block_table"):
                self.fields[prefix + "_" + name].copy_(getattr(metadata, name))
            table = self.fields[prefix + "_index_block_table"]
            source = metadata.index_block_table
            if source.shape[2] > table.shape[2]:
                raise ValueError("compressed page table exceeds its pre-fork capacity")
            table.fill_(-1)
            table[:, :, :source.shape[2]].copy_(source)
            rows = self._compressed_rope_rows(metadata.compressed_rope_positions)
            if prefix == "c2a":
                self.fields["c2a_compressed_rope_cos"].copy_(rows[0])
                self.fields["c2a_compressed_rope_sin"].copy_(rows[1])
            else:
                self.fields["compressed_rope_cos"].copy_(rows[0])
                self.fields["compressed_rope_sin"].copy_(rows[1])
        return swa.group_counts

    def arguments(self, mode):
        common = {name: self.fields[name] for name in
                  ("window_slots", "window_indices", "rope_cos", "rope_sin")}
        if mode == "swa":
            return common
        if mode not in ("c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse"):
            raise ValueError("unknown V4.1 prefill metadata mode")
        prefix = mode[:3]
        names = ("request_ids", "compressed_lens", "compressed_slots", "index_block_table")
        result = dict(common, **{name: self.fields[prefix + "_" + name] for name in names})
        if mode == "c2a_full":
            result.update({
                "freqs_cos": self.fields["freqs_cos"],
                "freqs_sin": self.fields["freqs_sin"],
                "compressed_freqs_cos": self.fields["compressed_freqs_cos"],
                "compressed_freqs_sin": self.fields["compressed_freqs_sin"],
                "token_to_req_indices": result["request_ids"],
                "position_ids": self.fields["c2a_position_ids"],
                "compressed_rope_positions": self.fields["c2a_compressed_rope_positions"],
                "compressed_rope_cos": self.fields["c2a_compressed_rope_cos"],
                "compressed_rope_sin": self.fields["c2a_compressed_rope_sin"],
                "query_start_loc": self.fields["c2a_query_start_loc"],
                "state_block_table": self.fields["c2a_state_block_table"],
            })
        elif mode.startswith("c1a_"):
            result["compressed_rope_cos"] = self.fields["compressed_rope_cos"]
            result["compressed_rope_sin"] = self.fields["compressed_rope_sin"]
        return result
