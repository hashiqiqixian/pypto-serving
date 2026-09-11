# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 cache metadata tests; no tensors, numerical golden, or device execution."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cache_module = _load("_v41_cache_test", "pypto_serving/model/deepseek_v41/cache.py")
config_module = _load("_v41_cache_config_test", "pypto_serving/model/deepseek_v41/config.py")
CacheStateError = cache_module.CacheStateError


@pytest.fixture
def config():
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text(encoding="utf-8"))
    raw["text_config"].update(
        {
            "num_hidden_layers": 5,
            "num_nextn_predict_layers": 0,
            "compress_ratios": [0, 2, 2, 1, 1],
            "kv_source_layer_ids": [1, 3],
            "index_source_layer_ids": [1, 2, 3, 4],
            "candidate_source_layer_id": 3,
            "engram_layer_ids": [],
            "engram_num_embeddings": [],
            "dspark_target_layer_ids": [],
            "sliding_window": 4,
            "head_dim": 32,
            "index_head_dim": 32,
            "qk_rope_head_dim": 16,
        }
    )
    return config_module.DeepSeekV41Config.from_dict(raw)


@pytest.fixture
def cache(config):
    return cache_module.V41CacheState(config, page_size=4, max_seq_len=64, max_chunk_tokens=4)


def _tables(cache, token_count, offset=0):
    return {
        group.name: tuple(
            range(
                offset,
                offset
                + min(
                    (token_count + group.page_size - 1) // group.page_size,
                    group.max_blocks_per_seq,
                ),
            )
        )
        for group in cache.groups
    }


def _prepare(cache, lease, start, end, offset=0):
    cache.bind(lease, _tables(cache, end, offset))
    return cache.prepare(lease, start, end)


@pytest.fixture
def allocator_modules(monkeypatch):
    """Load the real scheduler allocator, substituting only unavailable import dependencies.

    No torch operation is used by its group metadata path. KVCacheSpec,
    KVCacheGroupSpec, and KvCacheManager are their unmodified production classes.
    """
    if "torch" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    tokenizer = ModuleType("pypto_serving.model.tokenizer")
    tokenizer.TokenizerAdapter = object
    monkeypatch.setitem(sys.modules, "pypto_serving.model.tokenizer", tokenizer)
    types_spec = importlib.util.spec_from_file_location(
        "pypto_serving.config.types", ROOT / "pypto_serving/config/types.py"
    )
    types_module = importlib.util.module_from_spec(types_spec)
    monkeypatch.setitem(sys.modules, types_spec.name, types_module)
    types_spec.loader.exec_module(types_module)
    allocator = _load("_v41_real_allocator_test", "pypto_serving/serving/memory/kv_cache.py")
    return types_module, allocator


def test_published_sources_allocate_four_main_and_index_key_pools():
    config = config_module.DeepSeekV41Config.from_json(ROOT / "tests/fixtures/deepseek_v41/config.json")
    groups = cache_module.build_v41_cache_layouts(config, 128, 16384)
    assert len(groups) == 48
    assert [group.layer_id for group in groups if group.kind == "main_kv"] == [2, 8, 14, 20]
    assert [group.layer_id for group in groups if group.kind == "index_k"] == [2, 8, 14, 20]
    assert len([group for group in groups if group.kind == "swa"]) == 40
    main = next(group for group in groups if group.name == "main_kv.2")
    index = next(group for group in groups if group.name == "index_k.2")
    swa = next(group for group in groups if group.name == "swa.0")
    assert main.rows_per_page == 64
    assert main.row_bytes == 512 // 2 + 512 // 16
    assert index.row_bytes == 128 // 2 + 128 // 32
    assert swa.row_bytes == 512 + 512 // 32
    assert cache_module.CACHE_FORMAT == "v41_packed_v1"


def test_layer_lookup_reuses_source_keys_but_keeps_swa_private(cache):
    assert cache.group_for_layer(2, "main_kv") == "main_kv.1"
    assert cache.group_for_layer(2, "index_k") == "index_k.1"
    assert cache.group_for_layer(4, "index_k") == "index_k.3"
    assert cache.group_for_layer(4, "swa") == "swa.4"
    with pytest.raises(CacheStateError, match="has no"):
        cache.group_for_layer(0, "main_kv")


def test_chunk_boundary_only_emits_complete_compressed_rows(cache):
    lease = cache.create("request", generation=0)
    first = _prepare(cache, lease, 0, 3)
    assert [slot.logical_row for slot in first.write_slots("main_kv.1")] == [0]
    assert first.map_position("main_kv.1", 1, write=True).row_offset == 0
    with pytest.raises(CacheStateError, match="incomplete"):
        first.map_position("main_kv.1", 2, write=True)
    assert list(first.visible_rows("main_kv.1", 0)) == []
    assert list(first.visible_rows("main_kv.1", 1)) == [0]
    cache.commit(first)
    second = _prepare(cache, lease, 3, 5)
    assert [
        (slot.page_id, slot.row_offset, slot.logical_row) for slot in second.write_slots("main_kv.1")
    ] == [(0, 1, 1)]
    assert [(slot.page_id, slot.row_offset) for slot in second.write_slots("swa.0")] == [(0, 3), (1, 0)]
    assert list(second.visible_rows("swa.0", 4)) == [1, 2, 3, 4]
    cache.commit(second)
    third = _prepare(cache, lease, 5, 6)
    assert third.write_slots("main_kv.1")[0].page_id == 1
    assert third.write_slots("main_kv.1")[0].row_offset == 0


def test_index_results_and_candidates_are_shared_only_within_a_transaction(cache):
    lease = cache.create("request", generation=4)
    tx = _prepare(cache, lease, 0, 2)
    with pytest.raises(CacheStateError, match="has not published"):
        tx.index_for(2)
    first, reindexed, candidates = object(), object(), object()
    tx.publish_index(1, first)
    assert tx.index_for(1) is first
    with pytest.raises(CacheStateError, match="has not published"):
        tx.index_for(2)
    tx.publish_index(2, reindexed)
    assert tx.index_for(2) is reindexed
    with pytest.raises(CacheStateError, match="does not own"):
        tx.publish_index(0, object())
    with pytest.raises(CacheStateError, match="already published"):
        tx.publish_index(2, object())
    tx.publish_candidates(3, candidates)
    assert tx.candidates_for(4) is candidates
    with pytest.raises(CacheStateError, match="does not own"):
        tx.publish_candidates(2, object())
    cache.commit(tx)
    with pytest.raises(CacheStateError, match="inactive"):
        tx.index_for(2)
    next_tx = _prepare(cache, lease, 2, 3)
    with pytest.raises(CacheStateError, match="has not published"):
        next_tx.candidates_for(4)


def test_published_layer_reuse_reads_the_latest_configured_index_source():
    config = config_module.DeepSeekV41Config.from_json(ROOT / "tests/fixtures/deepseek_v41/config.json")
    cache = cache_module.V41CacheState(config, page_size=4, max_seq_len=64, max_chunk_tokens=4)
    tx = _prepare(cache, cache.create("request", generation=0), 0, 2)
    first, second, candidates = object(), object(), object()
    tx.publish_index(2, first)
    assert tx.index_for(3) is tx.index_for(7) is first
    with pytest.raises(CacheStateError, match="has not published"):
        tx.index_for(8)
    tx.publish_index(8, second)
    assert tx.index_for(13) is second
    assert tx.index_for(7) is first
    tx.publish_candidates(20, candidates)
    assert tx.candidates_for(24) is tx.candidates_for(36) is candidates
    with pytest.raises(CacheStateError, match="has not published"):
        tx.candidates_for(19)


def test_8k_chunk_metadata_crosses_page_and_partial_compression_boundaries():
    config = config_module.DeepSeekV41Config.from_json(ROOT / "tests/fixtures/deepseek_v41/config.json")
    cache = cache_module.V41CacheState(config, page_size=128, max_seq_len=16384, max_chunk_tokens=8192)
    lease = cache.create("8k-metadata-only", generation=0)
    initial = _prepare(cache, lease, 0, 8191)
    assert len(initial.visible_rows("main_kv.2", 8190)) == 4095
    cache.commit(initial)
    boundary = _prepare(cache, lease, 8191, 8193)
    assert [(slot.page_id, slot.row_offset) for slot in boundary.write_slots("swa.0")] == [(63, 127), (64, 0)]
    compressed = boundary.write_slots("main_kv.2")
    assert [(slot.page_id, slot.row_offset, slot.logical_row) for slot in compressed] == [(63, 63, 4095)]
    assert len(boundary.visible_rows("swa.0", 8192)) == 128
    assert boundary.visible_rows("swa.0", 8192).start == 8065
    with pytest.raises(CacheStateError, match="incomplete"):
        boundary.map_position("main_kv.2", 8192, write=True)
    cache.commit(boundary)
    assert cache.valid_length(lease) == 8193


def test_interleaved_requests_never_share_live_pages_or_temporaries(cache):
    first = cache.create("first", generation=0)
    second = cache.create("second", generation=0)
    a = _prepare(cache, first, 0, 2)
    with pytest.raises(CacheStateError, match="another live request"):
        cache.bind(second, _tables(cache, 2))
    b = _prepare(cache, second, 0, 2, offset=20)
    a.publish_index(1, "first-index")
    with pytest.raises(CacheStateError, match="has not published"):
        b.index_for(1)
    assert a.map_position("swa.0", 0).page_id != b.map_position("swa.0", 0).page_id
    cache.commit(b)
    cache.abort(a)
    assert cache.valid_length(first) == 0
    assert cache.valid_length(second) == 2


def test_commit_many_prevalidates_before_publishing_any_request(cache):
    first = cache.create("first", generation=0)
    second = cache.create("second", generation=0)
    a = _prepare(cache, first, 0, 2)
    b = _prepare(cache, second, 0, 2, offset=20)
    with pytest.raises(CacheStateError, match="duplicate"):
        cache.commit_many([a, a])
    assert cache.valid_length(first) == 0
    cache.abort(b)
    with pytest.raises(CacheStateError, match="inactive"):
        cache.commit_many([a, b])
    assert cache.valid_length(first) == 0
    b = cache.prepare(second, 0, 2)
    cache.validate_commit_many([a, b])
    assert cache.valid_length(first) == cache.valid_length(second) == 0
    cache.commit_many([a, b])
    assert cache.valid_length(first) == cache.valid_length(second) == 2


def test_abort_is_idempotent_but_cannot_revert_a_committed_transaction(cache):
    lease = cache.create("request", generation=0)
    tx = _prepare(cache, lease, 0, 2)
    cache.abort(tx)
    cache.abort(tx)
    assert cache.valid_length(lease) == 0
    with pytest.raises(CacheStateError, match="inactive"):
        tx.map_position("swa.0", 0)
    tx = cache.prepare(lease, 0, 2)
    cache.commit(tx)
    with pytest.raises(CacheStateError, match="inactive"):
        cache.abort(tx)


def test_generation_and_release_prevent_stale_reuse_and_clear_metadata(cache):
    old = cache.create("reused", generation=0)
    tx = _prepare(cache, old, 0, 2)
    assert cache.release(old)
    assert not cache.release(old)
    assert cache.active_request_count == 0
    assert cache._owners == {}
    new = cache.create("reused", generation=1)
    current = _prepare(cache, new, 0, 2)
    assert not cache.release(old)
    with pytest.raises(CacheStateError, match="stale"):
        cache.commit(tx)
    forged = cache_module.CacheLease(new.request_id, new.generation, new.partition)
    with pytest.raises(CacheStateError, match="stale"):
        cache.valid_length(forged)
    cache.commit(current)
    assert cache.valid_length(new) == 2


def test_partition_namespaces_can_use_same_physical_ids(config):
    cache = cache_module.V41CacheState(config, page_size=4, max_seq_len=64, num_partitions=2)
    a = _prepare(cache, cache.create("a", generation=0, partition=0), 0, 1)
    b = _prepare(cache, cache.create("b", generation=0, partition=1), 0, 1)
    assert a.map_position("swa.0", 0).page_id == b.map_position("swa.0", 0).page_id
    cache.commit_many([a, b])


def test_rebind_is_atomic_and_cannot_replace_committed_live_pages(cache):
    lease = cache.create("request", generation=0)
    cache.commit(_prepare(cache, lease, 0, 3))
    bad = _tables(cache, 4)
    bad["swa.4"] = (12,)
    with pytest.raises(CacheStateError, match="committed live page"):
        cache.bind(lease, bad)
    tx = cache.prepare(lease, 3, 4)
    assert tx.map_position("swa.4", 2).page_id == 0
    with pytest.raises(CacheStateError, match="pending"):
        cache.bind(lease, _tables(cache, 4))


@pytest.mark.parametrize("bad_pages", [(True,), (-1,), (0, 0), tuple(range(100))])
def test_invalid_tables_do_not_claim_any_pages(cache, bad_pages):
    lease = cache.create("request", generation=0)
    tables = _tables(cache, 1)
    tables["swa.4"] = bad_pages
    with pytest.raises(CacheStateError):
        cache.bind(lease, tables)
    assert cache._owners == {}


def test_missing_groups_and_insufficient_pages_fail_before_work(cache):
    lease = cache.create("request", generation=0)
    with pytest.raises(CacheStateError, match="exactly"):
        cache.bind(lease, {})
    cache.bind(lease, _tables(cache, 0))
    with pytest.raises(CacheStateError, match="enough pages"):
        cache.prepare(lease, 0, 1)
    assert cache.valid_length(lease) == 0


@pytest.mark.parametrize("start,end", [(1, 2), (0, 0), (0, 5), (True, 2), (0, 65)])
def test_noncontiguous_or_oversized_chunks_rejected(cache, start, end):
    lease = cache.create("request", generation=0)
    cache.bind(lease, _tables(cache, 4))
    with pytest.raises(CacheStateError):
        cache.prepare(lease, start, end)


def test_rollback_hides_suffix_and_checks_retained_swa_rows(cache):
    lease = cache.create("request", generation=0)
    for start in range(0, 20, 4):
        cache.commit(_prepare(cache, lease, start, start + 4))
    cache.rollback(lease, 18)
    assert cache.valid_length(lease) == 18
    tx = cache.prepare(lease, 18, 19)
    with pytest.raises(CacheStateError, match="incomplete"):
        tx.map_position("swa.0", 19)
    with pytest.raises(CacheStateError, match="previously committed"):
        tx.map_position("main_kv.1", 14, write=True)
    cache.abort(tx)
    with pytest.raises(CacheStateError, match="no longer retained"):
        cache.rollback(lease, 10)
    cache.rollback(lease, 0)
    assert cache.valid_length(lease) == 0
    cache.commit(cache.prepare(lease, 0, 4))


def test_rolling_rebind_then_abort_preserves_old_attention_tail(cache):
    lease = cache.create("request", generation=0)
    for start in (0, 4, 8):
        cache.commit(_prepare(cache, lease, start, start + 4))
    tables = _tables(cache, 16)
    for group in cache.groups:
        if group.kind == "swa":
            tables[group.name] = (3, 1, 2)
    cache.bind(lease, tables)
    tx = cache.prepare(lease, 12, 16)
    assert tx.map_position("swa.0", 11).page_id == 2
    assert tx.map_position("swa.0", 12, write=True).page_id == 3
    cache.abort(tx)
    assert cache.valid_length(lease) == 12
    cache.rollback(lease, 8)
    with pytest.raises(CacheStateError, match="no longer retained"):
        cache.rollback(lease, 4)


def test_existing_scheduler_allocator_owns_pages_through_growth_release_and_reuse(config, allocator_modules):
    types_module, allocator_module = allocator_modules
    specs = cache_module.build_v41_cache_group_specs(config, 4, 64, total_pages=32, max_chunk_tokens=4)
    assert all(isinstance(spec, types_module.KVCacheGroupSpec) for spec in specs)
    allocator = allocator_module.KvCacheManager(enable_prefix_cache=False)
    allocator.init_groups(specs, max_batch_size=2, primary_num_blocks=32)
    cache = cache_module.V41CacheState(config, page_size=4, max_seq_len=64, max_chunk_tokens=4)
    lease = cache.create("request", generation=0)
    for start in range(0, 24, 4):
        tables = allocator.ensure_group_blocks("request", start + 4)
        cache.bind(lease, tables)
        tx = cache.prepare(lease, start, start + 4)
        assert len(tx.write_slots("swa.0")) == 4
        cache.commit(tx)
    assert len(allocator.request_blocks) == 0
    assert cache.valid_length(lease) == 24
    assert cache.release(lease)
    allocator.release_all_group_requests("request")
    for spec in specs:
        assert allocator._group_pools[spec.name].num_free_blocks_in(0) == spec.num_blocks
    new = cache.create("new", generation=0)
    tables = allocator.ensure_group_blocks("new", 4)
    cache.bind(new, tables)
    cache.commit(cache.prepare(new, 0, 4))
    assert cache.valid_length(new) == 4


def test_pool_sizing_matches_existing_allocator_scaling(config, allocator_modules):
    specs = cache_module.build_v41_cache_group_specs(config, 4, 64, total_pages=32, max_chunk_tokens=4)
    assert specs[0].name == "main_kv.1"
    assert specs[0].num_blocks == 32
    assert next(spec for spec in specs if spec.name == "swa.0").num_blocks == 6
    assert all(spec.num_blocks == 2 * spec.max_blocks_per_seq for spec in specs)
    with pytest.raises(CacheStateError, match="multiple"):
        cache_module.build_v41_cache_group_specs(config, 4, 64, total_pages=17)


def test_page_geometry_and_context_limit_are_validated(config):
    with pytest.raises(CacheStateError, match="divisible"):
        cache_module.build_v41_cache_layouts(config, 3, 64)
    with pytest.raises(CacheStateError, match="max_position_embeddings"):
        cache_module.build_v41_cache_layouts(config, 4, config.max_position_embeddings + 1)
