# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host Engram correctness and ownership tests; no embedding table is allocated."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "v41_engram_under_test", ROOT / "pypto_serving/model/deepseek_v41/engram.py"
)
engram = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = engram
SPEC.loader.exec_module(engram)


def layout_config():
    return {
        "engram_layer_ids": [1, 14],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 2,
        "engram_vocab_size": 5,
        "engram_num_embeddings": [36, 88],
    }


def state(**kwargs):
    return engram.EngramHashState(
        engram.EngramLayout.from_config(layout_config()),
        (0, 1, 2, 3),
        compressed_vocab_size=4,
        pad_token_id=2,
        multipliers=((3, 5, 7), (11, 13, 17)),
        **kwargs,
    )


def test_prime_ranges_are_global_disjoint_and_match_checkpoint_row_counts():
    layout = engram.EngramLayout.from_config(layout_config())
    assert layout.primes == (((5, 7), (11, 13)), ((17, 19), (23, 29)))
    assert layout.hash_columns == 4
    config = layout_config()
    config["engram_num_embeddings"] = [37, 88]
    with pytest.raises(ValueError, match="prime buckets"):
        engram.EngramLayout.from_config(config)
    assert engram.EngramLayout.from_config({"engram_layer_ids": []}) is None


def test_released_layout_matches_two_real_table_dimensions():
    config = layout_config()
    config.update(
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_vocab_size=16000000,
        engram_num_embeddings=[384006168, 384016682],
        engram_head_dim=256,
    )
    layout = engram.EngramLayout.from_config(config)
    assert tuple(sum(sum(group) for group in layer) for layer in layout.primes) == (384006168, 384016682)


def test_first_position_uses_padding_and_bucket_offsets():
    hashes = state().advance([0], start_pos=0)
    # Layer 1: [0,2,2] * [3,5,7] -> rolling XORs 10 and 4.
    assert hashes[0][0] == (0, 8, 16, 27)
    # Layer 14: rolling XORs 26 and 56, with offsets [0,17,36,59].
    assert hashes[0][1] == (9, 24, 46, 86)


@pytest.mark.parametrize("split", [1, 2, 5, 9])
def test_chunk_continuity_matches_single_prefill(split):
    tokens = [0, 1, 3, 2, 0, 1, 1, 3, 0, 2]
    mask = [True, True, False, False, True, True, True, True, True, True]
    expected = state().advance(tokens, start_pos=0, token_mask=mask)
    chunked = state()
    result = chunked.advance(tokens[:split], start_pos=0, token_mask=mask[:split])
    result += chunked.advance(tokens[split:], start_pos=split, token_mask=mask[split:])
    assert result == expected


def test_image_boundaries_block_all_older_context():
    request = state()
    result = request.advance([0, 1, 3, 0], start_pos=0, token_mask=[True, False, False, True])
    assert result[-1] == state().advance([0], start_pos=0)[0]
    assert result[1] == result[2], "An image position hashes padding for every lookback"


def test_history_is_bounded_even_for_a_large_chunk_and_zero_rollback_window():
    tokens = [0, 1, 2, 3] * 100
    request = state(rollback_window=8)
    expected = state().advance(tokens, start_pos=0)
    assert request.advance(tokens, start_pos=0) == expected
    assert request.retained_tokens == 10
    assert request.minimum_rollback_position == 392
    no_rollback = state(rollback_window=0)
    assert no_rollback.advance(tokens, start_pos=0) == expected
    assert no_rollback.retained_tokens == 2
    with pytest.raises(ValueError, match="retained"):
        no_rollback.rollback(399)


def test_snapshot_restores_a_whole_chunk_and_cannot_cross_requests():
    request, other = state(rollback_window=2), state()
    request.advance([0, 1], start_pos=0)
    snapshot = request.snapshot()
    request.advance([3] * 100, start_pos=2)
    request.restore(snapshot)
    assert request.position == 2
    assert request.advance([2, 3], start_pos=2) == other.advance([0, 1, 2, 3], start_pos=0)[2:]
    with pytest.raises(ValueError, match="another request"):
        other.restore(snapshot)


def test_rollback_then_divergent_suffix_matches_fresh_history():
    request = state(rollback_window=3)
    request.advance([0, 1, 2, 3, 3, 3, 3, 3], start_pos=0)
    request.rollback(5)
    actual = request.advance([0, 1, 2], start_pos=5)
    expected = state().advance([0, 1, 2, 3, 3, 0, 1, 2], start_pos=0)[5:]
    assert actual == expected
    with pytest.raises(ValueError, match="retained"):
        request.rollback(0)


@pytest.mark.parametrize(
    "tokens,start,mask", [([0, 9], 0, None), ([0], 1, None), ([0], 0, [1]), ([0], 0, [])]
)
def test_invalid_advance_is_atomic(tokens, start, mask):
    request = state()
    before = request.snapshot()
    with pytest.raises(ValueError):
        request.advance(tokens, start_pos=start, token_mask=mask)
    assert request.snapshot() == before


def test_mapping_fingerprint_detects_same_size_but_different_mapping():
    digest = engram.token_map_sha256((0, 1, 2, 3))
    assert state(expected_token_map_sha256=digest).token_map_sha256 == digest
    with pytest.raises(ValueError, match="digest"):
        state(expected_token_map_sha256=engram.token_map_sha256((1, 0, 2, 3)))
    with pytest.raises(ValueError, match="compressed vocabulary"):
        engram.EngramHashState(
            engram.EngramLayout.from_config(layout_config()),
            (0, 2),
            compressed_vocab_size=3,
            pad_token_id=0,
            multipliers=((3, 5, 7), (11, 13, 17)),
        )


def test_default_multipliers_match_pinned_numpy_rng_stream():
    pytest.importorskip("numpy")
    layout = engram.EngramLayout.from_config(layout_config())
    # Fixed outputs of the reference's PCG64 seeds 10007 and 140098, vocab size 4.
    expected = (
        (1898406915353658309, 119898250459655815, 890828962367240933),
        (1677548552443721761, 1276077216879092079, 766014534253018689),
    )
    assert engram.compute_hash_multipliers(layout, 4) == expected
    actual = engram.EngramHashState(layout, (0, 1, 2, 3), compressed_vocab_size=4, pad_token_id=2)
    explicit = engram.EngramHashState(
        layout, (0, 1, 2, 3), compressed_vocab_size=4, pad_token_id=2, multipliers=expected
    )
    assert actual.advance([0, 1, 3], start_pos=0) == explicit.advance([0, 1, 3], start_pos=0)


def test_tokenizer_normalization_preserves_space_and_partial_byte_identity():
    pytest.importorskip("tokenizers")

    class Tokenizer:
        texts = [" The", "the", "THE", "é", "E", " ", "\t", "\ufffd", "\ufffd", ""]

        def __init__(self):
            self.backend_tokenizer = self

        def __len__(self):
            return len(self.texts)

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return self.texts[ids[0]]

        def id_to_token(self, token):
            return f"<0x{token:02X}>"

    mapping, vocab = engram.build_compressed_token_map(Tokenizer())
    assert mapping == (0, 0, 0, 1, 1, 2, 2, 3, 4, 5)
    assert vocab == 6


@pytest.mark.parametrize("array_module", ["numpy", "torch"])
def test_native_int64_overflow_and_xor_match_host_hashes(array_module):
    arrays = pytest.importorskip(array_module)
    tensor = arrays.asarray if array_module == "numpy" else arrays.tensor
    layout = engram.EngramLayout.from_config(layout_config())
    multipliers = ((2**63 - 1, 2**63 - 3, 2**63 - 5), (2**63 - 7, 2**63 - 9, 2**63 - 11))
    request = engram.EngramHashState(
        layout, (0, 1, 2, 3), compressed_vocab_size=4, pad_token_id=2, multipliers=multipliers
    )
    products = tensor([3, 2, 2], dtype=arrays.int64) * tensor(multipliers, dtype=arrays.int64)
    rolling, expected = products[:, 0], [[], []]
    for ngram in (1, 2):
        rolling = rolling ^ products[:, ngram]
        for layer in range(2):
            offset = sum(sum(group) for group in layout.primes[layer][: ngram - 1])
            for prime in layout.primes[layer][ngram - 1]:
                expected[layer].append(int(rolling[layer] % prime) + offset)
                offset += prime
    assert request.advance([3], start_pos=0)[0] == tuple(tuple(row) for row in expected)


def test_sparse_lookup_deduplicates_routes_and_reads_bounded_batches():
    layout, calls = engram.EngramLayout.from_config(layout_config()), []

    def reader(layer, rank, local_ids):
        calls.append((rank, local_ids))
        return {local: (float(rank * 18 + local), float(layer)) for local in local_ids}

    lookup = engram.SparseEngramLookup(layout, reader, world_size=2, max_rows_per_read=2)
    assert lookup.lookup(1, [35, 0, 35, 19, 18, 1]) == ((35, 1), (0, 1), (35, 1), (19, 1), (18, 1), (1, 1))
    assert calls == [(0, (0, 1)), (1, (0, 1)), (1, (17,))]
    assert lookup.lookup(1, []) == ()
    with pytest.raises(ValueError, match="unpadded"):
        lookup.lookup(1, [36])


def test_sparse_shard_padding_and_reader_failures_are_explicit():
    layout = engram.EngramLayout.from_config(layout_config())
    plan = engram.shard_plan(layout, 1, 5)
    assert [(shard.global_start, shard.valid_rows, shard.capacity_rows) for shard in plan] == [
        (0, 8, 8),
        (8, 8, 8),
        (16, 8, 8),
        (24, 8, 8),
        (32, 4, 8),
    ]
    for result in ({}, {0: (1.0,)}, {0: (float("nan"), 0.0)}, {0: (1.0, 2.0), 1: (3.0, 4.0)}):
        lookup = engram.SparseEngramLookup(layout, lambda *_: result, world_size=5)
        with pytest.raises(ValueError):
            lookup.lookup(1, [0])
