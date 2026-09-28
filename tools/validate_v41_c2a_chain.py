# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded real-weight C2A Full -> MoE -> Reuse -> MoE device diagnostic.

Start from the saved output of the two SWA layers. This does not claim that
that input passed its accumulated precision gate, or run a full model. Only
existing native same-input stage comparators gate this diagnostic.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def select_active_state(saved, topology, group_counts):
    """Select causal prefixes for a ragged diagnostic, preserving the source artifact."""
    import torch

    topology.counts(group_counts)
    residual, mix = saved["actual_residual"], saved["actual_pre_mix"]
    if residual.shape != (topology.world, topology.local_capacity, 4, 5120) or mix.shape != residual.shape[:-1]:
        raise ValueError("saved SWA state has incompatible topology or shape")
    if residual.dtype != torch.float32 or mix.dtype != torch.float32:
        raise ValueError("saved SWA state must preserve its FP32 storage")
    if not torch.isfinite(residual).all() or not torch.isfinite(mix).all():
        raise ValueError("saved SWA state must be finite")
    residual, mix = residual.clone(), mix.clone()
    for rank, count in enumerate(topology.counts(group_counts)[1]):
        residual[rank, count:] = 0
        mix[rank, count:] = 0
    return residual, mix


def prepare(args, topology, module):
    import torch
    from golden.spec import TensorSpec
    from models.deepseek_v4_1_flash.config import FLASH
    from models.deepseek_v4_1_flash.rope_tables import precompute_rope_tables
    from pypto_serving.model.deepseek_v41.compressed_metadata import prepare_compressed_metadata
    from pypto_serving.model.deepseek_v41.execution_plan import plan_layers
    from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
    from pypto_serving.model.deepseek_v41.swa_metadata import gather_swa_rope_rows, prepare_swa_window_metadata
    from pypto_serving.model.deepseek_v41.swa_weights import load_prefill_layer_weights

    saved = torch.load(args.input_state, map_location="cpu", weights_only=True)
    if saved.get("request_inputs") is None or saved.get("input_source") != "embeddings":
        raise ValueError("input must come from the explicit fresh-request embedding diagnostic")
    ids = saved["token_ids"]
    if ids is None or tuple(ids.shape) != (topology.dp, topology.capacity):
        raise ValueError("saved SWA diagnostic must contain exactly the same packed token capacity")
    group_counts = args.group_counts
    global_counts, local_counts = topology.counts(group_counts)
    residual, mix = select_active_state(saved, topology, group_counts)
    raw = json.loads((Path(args.model_dir) / "config.json").read_text())
    plans = plan_layers(raw)[2:4]
    if tuple(p.mode for p in plans) != ("c2a_full", "c2a_reuse"):
        raise ValueError("checkpoint layers 2/3 are not the required Full/Reuse pair")
    text = raw["text_config"]
    for key in ("qk_rope_head_dim", "rope_theta", "compress_rope_theta"):
        if text[key] != getattr(FLASH, key):
            raise ValueError(f"lib/checkpoint RoPE mismatch: {key}")
    for source, target in (("factor", "rope_factor"), ("beta_fast", "beta_fast"),
                           ("beta_slow", "beta_slow"), ("original_max_position_embeddings",
                                                      "original_max_position_embeddings")):
        if text["rope_scaling"][source] != getattr(FLASH, target):
            raise ValueError(f"lib/checkpoint compressed RoPE mismatch: {source}")
    # Fresh first-chunk history at layer 2, private pages per DP group.
    pages = (topology.capacity + 127) // 128
    step = ForwardStep("prefill", tuple(RequestSlice(str(g), g, 0, 0, tuple(row[:group_counts[g]].tolist()),
                       topology.capacity, {"window": tuple(range(pages)), "cmp": tuple(range(pages))})
                       for g, row in enumerate(ids) if group_counts[g]), 1)
    tables = precompute_rope_tables(topology.capacity, False)
    compressed_tables = precompute_rope_tables(topology.capacity, True)
    fixture = SimpleNamespace(tokens=topology.capacity, requests=1, dp=topology.dp,
                              seed=11, case="mixed", dp_tokens=None, epochs=1, bench=False)
    attention, moe = {}, {}
    for plan in plans:
        mode = plan.mode.removeprefix("c2a_")
        values = {s.name: s.create_tensor().contiguous() for s in module.build_specs(fixture, mode, {})
                  if isinstance(s, TensorSpec)}
        print(f"Loading real checkpoint layer {plan.layer_id}", flush=True)
        aw, mw = load_prefill_layer_weights(args.model_dir, plan.layer_id, topology)
        values.update(aw, x_hc=residual, pre_mix=mix)
        values["num_tokens"] = torch.tensor(global_counts, dtype=torch.int32).reshape(-1, 1)
        window = prepare_swa_window_metadata(step, topology, cache_pages=values["window_cache"].shape[1])
        values.update(window_slots=window.window_slots, window_indices=window.window_indices)
        values["window_cache"].view(torch.uint8).zero_()
        values["window_cache_scale"].view(torch.uint8).fill_(127)
        if mode == "full":
            cm = prepare_compressed_metadata(step, topology, ratio=2, compressed_group="cmp",
                cache_pages=values["compressed_cache"].shape[1], max_requests=1,
                state_blocks=values["state_cache"].shape[1])
            values.update(token_to_req_indices=cm.request_ids, compressed_lens=cm.compressed_lens,
                          compressed_slots=cm.compressed_slots, index_block_table=cm.index_block_table,
                          position_ids=cm.position_ids, query_start_loc=cm.query_start_loc,
                          state_block_table=cm.state_block_table,
                          compressed_rope_positions=cm.compressed_rope_positions)
            for name, table in zip(("freqs_cos", "freqs_sin", "compressed_freqs_cos", "compressed_freqs_sin"),
                                   (*tables, *compressed_tables)):
                values[name] = table.unsqueeze(0).repeat(topology.world, 1, 1)
            for name in ("compressed_cache", "index_cache"):
                values[name].view(torch.uint8).zero_()
            values["compressed_cache_scale"].view(torch.uint8).fill_(0x38)  # E4M3 scale 1
            values["index_cache_scale"].view(torch.uint8).fill_(127)  # E8M0 scale 1
            values["state_cache"].zero_()
            values["topk_indices"].fill_(-1)
        else:
            values["rope_cos"], values["rope_sin"] = gather_swa_rope_rows(window, tables)
        attention[plan.layer_id] = values
        moe[plan.layer_id] = dict(mw, next_pre_mix=torch.zeros_like(mix),
            x_mixed=torch.zeros(topology.world, topology.local_capacity, 5120, dtype=torch.bfloat16),
            x_next=torch.zeros_like(residual), num_tokens=torch.tensor(local_counts, dtype=torch.int32))
    return plans, attention, moe, residual, mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-state", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--group-counts", help="Comma-separated causal prefix lengths per DP group")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-dir", default="build_output/v41-c2a-chain")
    parser.add_argument("--artifact-dir", default=".validation-artifacts/c2a-chain")
    parser.add_argument("--ring-heap-mib", type=int, default=4096)
    args = parser.parse_args()
    import torch
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology, load_segment_modules
    from pypto_serving.model.deepseek_v41.prefill_segment import (
        PrefillSegment, bind_prefill_producers, compile_prefill_segments,
    )
    devices = tuple(map(int, args.devices.split(",")))
    if args.tp <= 0 or len(set(devices)) != len(devices) or len(devices) % args.tp:
        raise ValueError("unique devices must form complete TP groups")
    topology = SegmentTopology(tp=args.tp, dp=len(devices) // args.tp)
    args.group_counts = ([topology.capacity] * topology.dp if args.group_counts is None
                         else [int(n) for n in args.group_counts.split(",")])
    topology.counts(args.group_counts)
    torch.set_num_threads(4)
    _, moe_module = load_segment_modules(args.lib_root, topology)
    from models.deepseek_v4_1_flash import prefill_c2a_full as c2a
    plans, attention, moe, residual, mix = prepare(args, topology, c2a)
    bound = bind_prefill_producers(plans, attention)
    from pypto_serving.model.deepseek_v41.prefill_segment import PREFILL_ARGUMENTS
    dynamic = {"attention_epoch", "num_tokens"}
    for plan in plans:
        for name in PREFILL_ARGUMENTS[plan.mode]:
            if name not in dynamic:
                assert isinstance(bound[plan.layer_id][name], torch.Tensor), name
    print("REAL CHECKPOINT C2A PREPARATION PASS", flush=True)
    if args.prepare_only:
        return
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.composite import LayerState
    from pypto_serving.model.deepseek_v41.swa_segment import make_segment_worker
    from golden.validation import ratio_allclose
    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / "comparison.pt").exists():
        raise FileExistsError("refusing to overwrite a previous comparison")
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(devices)),
                       ring_heap=args.ring_heap_mib << 20, ring_task_window=131072, ring_dep_pool=131072)
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    programs, ffn = compile_prefill_segments(compiler, args.lib_root, topology, [p.mode for p in plans])
    ac = torch.zeros(topology.world, 1, dtype=torch.int32).share_memory_()
    mc = torch.zeros(topology.world, dtype=torch.int32).share_memory_()
    captured = {}
    for plan in plans:
        layer = plan.layer_id
        names = ("attn_input", "attn_output", "next_pre_mix", "x_hc_out") + c2a.ATTENTION_STATE[
            plan.mode.removeprefix("c2a_")]
        captured[layer] = ({n: torch.empty_like(bound[layer][n]).share_memory_() for n in names},
                          {n: torch.empty_like(moe[layer][n]).share_memory_()
                           for n in ("x_next", "next_pre_mix", "x_mixed")})
    sources = [v for maps in (bound, moe) for values in maps.values() for v in values.values()]
    with make_segment_worker([*programs.values(), ffn], config, sources) as worker:
        uploaded = {}
        def upload(values):
            result = {}
            for name, value in values.items():
                if name == "num_tokens":
                    continue
                if id(value) not in uploaded:
                    uploaded[id(value)] = worker.alloc_stacked_tensor(value)
                result[name] = uploaded[id(value)]
            return result
        da = {layer: upload(values) for layer, values in bound.items()}
        dm = {layer: upload(values) for layer, values in moe.items()}
        runner = PrefillSegment(worker, programs, ffn, topology, ac, mc, config)
        runner.run_chain(LayerState(da[2]["x_hc"], da[2]["pre_mix"], "tp_local_token"),
                         plans, da, dm, group_counts=args.group_counts)
        for layer, (ca, cm) in captured.items():
            for device, host in ((da[layer], ca), (dm[layer], cm)):
                for name, destination in host.items():
                    worker.copy_stacked_from(device[name], destination)
        for handle in reversed(list(uploaded.values())):
            worker.free_stacked_tensor(handle)
    # CPU same-input references run only after worker shutdown. Device state
    # never depends on these diagnostic readbacks or reference results.
    records, passed = [], True
    for plan in plans:
        layer, mode = plan.layer_id, plan.mode.removeprefix("c2a_")
        actual_a, actual_m = captured[layer]
        inputs = dict(bound[layer], x_hc=residual, pre_mix=mix)
        if mode == "reuse":
            for name in ("compressed_cache", "compressed_cache_scale"):
                inputs[name] = captured[2][0][name]
            inputs["compressed_indices"] = captured[2][0]["topk_indices"]
        initial_cache = {n: inputs[n].clone() for n in c2a.MODES[mode][1]}
        expected_a = dict(inputs)
        for name in actual_a:
            expected_a[name] = inputs[name].clone()
        c2a.make_golden(mode, 1)(expected_a)
        expected_m = dict(moe[layer], x_hc=actual_a["x_hc_out"], pre_mix=actual_a["next_pre_mix"])
        for name in actual_m:
            expected_m[name] = moe[layer][name].clone()
        moe_module.golden_moe(expected_m)
        mc_check = {
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "x_next": moe_module._local_mhc_compare(list(topology.counts(args.group_counts)[1])),
        }
        for label, actual, expected, checks in (("attention", actual_a, expected_a,
                c2a.make_compare(mode, 1, initial_cache)), ("moe", actual_m, expected_m, mc_check)):
            results = {}
            for name, check in checks.items():
                ok, detail = check(actual[name], expected[name], inputs=expected,
                    actual_outputs=actual, expected_outputs=expected, rtol=1e-3, atol=1e-3)
                print(f"NATIVE STAGE layer={layer} {label}.{name}: {ok} {detail}", flush=True)
                results[name] = (bool(ok), detail)
                passed &= bool(ok)
            records.append(dict(layer=layer, stage=label, results=results, actual=actual,
                                expected={n: expected[n] for n in actual}))
        residual, mix = actual_m["x_next"], actual_m["next_pre_mix"]
    torch.save({"input_state": str(args.input_state), "group_counts": args.group_counts, "stages": records,
                "actual_residual": residual, "actual_pre_mix": mix}, artifact / "comparison.pt")
    if not passed:
        raise AssertionError("C2A chain native stage check failed; see comparison.pt")
    print("C2A CHAIN NATIVE STAGES PASS; accumulated full-model acceptance remains pending", flush=True)


if __name__ == "__main__":
    main()
