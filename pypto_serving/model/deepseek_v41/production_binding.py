# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production V4.1 binding of lib composites to one persistent worker."""

import importlib
import json
import logging
import os
from pathlib import Path
import subprocess

from .cache_contract import build_cache_groups
from .cache_resources import (
    make_cache_page_reset_buffers, make_state_reset_buffer, plan_decode_cache_pools,
)
from .composite import CompositeBindings, MissingCompositeInterface
from .decode_backbone import DecodeLayerSegment, compile_decode_layer
from .decode_weight_resources import DecodeWeightBank, DecodeWeightLayout
from .decode_weights import iter_decode_weight_slices
from .execution_plan import plan_layers
from .final_output import compile_final_output
from .final_output_resources import load_final_output_host_resources
from .input_resources import make_input_host_resources
from .prefill_metadata_resources import PrefillMetadataResources
from .prefill_segment import PrefillSegment, compile_prefill_segments
from .resident_resources import V41ResidentResources
from .swa_segment import SegmentTopology


_MODES = ("swa", "c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse")
_LOG = logging.getLogger(__name__)


def _lib_root():
    configured = os.environ.get("PYPTO_LIB_ROOT")
    root = Path(configured) if configured else Path.cwd() / "pypto-lib"
    root = root.resolve()
    if not (root / "models/deepseek_v4_1_flash/final_output.py").is_file():
        raise MissingCompositeInterface(
            f"V4.1 final-output composite is absent under {root}; set PYPTO_LIB_ROOT "
            "to a lib checkout containing the verified output entry")
    return root


def _revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()


def _sources(layers):
    return (
        tuple(layer.layer_id for layer in layers if layer.kv_source == layer.layer_id),
        tuple(layer.layer_id for layer in layers if layer.index_source == layer.layer_id),
        tuple(layer.layer_id for layer in layers if layer.mode == "c2a_full"),
    )


def _check_runtime_binaries(platform):
    from simpler_setup.runtime_builder import RuntimeBuilder

    builder = RuntimeBuilder(platform=platform)
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
        builder.get_binaries(runtime)


class _CompositeCompiler:
    def __init__(self, compiler, use_cache):
        self.compiler = compiler
        self.use_cache = use_cache

    def compile(self, name, entry, **kwargs):
        return self.compiler.compile(name, entry, use_cache=self.use_cache, **kwargs)


def make_production_bindings(model_dir, max_seq_len):
    model_dir = Path(model_dir).resolve()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    groups = build_cache_groups(layers, max_seq_len)
    root = _lib_root()
    revision = _revision(root)

    def allocate(plan, device_ids, runtime, options):
        from pypto.ir import DistributedConfig
        from pypto.runtime import RunConfig

        from pypto_serving.model.common.compiler.compiler import KernelCompiler

        if Path(plan.weights.model_dir).resolve() != model_dir or tuple(plan.layers) != layers:
            raise ValueError("V4.1 binding and execution plan refer to different checkpoints")
        if runtime.max_seq_len != max_seq_len or runtime.kv_cache_groups != groups:
            raise ValueError("V4.1 scheduler and resident cache contracts differ")
        if len(device_ids) != 8 or options.platform != "a5":
            raise ValueError("V4.1 production binding requires eight A5 devices")
        _check_runtime_binaries(options.platform)
        topology = SegmentTopology.for_decode()
        run_config = RunConfig(
            platform=options.platform,
            distributed_config=DistributedConfig(device_ids=list(device_ids)),
            ring_heap=1024 * 1024 * 1024,
        )
        compiler = _CompositeCompiler(
            KernelCompiler(run_config=run_config, cache_dir=Path(options.pypto_build_dir)),
            options.use_compile_cache,
        )
        attention, moe = compile_prefill_segments(compiler, root, topology, _MODES)
        decode, decode_names = compile_decode_layer(compiler, root, topology)
        final, final_names = compile_final_output(compiler, root, topology)
        abi = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
        kv_sources, index_sources, c2a_sources = _sources(layers)
        layout = DecodeWeightLayout.from_lib(abi)
        if (layout.layers, layout.kv_sources, layout.index_sources,
                layout.c2a_sources, layout.world) != (
                    len(layers), len(kv_sources), len(index_sources), len(c2a_sources), 8):
            raise ValueError("V4.1 checkpoint source axes disagree with the selected lib")
        DecodeWeightBank.validate_abi(decode_names)

        bank = DecodeWeightBank(layout)
        for item in iter_decode_weight_slices(model_dir, topology):
            bank.append(item)
            if item.name == "hc_attn_fn":
                _LOG.info("V4.1 checkpoint layer %d/%d loaded into shared weight bank",
                          item.slot + 1, layout.layers)
        bank.finish(decode_names)
        _LOG.info("V4.1 shared weight bank ready: %.1f GiB",
                  bank.allocated_bytes / (1 << 30))

        rope = importlib.import_module("models.deepseek_v4_1_flash.rope_tables")
        swa_tables = rope.precompute_rope_tables(max_seq_len, False)
        compressed_tables = rope.precompute_rope_tables(max_seq_len, True)
        cache_plan = plan_decode_cache_pools(
            layers, groups, max_seq_len=max_seq_len,
            primary_num_blocks=groups[0].max_blocks_per_seq,
            request_slots=1, abi=abi,
        )
        if (runtime.total_kv_pages is not None and
                runtime.total_kv_pages < cache_plan.primary_num_blocks):
            raise ValueError("configured cache pages cannot hold one V4.1 request")
        resets = make_cache_page_reset_buffers(cache_plan)
        state_reset = make_state_reset_buffer(cache_plan)
        family = cache_plan.family_groups
        host_kwargs = dict(
            max_seq_len=max_seq_len,
            max_requests=cache_plan.physical_request_slots,
            window_pages=cache_plan.physical_group_blocks[family.window],
            c2a_pages=cache_plan.physical_group_blocks[family.c2a],
            c1a_pages=cache_plan.physical_group_blocks[family.c1a],
            rope_tables=swa_tables,
            compressed_rope_tables=compressed_tables,
        )
        prefill_host = PrefillMetadataResources(topology, **host_kwargs)
        decode_host = PrefillMetadataResources(topology, **host_kwargs)
        input_host = make_input_host_resources(topology, plan.weights.config.hidden_size)
        decode_input_host = make_input_host_resources(topology, plan.weights.config.hidden_size)
        final_host = load_final_output_host_resources(model_dir, topology)
        programs = (*attention.values(), moe, decode, final)
        resident = V41ResidentResources(
            programs, run_config, weight_bank=bank, weight_param_names=decode_names,
            cache_plan=cache_plan, state_reset=state_reset, page_resets=resets,
            input_host=input_host, decode_input_host=decode_input_host,
            prefill_host=prefill_host, decode_host=decode_host,
            final_host=final_host, final_program=final, final_param_names=final_names,
            topology=topology,
        )
        try:
            resident.bind_prefill(
                layers, kv_sources=kv_sources, index_sources=index_sources,
                c2a_sources=c2a_sources, max_seq_len=max_seq_len,
            )
            resident.bind_decode_layers(
                layers, decode_names, kv_sources=kv_sources,
                index_sources=index_sources, c2a_sources=c2a_sources,
            )
            resident.prefill_segment = PrefillSegment(
                resident.worker, attention, moe, topology,
                prefill_host.fields["attention_counts"], prefill_host.fields["moe_counts"],
                run_config,
            )
            resident.decode_segment = DecodeLayerSegment(
                resident.worker, decode, decode_names, run_config, topology,
                decode_host.fields["moe_counts"],
            )
        except BaseException:
            resident.close()
            raise
        return resident, cache_plan.primary_num_blocks

    return CompositeBindings(
        revision=revision,
        entries={},
        initialize=lambda embeddings, step, resources: resources.begin_step(embeddings, step),
        output=lambda state, step, resources: resources.finish_step(state, step),
        allocate=allocate,
        prepare_weights=lambda plans, layer, resources: None,
        reset_request=lambda resources, key, owner: resources.reset_request(owner),
        wait=lambda resources: None,
        close=lambda resources: resources.close(),
        cache_groups=groups,
        input_layout="tp_local_token",
        output_layout="tp_local_token",
        prefill_backbone=lambda state, step, resources, plans: resources.run_prefill(
            resources.prefill_segment, state, step),
        decode_backbone=lambda state, step, resources, plans: resources.run_decode_layers(
            resources.decode_segment, state, step),
    )
