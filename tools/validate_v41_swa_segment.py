# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synthetic A5 smoke of serving SWA -> MoE -> next-layer SWA/MoE.

This is not real-checkpoint validation or numerical acceptance. Lib fixture
builders are used only here, never by the serving segment implementation.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--build-dir", default="build_output/v41-swa-segment")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.lib_root).resolve()))
    import torch
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig
    from golden.spec import TensorSpec
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.composite import LayerState
    from pypto_serving.model.deepseek_v41.swa_segment import (
        SegmentTopology, SwaSegment, compile_segment, load_segment_modules,
    )

    torch.set_num_threads(4)
    devices = [int(v) for v in args.devices.split(",")]
    if len(devices) % args.tp or len(set(devices)) != len(devices):
        raise ValueError("unique devices must form complete TP groups")
    topology = SegmentTopology(tp=args.tp, dp=len(devices) // args.tp)
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=devices),
                       ring_heap=536870912, ring_task_window=131072, ring_dep_pool=131072)
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    programs = compile_segment(compiler, args.lib_root, topology)
    print("COMPILE PASS", flush=True)
    if args.compile_only:
        return
    swa, moe = load_segment_modules(args.lib_root, topology)
    fixture = SimpleNamespace(tokens=topology.capacity, requests=1, dp=topology.dp,
                              seed=11, case="normal", fixture="checkpoint", dp_tokens=None,
                              epochs=1, bench=False)

    def materialize(specs):
        return {s.name: s.create_tensor().contiguous() for s in specs if isinstance(s, TensorSpec)}

    a = materialize(swa.build_hc_specs(fixture))
    m = materialize(moe.build_tensor_specs([16] * topology.world))
    ac = torch.zeros(topology.world, 1, dtype=torch.int32).share_memory_()
    mc = torch.zeros(topology.world, dtype=torch.int32).share_memory_()
    readback = torch.empty_like(m["x_next"]).share_memory_()
    mix_readback = torch.empty_like(m["next_pre_mix"]).share_memory_()
    sources = [*a.values(), *m.values()]
    with DistributedWorker([p.compiled for p in programs], persistent=True,
                           inherited_host_tensors=sources, config=config) as worker:
        allocations = []

        def upload(values):
            result = {}
            for name, value in values.items():
                if name == "num_tokens":
                    continue
                device = worker.alloc_stacked_tensor(value)
                allocations.append(device)
                result[name] = device
            return result

        da, dm = upload(a), upload(m)
        state = LayerState(da.pop("x_hc"), da.pop("incoming_pre_mix"), "tp_local_token")
        dm.pop("x_hc")
        dm.pop("pre_mix")
        # Diagnostic uses tied synthetic weights but independent layer caches.
        second_a = dict(da)
        for name in ("window_cache", "window_cache_scale"):
            second_a[name] = worker.alloc_stacked_tensor(a[name])
            allocations.append(second_a[name])
        runner = SwaSegment(worker, programs, topology, ac, mc, config)
        first = runner.run_layer(state, da, dm, group_counts=[topology.capacity] * topology.dp)
        second_m = dict(dm, x_next=state.residual, next_pre_mix=state.pre_mix)
        final = runner.run_layer(first, second_a, second_m,
                                 group_counts=[topology.capacity] * topology.dp)
        # Read back only after the full chain; no intermediate host round trip.
        worker.copy_stacked_from(final.residual, readback)
        worker.copy_stacked_from(final.pre_mix, mix_readback)
        assert torch.isfinite(readback).all() and torch.isfinite(mix_readback).all()
        assert readback.abs().max() > 0 and mix_readback.abs().max() > 0
        print("DEVICE TWO-LAYER SMOKE PASS (finite/nonzero only; not numerical acceptance)", flush=True)
        for value in reversed(allocations):
            worker.free_stacked_tensor(value)


if __name__ == "__main__":
    main()
