"""Compare two independently compiled prefill capacities in one benchmark worker."""
import json

from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.deepseek_v41 import npu_runner as model


def install(path):
    params = json.loads((path / "distributed_meta.json").read_text())["params"]
    shapes = {p["name"].split("__ssa")[0]: p["shape"] for p in params}
    world, capacity, hidden = shapes["embedding"]
    init_metadata = model.PrefillMetadataResources.__init__
    init_resident = model.V41ResidentResources.__init__
    begin = model.V41ResidentResources.begin_step
    close = model.V41ResidentResources.close

    def metadata(self, topology, **kwargs):
        init_metadata(self, topology, **kwargs)
        if kwargs.get("full_hidden_size") and topology.local_capacity > capacity:
            assert topology.world == world and kwargs["full_hidden_size"] == hidden
            small = object.__new__(model.PrefillMetadataResources)
            small_topo = model.SegmentTopology.for_prefill(dp=topology.dp, max_tokens=capacity)
            init_metadata(small, small_topo, **kwargs)
            self.short_chunk_host = small

    def resident(self, programs, config, **kwargs):
        large_host = kwargs["prefill_host"]
        small_host = large_host.short_chunk_host
        assert tuple(shapes) == tuple(kwargs["prefill_param_names"])
        compiled = DistributedCompiledProgram.from_dir(
            path, backend_type=config.backend_type, execution_mode=config.execution_mode,
            distributed_config=config.distributed_config)
        handle = L3Callable(compiled, "v41_prefill_tail", config.distributed_config.aicpu_thread_num)
        original_factory = kwargs.get("worker_factory", model.make_segment_worker)
        def create_worker(programs, config, inherited):
            return original_factory(programs, config, inherited + small_host.inherited_host_tensors)
        kwargs["worker_factory"] = create_worker
        init_resident(self, (*programs, handle), config, **kwargs)
        self._bench_large_prefill = (large_host, self._prefill_fwd)
        try:
            small_host.window_pages = large_host.window_pages
            small_host.window_scratch_base = large_host.window_scratch_base
            small_host.compressed_pages = large_host.compressed_pages.copy()
            segment = model.PrefillForwardSegment(
                self.worker, handle, tuple(shapes), config, small_host.topology,
                self.weights, small_host, self.cache, self.final, self.engram)
            self._bench_small_prefill = (small_host, self._own(segment))
            self._bench_prefill_counts = [0, 0]
        except BaseException:
            self.close()
            raise

    def begin_step(self, embeddings, step):
        if step.phase == "prefill":
            counts = [0] * world
            for request in step.requests:
                counts[request.partition] += len(request.token_ids)
            small = max(counts) <= capacity
            self.prefill_host, self._prefill_fwd = self._bench_small_prefill if small else self._bench_large_prefill
            self._bench_prefill_counts[int(small)] += 1
        return begin(self, embeddings, step)

    def close_resident(self):
        if hasattr(self, "_bench_large_prefill"):
            self.prefill_host, self._prefill_fwd = self._bench_large_prefill
            if hasattr(self, "_bench_prefill_counts"):
                print("PREFILL_GRAPH_DISPATCH full,short", self._bench_prefill_counts, flush=True)
        return close(self)

    model.PrefillMetadataResources.__init__ = metadata
    model.V41ResidentResources.__init__ = resident
    model.V41ResidentResources.begin_step = begin_step
    model.V41ResidentResources.close = close_resident
