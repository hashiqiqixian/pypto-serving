"""Bind the checkpoint's native MXFP8 drafter projection for a newer recurrent graph."""
import json
import os
from pathlib import Path

from pypto_serving.model.common.weights.nz import pack_nz
from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources
from pypto_serving.model.deepseek_v41.weight_spec import fp8_input_major


def install():
    original_init = V41ResidentResources.__init__
    original_dispatch = V41ResidentResources.dispatch_dspark
    artifact = json.loads(Path(os.environ["DECODE_MANIFEST"]).read_text())["decode"]
    metadata = json.loads((Path(artifact) / "distributed_meta.json").read_text())
    names = tuple(p["name"].split("__ssa")[0] for p in metadata["params"])

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        existing = self.dspark_param_names["decode"]
        if tuple(k for k in names if k != "draft_main_proj_scale") != tuple(existing):
            raise ValueError("unexpected recurrent graph parameter order")
        weight, scale = "mtp.0.main_proj.weight", "mtp.0.main_proj.scale"
        source = self.dspark_weight_contract.loader.store.load_many((weight, scale))
        w, s = fp8_input_major(source[weight], source[scale])
        self._bench_native_proj = {}
        for key, value in (("draft_main_proj_weight", pack_nz(w)), ("draft_main_proj_scale", s)):
            ranked = value.unsqueeze(0).expand(self.prefill_host.topology.world, *value.shape)
            self._bench_native_proj[key] = self.worker.alloc_stacked_tensor(ranked.contiguous().share_memory_())

    def dispatch(self, stage, arguments):
        if stage != "decode":
            return original_dispatch(self, stage, arguments)
        previous = self.dspark_param_names[stage]
        self.dspark_param_names[stage] = names
        try:
            return original_dispatch(self, stage, arguments | self._bench_native_proj)
        finally:
            self.dspark_param_names[stage] = previous

    V41ResidentResources.__init__ = initialize
    V41ResidentResources.dispatch_dspark = dispatch
