"""Capture independent graph and task clocks from separate real decode rounds."""
from dataclasses import replace
import json
import os
from pathlib import Path


def install():
    from pypto.runtime import distributed_runner
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources

    capture_root = Path(os.environ["DEVICE_CAPTURE_ROOT"])
    original_batch = WorkerProcess._batch_decode
    original_dispatch = V41ResidentResources.dispatch_dspark
    original_config = distributed_runner._make_call_config
    counts = {}
    request = None
    selected = None

    def batch(self, scheduled, *args, **kwargs):
        nonlocal request
        request = scheduled[0].request_id if len(scheduled) == 1 else None
        try:
            return original_batch(self, scheduled, *args, **kwargs)
        finally:
            request = None

    def dispatch(self, stage, arguments):
        nonlocal selected
        if request and "-pass0-" in request and stage == "decode":
            counts[request] = number = counts.get(request, 0) + 1
            if 4 <= number <= 7:
                selected = (request, number)
        try:
            return original_dispatch(self, stage, arguments)
        finally:
            selected = None

    def config(dc, run_config=None, **kwargs):
        if selected and run_config is not None:
            case, number = selected
            dest = capture_root / case / str(number)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "source.json").write_text(json.dumps(str(kwargs["dfx_base"])))
            kwargs.update(dfx_base=dest, co_enable_swimlane_dep_gen=False)
            run_config = replace(run_config, enable_dep_gen=number == 4,
                                 enable_chip_swimlane=int(number > 4))
        return original_config(dc, run_config, **kwargs)

    WorkerProcess._batch_decode = batch
    V41ResidentResources.dispatch_dspark = dispatch
    distributed_runner._make_call_config = config
