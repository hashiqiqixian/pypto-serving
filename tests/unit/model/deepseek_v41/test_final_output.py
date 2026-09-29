from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.final_output import FinalOutput


PARAMS = ("x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
          "normed", "logits", "sampled_ids", "done_epoch")


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def run(self, program, *args, config):
        self.calls.append((program, args, config))


def test_output_uses_final_state_and_advances_epoch():
    worker = RecordingWorker()
    output = FinalOutput(worker, SimpleNamespace(compiled="program"), PARAMS, "config")
    state = LayerState("final-residual", "final-mix", "tp_local_token")
    arguments = {name: object() for name in PARAMS if name not in ("x_hc", "pre_mix", "done_epoch")}

    for epoch in (1, 2):
        assert output.run(state, arguments) == (arguments["logits"], arguments["sampled_ids"])
        _, values, _ = worker.calls[-1]
        assert values[:2] == ("final-residual", "final-mix")
        assert values[-1].value == epoch


def test_output_rejects_missing_weights_before_launch():
    worker = RecordingWorker()
    output = FinalOutput(worker, SimpleNamespace(compiled="program"), PARAMS, "config")
    state = LayerState(object(), object(), "tp_local_token")
    with pytest.raises(ValueError, match="head_weight"):
        output.run(state, {"norm_weight": object()}, )
    assert worker.calls == []
