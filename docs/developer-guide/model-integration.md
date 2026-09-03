# Model Integration

Adding a model is a model-integration task, not just a config-file change. Each model needs a loader, executor, runner, examples, and tests that match its kernel layout.

## Checklist

- Define model-family detection from `config.json`.
- Implement or extend model loading and tokenizer handling.
- Define `RuntimeConfig` and KV cache requirements.
- Define weight-staging rules under `pypto_serving/model/<family>/weight_spec.py`
  when fused kernels consume whole-model slabs.
- Implement the NPU executor and runner.
- Add PyPTO kernel sources or point to existing kernels in `pypto-lib/`.
- Add offline generation example commands.
- Add HTTP serving topology validation in the CLI path.
- Add unit tests for config and scheduler behavior.
- Add NPU validation or accuracy checks for generated output.
- Document the model under `docs/user-guide/`.

## Topology

Be explicit about the supported device topology before exposing a new model in the CLI. The current code has two patterns:

- Qwen-style replica placement.
- DeepSeek V4-style overlapped placement.
