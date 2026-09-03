# PyPTO Serving

PyPTO Serving is a local inference stack for running selected large language models with PyPTO kernels on Ascend NPUs. It provides offline generation tools, a small OpenAI-compatible HTTP server, model-specific NPU executors, and profiling hooks for understanding host and kernel time.

The project is intentionally focused. The current external surface is for Qwen3-14B and DeepSeek V4 Flash W8A8 inference on Ascend NPU environments.

## Features

- Ascend NPU backend with PyPTO kernel execution.
- Offline generation for model validation and local inference.
- OpenAI-compatible HTTP API subset for completions, chat completions, model listing, health checks, and server-side streaming.
- Qwen3-14B single-device serving, tensor-parallel offline execution, and data-parallel online replicas.
- DeepSeek V4 Flash W8A8 eight-device execution with overlapped attention DP=8 and MoE EP=8.
- Continuous batching, paged KV cache management, chunked prefill, prefix caching for supported models, and DeepSeek V4 MTP speculative decoding.
- Chrome Trace Event Format profiling across the HTTP API, scheduler, engine, worker, executor, and NPU dispatch paths.

## Start Here

- [Installation](get-started/installation.md): clone, initialize submodules, prepare the runtime, install, and verify the CLI.
- [Quickstart](get-started/quickstart.md): run Qwen3-14B offline and start the HTTP server.
- [Online Serving](user-guide/online-serving.md): use the HTTP API subset.
- [CLI Reference](cli-reference/index.md): understand command-line arguments and runtime environment variables.

## Documentation Map

- **User Guide** explains how to install, run offline inference, serve HTTP traffic, scale across devices, benchmark, profile, and tune runtime behavior.
- **CLI Reference** documents installed commands, repository utilities, command-line arguments, and runtime environment variables.
- **Developer Guide** explains the serving architecture, model integration, runtime internals, and contribution workflow.
