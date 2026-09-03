# Installation

Install PyPTO Serving from a source checkout. The package does not vendor the Ascend runtime, PyPTO runtime, PyTorch, or model weights; those must be provided by the active environment.

## Before You Start

Prepare a Linux host with Ascend NPUs visible to the serving user, an Ascend-compatible Python 3.10+ environment, CANN, PyPTO runtime pieces, PyTorch, and the model checkpoint you plan to run.

Qwen3-14B expects a local Hugging Face style checkpoint directory containing `config.json`, tokenizer files, and weight shards. DeepSeek V4 serving expects a converted W8A8 compressed-tensors checkpoint and exactly eight device IDs with `--dp 8 --ep 8 --tp 1`.

## Clone

Clone the repository and initialize the kernel submodule:

```bash
git clone https://github.com/hw-native-sys/pypto-serving.git
cd pypto-serving
git submodule update --init --recursive
```

## Install the Python Package

Install the package in editable mode:

```bash
python -m pip install --no-deps -e .
```

`--no-deps` is intentional. The project expects the Python environment to already contain the Ascend-compatible PyTorch build, PyPTO runtime pieces, and serving dependencies that match the target machine.

For HTTP serving, make sure the environment also has:

```bash
python -m pip install fastapi uvicorn pydantic
```

For model conversion and checkpoint loading, make sure `safetensors` and `transformers` are available:

```bash
python -m pip install safetensors transformers
```

## Verify the Install

After installation, confirm the CLI is available:

```bash
pypto-serving --help
```

Then run the Qwen quickstart once the model checkpoint is available:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 5}'
```
