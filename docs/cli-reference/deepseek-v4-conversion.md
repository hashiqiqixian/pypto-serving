# DeepSeek V4 Conversion

The repository-local `scripts/convert_deepseek_v4_to_w8a8.py` utility converts the DeepSeek V4 Flash source checkpoint variant validated by this repository to the W8A8 layout expected by PyPTO Serving.

The conversion can run on CPU and does not require `torch_npu`. The source and output directories must be different, and the host must have enough free disk space for both copies.

## Prepare Dependencies

```bash
python -m pip install --upgrade huggingface_hub safetensors
python -c "import torch, safetensors; print(torch.__version__)"
```

Use the PyTorch build already validated for the active Ascend environment.

## Download or Locate the Source Checkpoint

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /path/to/DeepSeek-V4-Flash
```

If an official mirror is already available locally, use that snapshot directory as `--input-dir`.

## Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` | yes | DeepSeek V4 Flash source checkpoint directory. |
| `--output-dir PATH` | yes | Output directory for the converted PyPTO W8A8 checkpoint. |
| `--resume` | no | Validate and skip completed output shards after an interrupted conversion. |
| `--dry-run` | no | Validate the source and print the conversion plan without writing files. |

## Dry Run

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /path/to/DeepSeek-V4-Flash \
  --output-dir /path/to/dsv4-flash-w8a8 \
  --dry-run
```

## Convert

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /path/to/DeepSeek-V4-Flash \
  --output-dir /path/to/dsv4-flash-w8a8
```

If the process is interrupted, rerun with `--resume`:

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /path/to/DeepSeek-V4-Flash \
  --output-dir /path/to/dsv4-flash-w8a8 \
  --resume
```

A successful run prints `Conversion complete` and leaves a converted `config.json`, `model.safetensors.index.json`, safetensors shards, and a `.pypto-w8a8-conversion.json` marker in the output directory.
