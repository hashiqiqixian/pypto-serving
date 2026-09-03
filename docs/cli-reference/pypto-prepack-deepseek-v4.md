# `pypto-prepack-deepseek-v4`

`pypto-prepack-deepseek-v4` builds the optional DeepSeek V4 hidden-layer weight sidecar. The sidecar stores the rank-stacked host layout consumed by the serving runner and can reduce repeated startup work on later launches.

Run it after converting a DeepSeek V4 Flash checkpoint to the W8A8 layout expected by PyPTO Serving.

## Usage

```bash
pypto-prepack-deepseek-v4 /path/to/dsv4-flash-w8a8
```

The default output path is the serving auto-discovery path beside the checkpoint.

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `model_dir` | Required | DeepSeek V4 W8A8 checkpoint directory. |
| `--ranks N` | `8` | Rank count for the packed layout. |
| `--output PATH` | auto-discovery path | Output sidecar path. |
| `--force` | off | Replace an existing sidecar. |

## Replace an Existing Sidecar

```bash
pypto-prepack-deepseek-v4 /path/to/dsv4-flash-w8a8 --force
```

Rebuild the sidecar after replacing checkpoint shards or changing the packed rank layout.

See [DeepSeek V4 Prepacked Weights](../user-guide/deepseek-v4.md#prepacked-weights) for runtime behavior after the sidecar is built.
