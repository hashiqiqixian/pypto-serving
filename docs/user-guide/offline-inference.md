# Offline Inference

Offline inference runs `pypto-serving --prompt` without opening an HTTP port. Use it for model validation, kernel checks, profiling a single workload, or running batch generation from a shell.

## Qwen3-14B

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 32}'
```

See [pypto-serving](../cli-reference/pypto-serving.md) for offline mode, parallel placement, generation, profiling, and prefix-cache options.

## DeepSeek V4

DeepSeek V4 offline inference requires the converted W8A8 checkpoint and eight devices:

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --long-prefill-token-threshold 2048 \
  --no-enable-prefix-caching \
  --generate-config '{"max_new_tokens": 32}'
```

Add `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to enable one-token MTP speculative decoding. Use `--profile --profile-output /path/to/profile` to capture a trace for the generation window.

## Output

Offline generate mode prints generated text, token IDs, finish reason, and a concise throughput summary. If startup fails before generation, check the model path, NPU visibility, CANN environment, and PyPTO kernel checkout first.
