# Qwen3-14B

PyPTO Serving supports Qwen3-14B through the bundled Qwen model loader, NPU executor, and PyPTO kernels. Use this path for single-device validation, offline tensor-parallel runs, and HTTP serving with one or more data-parallel replicas.

## Checkpoint

Use a local Hugging Face style Qwen3-14B checkpoint directory. The directory must contain `config.json`, tokenizer files, and model weight shards readable by the active Python environment.

## Offline Generation

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 32}'
```

For one tensor-parallel worker group, provide `--devices` and `--tp`:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1 \
  --tp 2 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 32}'
```

Offline generate mode shares the serving engine. `--dp` creates independent replica engines, and repeated `--prompt` values are routed across them by the `least_pending_tokens` policy.

## HTTP Serving

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Send a request after startup:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

## DP=2 Serving

Data parallel serving creates independent replicas and routes requests by the `least_pending_tokens` policy:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --devices 0,1 \
  --dp 2 \
  --tp 1 \
  --max-model-len 512 \
  --port 8899
```

## Weight Staging

Layer weights are described in `pypto_serving/model/qwen/weight_spec.py` and staged by the shared pipeline documented in [Weight Staging](../developer-guide/weight-staging.md).

The loader reads metadata only for the per-layer weights. Each layer is read, written into its slab slice, and dropped before the next one, which keeps the staging peak at roughly one layer per worker instead of a second copy of the model. The globals stay eager because `Executor.lookup_embeddings` reads `embed_tokens` at request time.

Qwen-specific details: layers stack on axis 0 because there is no rank axis, projections are stored transposed so every projection rule carries `transpose=True`, and slabs are allocated in shared memory because the upload reads them from a forked child. A checkpoint without QK norms is a supported variant; the rules default those gammas to ones.

## Runtime Notes

Qwen uses the standard `pypto-serving` runtime capacity, generation, prefix caching, chunked prefill, and compile-cache controls. See [pypto-serving](../cli-reference/pypto-serving.md) for the command-line reference.

Prefix caching and chunked prefill are enabled by default for Qwen serving.
