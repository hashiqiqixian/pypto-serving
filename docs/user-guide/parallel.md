# Parallelism and Scaling

PyPTO Serving supports two placement modes: replica placement for standard models and overlapped placement for DeepSeek V4. Single-device serving remains the default.

## Replica Placement

Replica placement is used by Qwen. Data parallelism creates independent serving replicas. Tensor parallelism passes one device group to the PyPTO L3 distributed worker for each replica.

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

The server routes requests across replicas using `least_pending_tokens`.

## Offline Generate Runs

Offline generation uses `pypto-serving --prompt`, the same engine as HTTP serving. For one tensor-parallel worker group:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1 \
  --tp 2 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 16}'
```

Repeat `--prompt` to schedule multiple prompts. `--devices`, `--tp`, and `--dp` behave exactly as in serving: prompts are routed across replicas with the same least-pending-tokens policy.

## DeepSeek V4 Overlapped Placement

DeepSeek V4 uses a model-local overlapped placement. Its attention DP ranks and MoE EP ranks reuse the same eight physical devices.

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --served-model-name dsv4-flash-w8a8 \
  --backend npu \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 2048 \
  --no-enable-prefix-caching \
  --port 8225
```
