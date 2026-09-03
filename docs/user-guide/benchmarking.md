# Benchmarking

Benchmark PyPTO Serving at two levels:

- Offline generate mode with `pypto-serving --prompt`, which measures generation without HTTP request overhead.
- HTTP serving mode, which includes request handling, scheduler behavior, worker dispatch, and model execution under client load.

Use profiling when you need to explain where time is spent. Use benchmarking when you need externally visible latency or throughput numbers.

## What to Record

Record the following with every result:

- Model and checkpoint path.
- Platform and device IDs.
- Full command line.
- Prompt length and generated token count.
- Sampling settings.
- Stream mode.
- `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens`.
- Whether profiling was enabled.
- Whether compile cache or DeepSeek V4 prepacked weights were used.

Separate first-run startup cost from steady-state generation. Kernel compile and checkpoint packing work can dominate first launch results.

## Offline Generate Benchmark

Offline generate mode uses the same serving engine, scheduler, worker process, executor, and KV cache path as HTTP serving, but does not open an HTTP port.

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens":128,"temperature":0.0}' \
  --profile \
  --profile-output /tmp/pypto-qwen-offline
```

The CLI prints generated text, token IDs, finish reason, total generated token count, elapsed generation time, and overall tokens per second.

Repeat `--prompt` to schedule multiple offline requests through the same engine. Use the model guide for model-specific device topology and runtime requirements, especially for DeepSeek V4.

## HTTP Serving Benchmark

Start the server with the model, device, and capacity settings you want to measure:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --port 8899
```

Drive load with a repeatable HTTP client that records request rate or concurrency, end-to-end latency, output token throughput, and error rate. For streaming requests, record time to first token separately from full response latency.

Keep workload inputs explicit. Small changes in prompt length, output length, sampling settings, stream mode, chunked prefill, MTP, or `--max-num-seqs` can change results materially.

## Profile a Benchmark Window

Launch with profiling enabled:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --port 8899 \
  --profile \
  --profile-output /tmp/pypto-profile \
  --profile-level e2e,kernel
```

Start profiling, run the benchmark workload, then stop profiling:

```bash
curl --noproxy "*" -X POST http://127.0.0.1:8899/start_profile
# Run the benchmark workload.
curl --noproxy "*" -X POST http://127.0.0.1:8899/stop_profile
```

Open `/tmp/pypto-profile/trace.json` in Perfetto or another Chrome trace viewer.

Inspect prefill spans, decode step duration and variance, scheduler gaps, worker or executor spans, and kernel dispatch spans. If kernel spans are missing, first check that `--profile-level` includes `kernel` and that the profile window covers the workload.
