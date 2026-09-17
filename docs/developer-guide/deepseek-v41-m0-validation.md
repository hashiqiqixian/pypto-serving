# DeepSeek V4.1 M0 evidence collection

The collector supports separate `a2a3` and `a5` reports. Selecting a platform
does not establish native backend support or successful hardware execution.
Use the real checkpoint and eight devices for each report.

## Independent reference and complete serving token traces

Prepare `reference.json` from the independent DeepSeek reference at revision
`dba1be0a40aa45a94ad051997016db3960a90277`. Do not generate the expected IDs with
the PyPTO implementation under test. The JSON object contains:

- `reference_revision`: the pinned revision above.
- `checkpoint_revision`: the checkpoint revision used by both executions.
- `config_sha256` and `tokenizer_sha256`: lowercase SHA-256 hashes of the local
  `config.json` and `tokenizer.json` files used by both executions.
- `cases`: 1–16 objects, each with a unique `case_id`, exactly 8192
  `prompt_token_ids` and exactly 128 independently generated `token_ids`.

Use at least two different cases (A and B) to exercise a different request
between repeats. The collector runs A, B, …, A using the production engine,
pretokenized prompts and target-only greedy decoding. It records all generated
IDs, actual scheduler phase/token counts and host request cleanup. Early EOS
before 128 IDs fails the requested workload; it is not padded or counted as a
128-token run. The first differing output position is reported.

From the repository root on the prepared device host:

```bash
python pypto_serving/tools/validate_deepseek_v41_m0.py \
  --platform a5 --reference artifacts/reference.json \
  --output artifacts/a5-engine-report.json --run -- \
  --model /path/to/DeepSeek-V4.1-Flash \
  --platform a5 --devices 0,1,2,3,4,5,6,7 --dp 1 --tp 8 --ep 8 \
  --max-model-len 9216 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 8192 --no-enable-prefix-caching
```

For A3 use `a2a3` in both platform arguments and its supported backend
configuration. Backend rejection is recorded as a failed execution, never as
an inferred success. The result embeds the full `actual_trace`; pass this report
directly as `--actual` when combining it with other evidence.
The recorded checkpoint revision is a declaration from the reference;
config/tokenizer hashes are read locally. The metadata inspection described in
`deepseek-v41-p0.md` supplies additional checkpoint identity/coverage evidence.

## Consecutive HTTP requests

Start the normal HTTP server with the same platform, model and runtime flags.
Create a JSONL workload containing A and B as `{"prompt":"..."}` or
`{"messages":[...]}` objects. Each fully formatted HTTP prompt must produce
8192 tokens according to the server's authoritative usage. HTTP formatting may
add tokens, so character counts or offline text length are insufficient.

Record `server-manifest.json` with the actual server's `platform`, `world_size`
(8), `checkpoint_revision`, `config_sha256` and `tokenizer_sha256`, and retain
the corresponding server startup/device logs. Run:

```bash
python pypto_serving/tools/benchmark_http.py \
  --endpoint http://127.0.0.1:8899/v1/completions \
  --model DeepSeek-V4.1-Flash --prompts artifacts/prompts.jsonl \
  --server-manifest artifacts/server-manifest.json \
  --requests 3 --concurrency 1 --max-tokens 128 \
  --expect-prompt-tokens 8192 --expect-completion-tokens 128 --check-repeats \
  --output artifacts/a5-http.json
```

The benchmark requires successful stream termination and usage. It records
unique request IDs and hashes of the complete text/reasoning channels,
independent of SSE fragmentation. A/B/A must reproduce A's digest, usage and
finish reason. These observations test HTTP repeatability; HTTP text is not
retokenized to claim token-ID equality or device cache isolation.

## Combine and interpret evidence

```bash
python pypto_serving/tools/validate_deepseek_v41_m0.py \
  --platform a5 --reference artifacts/reference.json \
  --actual artifacts/a5-engine-report.json --inspection artifacts/checkpoint-inspection.json \
  --http-artifact artifacts/a5-http.json --output artifacts/a5-m0-report.json
```

Offline comparison imports no accelerator runtime. Input artifacts are bounded
to 64 MiB each, hashed in the report and protected from output overwrites.
The gate recomputes token comparisons, measured metadata coverage and HTTP
request checks; a standalone `PASS` or `passed: true` does not satisfy them.

Exit code 1 means a provided check failed. Exit code 2 means no provided check
failed but M0 remains incomplete. This collector does not issue full M0 PASS:
real weight conversion/layer golden, device cache rollback/reuse, per-rank
peak HBM, backend/toolchain provenance and mismatch logits remain `NOT_RUN`
pending their independent device/log artifacts and review. Host request cleanup
and repeated HTTP responses do not substitute for those observations. No device
result is implied by the synthetic unit tests for these tools.
