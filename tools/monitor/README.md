# PyPTO Monitor

PyPTO Monitor is a local companion dashboard for PyPTO Serving. It polls the
structured metrics endpoint, keeps detailed recent samples and daily totals in
SQLite, and serves a browser dashboard without Prometheus, Grafana, or external
web assets.

Start PyPTO Serving normally, then launch the monitor from the repository root:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --port 9090
```

Open <http://127.0.0.1:9090>. The dashboard listener defaults to
`127.0.0.1`, so it is not reachable from other hosts unless `--host` is changed.

The default database is
`~/.local/state/pypto-serving/monitor.sqlite3`. Override it when running in a
container or when history needs to live on a persistent volume:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --database /var/lib/pypto-monitor/metrics.sqlite3 \
  --timezone Asia/Shanghai
```

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--interval` | `1.0` | Serving metrics polling interval in seconds |
| `--timeout` | `2.0` | Per-poll HTTP timeout in seconds |
| `--retention-hours` | `24` | Detailed time-series retention |
| `--timezone` | `local` | Timezone used for daily totals |

PyPTO Serving exposes two metrics representations:

- `/metrics` is Prometheus-compatible text for standard collectors.
- `/metrics/json` is the versioned structured interface used by this tool.

## vLLM Tool Compatibility

`/metrics` also exports vLLM V1 metric families, with `model_name` and `engine`
labels. Existing vLLM Prometheus queries and the upstream
`examples/observability/prometheus_grafana/grafana.json` dashboard can target
this endpoint directly. Compatibility was checked against vLLM `69db1c26b4`.
Install `pypto_serving/requirements.txt`, which includes `prometheus-client`.

For an existing Prometheus instance, add the serving address to its scrape
configuration (use an address reachable from the Prometheus host/container):

```yaml
scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets: ['serving-host:8899']
```

The compatible families cover queues, KV cache usage, prefix-cache hits,
prompt/output traffic, request outcomes and token lengths, TTFT, ITL, TPOT,
request queue/prefill/decode/inference latency, iteration token counts, and
speculative decoding. Finish labels use `stop`, `length`, `abort`, and `error`;
EOS and stop strings both map to `stop`. Histogram buckets match vLLM, including
the model-length-dependent token buckets. Unavailable phases, such as prefill
for a request aborted before scheduling, have no observations.

`vllm:prompt_tokens_total` records the full prompt (including cache hits) when
its first output arrives. `vllm:prefix_cache_queries_total` counts all queried
prompt tokens, including a partial final block. `vllm:inter_token_latency_seconds`
records one interval per output batch, including MTP/DSpark batches; per-request
TPOT divides decode time by output tokens minus one. Phase timings are engine
wall times, including queueing or preemption within the phase, not kernel timings.

MTP/DSpark use the upstream speculative metric names and zero-based `position`
labels on `vllm:spec_decode_num_accepted_tokens_per_pos_total`. For example:

```promql
# Draft acceptance across replicas
sum by (model_name) (rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
/
sum by (model_name) (rate(vllm:spec_decode_num_draft_tokens_total[5m]))

# Mean acceptance length, including the bonus token
1 + sum by (model_name) (rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
/
sum by (model_name) (rate(vllm:spec_decode_num_drafts_total[5m]))
```

The `pypto:` metrics and existing JSON histogram buckets remain available for
the local monitor. Query one namespace at a time to avoid double-counting.
Feature-specific vLLM metrics for LoRA, multimodal inputs, GPU memory, and KV
connectors are not emulated. Per-request vLLM response metrics and `LLM.get_metrics()`
are separate APIs and are not provided by the HTTP metrics compatibility layer.

## Local Speculative Metrics

MTP and DSpark export four additional cumulative counters per replica in both
metrics formats: `speculative_drafts`, `draft_tokens`, `accepted_tokens`, and
`speculative_fallbacks` (Prometheus adds the `pypto:` prefix and `_total` suffix).
Draft tokens count proposals actually submitted for verification, excluding
prefill seeding and proposals generated for a future round. Accepted tokens
count matching drafts before EOS, stop-string, or output-length truncation;
the target/bonus token is excluded. Fallbacks count speculative decode rounds
that execute without drafts. Ordinary non-speculative decoding adds no counts.
The `draft_acceptance_rate` and `mean_acceptance_length` gauges expose cumulative
per-replica ratios directly (`null` in JSON or `NaN` in Prometheus when no drafts
have been verified). Use counter deltas for interval and multi-replica rates.

The dashboard uses summed counter deltas over the last five minutes:

- Draft acceptance = accepted tokens / verified draft tokens.
- Acceptance length = 1 + accepted tokens / speculative verification rounds,
  including the target/bonus token and excluding fallback rounds.
- Fallback rate = fallback rounds / (verification rounds + fallback rounds).

Ratios with no observations are shown as `--`. History buckets and daily totals
use the same weighted calculation across replicas, rather than averaging rates.
Existing SQLite databases gain the new columns automatically; older samples
have zero counters and do not contribute observations. The first scrape after
collector startup or a serving restart establishes a baseline.

The monitor records only aggregate operational metrics. It does not store
prompts, generated text, request IDs, or API credentials.
