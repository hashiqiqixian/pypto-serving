# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prometheus compatibility with vLLM V1, including its bundled Grafana dashboard."""

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily

_LABELS = ["model_name", "engine"]
_FINISH_REASONS = {
    "finished_eos": "stop", "finished_stop": "stop", "finished_length": "length",
    "finished_aborted": "abort", "error": "error",
}
_COUNTERS = {
    "prompt_tokens": "processed_prompt_tokens",
    "generation_tokens": "generation_tokens",
    "prefix_cache_queries": "prefix_cache_query_tokens",
    "prefix_cache_hits": "prefix_cache_hits",
    "num_preemptions": "preemptions",
    "spec_decode_num_drafts": "speculative_drafts",
    "spec_decode_num_draft_tokens": "draft_tokens",
    "spec_decode_num_accepted_tokens": "accepted_tokens",
}
_HISTOGRAMS = {
    "time_to_first_token_seconds": "ttft",
    "inter_token_latency_seconds": "itl",
    "request_time_per_output_token_seconds": "tpot",
    "e2e_request_latency_seconds": "e2e",
    "request_queue_time_seconds": "queue",
    "request_prefill_time_seconds": "prefill",
    "request_decode_time_seconds": "decode",
    "request_inference_time_seconds": "inference",
    "request_prompt_tokens": "prompt_tokens",
    "request_generation_tokens": "generation_tokens",
    "request_max_num_generation_tokens": "max_generation_tokens",
    "request_params_max_tokens": "max_tokens",
    "request_params_n": "n",
    "iteration_tokens_total": "iteration_tokens",
}
_TPOT_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)
_REQUEST_BUCKETS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
    40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)


def histogram_buckets(max_model_len: int) -> dict[str, tuple]:
    tokens = []
    scale = 1
    while scale <= max_model_len:
        tokens.extend(value * scale for value in (1, 2, 5) if value * scale <= max_model_len)
        scale *= 10
    return {
        "itl": _TPOT_BUCKETS, "tpot": _TPOT_BUCKETS,
        **{key: _REQUEST_BUCKETS for key in ("e2e", "queue", "prefill", "decode", "inference")},
        **{key: tuple(tokens) for key in ("prompt_tokens", "generation_tokens", "max_generation_tokens", "max_tokens")},
        "n": (1, 2, 5, 10, 20),
        "iteration_tokens": (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    }


class _SnapshotCollector:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    def collect(self):
        replicas = self.snapshot["replicas"]
        model = self.snapshot["model_name"]
        for name, key in (
            ("num_requests_running", "running"), ("num_requests_waiting", "waiting"),
            ("kv_cache_usage_perc", "kv_cache_usage"),
        ):
            metric = GaugeMetricFamily(f"vllm:{name}", name.replace("_", " "), labels=_LABELS)
            for replica in replicas:
                metric.add_metric([model, str(replica["engine"])], replica["gauges"][key])
            yield metric
        for name, key in _COUNTERS.items():
            metric = CounterMetricFamily(f"vllm:{name}", name.replace("_", " "), labels=_LABELS)
            for replica in replicas:
                metric.add_metric([model, str(replica["engine"])], replica["counters"][key])
            yield metric

        finished = CounterMetricFamily(
            "vllm:request_success", "Count of terminated requests.", labels=_LABELS + ["finished_reason"],
        )
        positions = CounterMetricFamily(
            "vllm:spec_decode_num_accepted_tokens_per_pos", "Accepted tokens per draft position.",
            labels=_LABELS + ["position"],
        )
        for replica in replicas:
            labels = [model, str(replica["engine"])]
            reasons = dict.fromkeys(("stop", "length", "abort", "error", "repetition"), 0)
            for reason, count in replica["finish_reasons"].items():
                mapped = _FINISH_REASONS.get(reason, reason)
                reasons[mapped if mapped in reasons else "error"] += count
            for reason, count in reasons.items():
                finished.add_metric(labels + [reason], count)
            for pos, count in enumerate(replica["accepted_tokens_per_pos"]):
                positions.add_metric(labels + [str(pos)], count)
        yield finished
        yield positions

        for name, key in _HISTOGRAMS.items():
            metric = HistogramMetricFamily(f"vllm:{name}", name.replace("_", " "), labels=_LABELS)
            for replica in replicas:
                histogram = replica["histograms"]["ttft"] if key == "ttft" else replica["vllm_histograms"][key]
                buckets = [(str(float(bucket["le"])), bucket["count"]) for bucket in histogram["buckets"]]
                buckets.append(("+Inf", histogram["count"]))
                metric.add_metric([model, str(replica["engine"])], buckets, histogram["sum"])
            yield metric


def render_metric_families(families) -> str:
    class Collector:
        def collect(self):
            yield from families

    # A private registry avoids global collectors and multiprocess environment state.
    registry = CollectorRegistry()
    registry.register(Collector())
    return generate_latest(registry).decode("utf-8")


def render_vllm(snapshot: dict) -> str:
    return render_metric_families(_SnapshotCollector(snapshot).collect())
