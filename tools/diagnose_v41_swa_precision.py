# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU replay of a saved two-layer device trace to bisect accumulated error.

Uses the original seed-11 full-prefix fixture and checkpoint layers 0/1.
No device execution, production operator composition or tolerance changes.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def metrics(actual, expected):
    import torch

    a, e = actual.double(), expected.double()
    diff = (a - e).abs()
    floor = (1.0 / (1 << 14)) / 0.003
    denom = torch.maximum(a.abs(), e.abs()).clamp_min(floor) + 1e-9
    relative = torch.where(diff < 0.003, diff, diff / denom)
    return {"rel_l2": float(diff.norm() / e.norm().clamp_min(1e-12)),
            "max_abs": float(diff.max()), "bad_fraction": float((relative > .003).double().mean()),
            "equal_fraction": float((a == e).double().mean())}


def trace_attention(swa, tensors, actual_hidden, reference_hidden):
    """Bisect the second attention on CPU, recording its public reference calls."""
    from models.deepseek_v4_1_flash import decode_attn_swa as ref

    original_linear, original_rope = ref.official_linear, ref.official_rope
    traces = []
    try:
        for hidden in (actual_hidden, reference_hidden):
            trace = {}
            calls = iter(("q_a", "q_b", "kv", "o_b"))
            ropes = iter(("q_rope", "kv_rope", "out_rope"))

            def linear(x, weight, scale, fp32=False):
                name = next(calls)
                trace[name + ".input"] = x.clone()
                payload, codes = ref.official_quantize(x)
                trace[name + ".quant"] = payload.float()
                trace[name + ".scale"] = codes.float()
                result = original_linear(x, weight, scale, fp32=fp32)
                trace[name + ".output"] = result.clone()
                return result

            def rope(x, cos, sin, inverse=False):
                name = next(ropes)
                trace[name + ".input"] = x.clone()
                result = original_rope(x, cos, sin, inverse=inverse)
                trace[name + ".output"] = result.clone()
                return result

            ref.official_linear, ref.official_rope = linear, rope
            inputs = {name: tensors[name][0] for name in swa.HC_INPUT_NAMES if name not in (
                "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
            )}
            inputs["x"] = hidden
            ref.official_reference(inputs)
            traces.append(trace)
    finally:
        ref.official_linear, ref.official_rope = original_linear, original_rope
    for name in traces[0]:
        print("TRACE", name, json.dumps(metrics(traces[0][name], traces[1][name])), flush=True)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--saved", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-attention", action="store_true",
                        help="Use the saved full-reference replay to bisect layer-1 rank-0 attention")
    parser.add_argument("--cut-after", type=int, choices=(0, 1, 2),
                        help="Restart the CPU reference from a saved device boundary; diagnostic only")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.lib_root).resolve()))
    import torch
    from golden.spec import TensorSpec
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology, load_segment_modules
    from pypto_serving.model.deepseek_v41.swa_weights import load_swa_layer_weights

    torch.set_num_threads(4)
    topology = SegmentTopology(tp=2, dp=2)
    swa, moe = load_segment_modules(args.lib_root, topology)
    saved = torch.load(args.saved, map_location="cpu", weights_only=True)
    fixture = SimpleNamespace(tokens=32, requests=1, dp=2, seed=11, case="normal",
                              fixture="checkpoint", dp_tokens=None, epochs=1, bench=False)
    a = {s.name: s.create_tensor().contiguous() for s in swa.build_hc_specs(fixture)
         if isinstance(s, TensorSpec)}
    if args.trace_attention:
        full = torch.load(args.output, map_location="cpu", weights_only=True)
        print("Loading layer 1 for attention trace", flush=True)
        aw, unused_moe = load_swa_layer_weights(args.model_dir, 1, topology)
        del unused_moe
        traces = trace_attention(swa, dict(a, **aw), saved["stages"][2]["actual"]["hidden"][0],
                                 full[2]["expected"]["hidden"][0])
        torch.save(traces, str(args.output) + ".attention-trace.pt")
        return
    residual, mix = a["x_hc"], a["incoming_pre_mix"]
    records = []
    for layer in (0, 1):
        if args.cut_after is not None and args.cut_after >= 2 * layer + 1:
            cut = saved["stages"][2 * layer + 1]["actual"]
            residual, mix = cut["x_next"], cut["next_pre_mix"]
            continue
        print(f"Loading layer {layer}", flush=True)
        aw, mw = load_swa_layer_weights(args.model_dir, layer, topology)
        la = dict(a, **aw, x_hc=residual, incoming_pre_mix=mix)
        for name in ("output", "next_pre_mix", "hidden", "attn_out", "window_cache", "window_cache_scale"):
            la[name] = a[name].clone()
        swa.golden_prefill_swa_case(la)
        if args.cut_after == 2 * layer:
            cut = saved["stages"][2 * layer]["actual"]
            la["output"], la["next_pre_mix"] = cut["output"], cut["next_pre_mix"]
        lm = dict(mw, x_hc=la["output"], pre_mix=la["next_pre_mix"],
                  next_pre_mix=torch.zeros_like(mix), x_mixed=torch.zeros_like(a["attn_out"]),
                  x_next=torch.zeros_like(residual), num_tokens=torch.full((4,), 16, dtype=torch.int32))
        moe.golden_moe(lm)
        for kind, expected in (("attention", la), ("moe", lm)):
            actual = saved["stages"][2 * layer + (kind == "moe")]["actual"]
            result = {name: metrics(actual[name], expected[name]) for name in actual
                      if name not in ("window_cache", "window_cache_scale")}
            print(json.dumps({"layer": layer, "stage": kind, "metrics": result}), flush=True)
            records.append({"layer": layer, "stage": kind, "metrics": result,
                            "expected": {name: expected[name].clone() for name in actual}})
        residual, mix = lm["x_next"], lm["next_pre_mix"]
        del aw, mw, la, lm
    if args.cut_after is not None:
        print("CUT", args.cut_after, "final residual", json.dumps(metrics(saved["actual_residual"], residual)),
              "final pre_mix", json.dumps(metrics(saved["actual_pre_mix"], mix)), flush=True)
        from validate_v41_swa_segment import compare_saved
        result = {"actual_residual": saved["actual_residual"], "expected_residual": residual,
                  "actual_pre_mix": saved["actual_pre_mix"], "expected_pre_mix": mix}
        torch.save(result, str(args.output) + f".cut-{args.cut_after}.pt")
        compare_saved(result, moe, topology)
        return
    print("Replay matches saved reference:", torch.equal(residual, saved["expected_residual"]),
          torch.equal(mix, saved["expected_pre_mix"]), flush=True)
    assert torch.equal(residual, saved["expected_residual"])
    assert torch.equal(mix, saved["expected_pre_mix"])
    torch.save(records, args.output)


if __name__ == "__main__":
    main()
