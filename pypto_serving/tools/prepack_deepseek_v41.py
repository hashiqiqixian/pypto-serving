# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the optional V4.1 packed decode-weight sidecar once."""

import argparse
import time
from pathlib import Path

from pypto_serving.model.deepseek_v41.decode_weight_resources import DecodeWeightBank, DecodeWeightLayout
from pypto_serving.model.deepseek_v41.decode_weights import iter_decode_weight_slices
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
from pypto_serving.model.deepseek_v41.weight_sidecar import sidecar_path, write_prepacked_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = sidecar_path(args.model_dir) if args.output is None else args.output
    if output.exists() and not args.force:
        parser.error(f"sidecar already exists: {output}")
    started = time.perf_counter()
    layout = DecodeWeightLayout(40, 4, 8, 3, 8)
    bank = DecodeWeightBank(layout)
    for item in iter_decode_weight_slices(args.model_dir, SegmentTopology.for_decode()):
        bank.append(item)
        if item.name == "hc_attn_fn":
            print(f"packed layer {item.slot + 1}/{layout.layers}", flush=True)
    path = write_prepacked_bank(args.model_dir, bank, path=output, force=args.force)
    print(f"Wrote {path} ({bank.allocated_bytes / (1 << 30):.1f} GiB) "
          f"in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
