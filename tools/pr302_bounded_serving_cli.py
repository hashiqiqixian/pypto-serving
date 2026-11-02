#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validation-only CLI launcher for PyPTO be23606 native compilation.

Run this file as a normal Python script with the usual pypto-serving arguments.
The top-level installation also runs when multiprocessing spawn imports this
file as __mp_main__. It only limits device_runner's native compile pool; it
does not construct workers or change device execution, weights, or cache policy.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class _BoundedNativeCompilePool(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        bounded = 4 if max_workers is None else min(max_workers, 4)
        super().__init__(bounded, *args, **kwargs)


def _install_native_compile_bound():
    from pypto.runtime import device_runner

    # Fail closed if the pinned compiler structure or its pool binding changed.
    tree = ast.parse(Path(device_runner.__file__).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_compile_and_assemble_locked"]
    if len(functions) != 1 or device_runner.ThreadPoolExecutor is not ThreadPoolExecutor:
        raise RuntimeError("Expected unpatched PyPTO be23606 native compiler")
    function = functions[0]
    bounds = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "max_workers"
                      for target in node.targets)]
    expected = ast.parse("max_workers = min(64, 1 + len(code_units))").body[0]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ThreadPoolExecutor"]
    expected_call = ast.parse("ThreadPoolExecutor(max_workers=max_workers)", mode="eval").body
    if (len(bounds) != 1 or ast.dump(bounds[0]) != ast.dump(expected)
            or len(calls) != 1 or ast.dump(calls[0]) != ast.dump(expected_call)
            or not any(node is calls[0] for node in ast.walk(function))):
        raise RuntimeError("PyPTO native compile pool differs from the be23606 contract")
    device_runner.ThreadPoolExecutor = _BoundedNativeCompilePool


_install_native_compile_bound()


if __name__ == "__main__":
    from pypto_serving.cli.main import main

    raise SystemExit(main())
