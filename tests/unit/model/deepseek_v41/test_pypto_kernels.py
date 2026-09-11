# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real A5 compiler IR/codegen checks; no assembler, simulator, or device execution."""

import importlib
import importlib.util
import re
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="kernel IR tests require Torch tensor metadata")
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("rows,columns,inner", [(16, 64, 64), (32, 128, 128)])
def test_a5_bf16_matmul_lowers_to_fp32_cube_with_out_parameter(tmp_path, monkeypatch, rows, columns, inner):
    pytest.importorskip("pypto.pypto_core", reason="real PyPTO compiler extension is required")
    from pypto import ir
    from pypto.pypto_core import DataType
    from pypto.runtime import RunConfig

    decorator = importlib.import_module("pypto.jit.decorator")
    # Exercise the real specializer, parser, passes and A5 code generator. Only
    # the external assembler is disabled, so this test cannot launch C++ builds.
    monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
    spec = importlib.util.spec_from_file_location(
        "_v41_real_kernel_test", ROOT / "pypto_serving/model/deepseek_v41/kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.make_bf16_matmul_kernel()
    compiled = kernel.compile(
        torch.empty((rows, inner), dtype=torch.bfloat16),
        torch.empty((inner, columns), dtype=torch.bfloat16),
        torch.empty((rows, columns), dtype=torch.float32),
        config=RunConfig(platform="a5", codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path)),
    )
    assert compiled.platform == "a5"
    assert compiled.output_indices == [2]
    program = compiled.program
    orchestration = next(
        function
        for function in program.functions.values()
        if function.func_type == ir.FunctionType.Orchestration
    )
    assert [parameter.type.dtype for parameter in orchestration.params] == [
        DataType.BF16,
        DataType.BF16,
        DataType.FP32,
    ]
    assert [
        [dimension.value for dimension in parameter.type.shape] for parameter in orchestration.params
    ] == [[rows, inner], [inner, columns], [rows, columns]]

    # CompiledProgram.program retains the input tensor IR. The lowered Cube
    # operations must be checked in the actual emitted PTO below.
    class MatmulVisitor(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls = []

        def visit_call(self, call):
            if call.op.name in ("tensor.matmul", "tensor.matmul_acc"):
                self.calls.append(call)
            super().visit_call(call)

    visitor = MatmulVisitor()
    visitor.visit_program(program)
    assert {call.op.name for call in visitor.calls} == {"tensor.matmul", "tensor.matmul_acc"}
    for call in visitor.calls:
        assert call.type.dtype == DataType.FP32
        assert [argument.type.dtype for argument in call.args[-2:]] == [DataType.BF16, DataType.BF16]
    artifacts = list(tmp_path.rglob("*.pto"))
    assert artifacts, "A5 PTO codegen must emit kernel artifacts"
    emitted = "\n".join(path.read_text() for path in artifacts)
    assert "#pto.kernel_kind<cube>" in emitted
    cube_calls = re.findall(r"^\s*pto\.tmatmul(?:\.acc)?\s+.*$", emitted, re.MULTILINE)
    assert cube_calls, "lowering must emit actual Cube matrix operations"
    assert any("pto.tmatmul ins(" in call for call in cube_calls)
    if inner > 64:
        assert any("pto.tmatmul.acc ins(" in call for call in cube_calls)
    for call in cube_calls:
        inputs, outputs = call.split(" outs(", 1)
        assert inputs.count("dtype=bf16") == 2
        assert "loc=left, dtype=bf16" in inputs and "loc=right, dtype=bf16" in inputs
        assert "loc=acc, dtype=f32" in outputs
