# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit BF16 matrix providers with actual PyPTO A3 and A5 dispatch paths.

Both accept CPU BF16 [M,K], [K,N] and return owned CPU FP32 [M,N]. The Torch
provider is a CPU reference. The PyPTO provider runs the Cube kernel, including
host/device transfers on every call; it is not a fused device-resident backend.
Decoded low-precision values must already be BF16. FP32-critical HC, routing and
head arithmetic belongs to the caller and must not be silently rounded here.

The per-call budget counts tensor buffers, including padding, estimated device
copies, and conservative finite-check scratch. Compiler/runtime workspaces and
previously returned tensors are outside this limit. Compilation has a separate
bounded number of shape specializations; no persistent binary cache is assumed.
"""

from __future__ import annotations

import tempfile
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch


DEFAULT_MAX_BUFFER_BYTES = 256 << 20


class MatmulOps(Protocol):
    """BF16 inputs, FP32 accumulation/output, with caller-owned input storage."""

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return [M,N] for a[M,K] @ b[K,N], without mutating either input."""
        ...

    def close(self) -> None:
        """Release resources owned by this provider."""
        ...


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    for name, tensor in (("a", a), ("b", b)):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or any(size < 1 for size in tensor.shape):
            raise ValueError(f"{name} must be a nonempty BF16 matrix")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul inner dimensions disagree")
    return a.shape[0], b.shape[1], a.shape[1]


def _finite(tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("matmul inputs and output must be finite")


def _budget(estimate: int, limit: int) -> None:
    if estimate > limit:
        raise ValueError(f"matmul tensor-buffer estimate {estimate} exceeds budget {limit}")


class TorchMatmulOps:
    """CPU golden provider; BF16 values are multiplied and accumulated in FP32."""

    def __init__(self, *, max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES) -> None:
        self.max_buffer_bytes = _positive(max_buffer_bytes, "max_buffer_bytes")

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        rows, columns, inner = _shape(a, b)
        _budget(
            6 * (rows * inner + inner * columns)
            + 4 * rows * columns
            + 16 * max(rows * inner, inner * columns, rows * columns),
            self.max_buffer_bytes,
        )
        _finite(a)
        _finite(b)
        result = a.detach().float() @ b.detach().float()
        _finite(result)
        return result.contiguous()

    def close(self) -> None:
        """The CPU reference retains no tensor or runtime resources."""


class PyptoMatmulOps:
    """Synchronous Ascend JIT dispatch with bounded, instance-owned shape caching.

    Args:
        platform: PyPTO hardware backend, "a2a3" (default, including A3) or "a5".
        device_id: Nonnegative device index, passed to PyPTO RunConfig.
        build_dir: Optional existing parent for the private temporary artifact directory.
        max_buffer_bytes: Positive tensor-buffer budget checked before padding/allocation.
        max_cached_shapes: Maximum cached padded shapes; least-recently-used entries are evicted.

    Close the provider after use. Its private artifact directory is removed on
    close; externally supplied build_dir contents are preserved. PyPTO's native
    synchronous dispatcher releases its temporary worker in its finally block.
    """

    def __init__(
        self,
        *,
        platform: str = "a2a3",
        device_id: int = 0,
        build_dir: str | Path | None = None,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        max_cached_shapes: int = 64,
    ) -> None:
        if type(platform) is not str or platform not in ("a2a3", "a5"):
            raise ValueError("platform must be 'a2a3' or 'a5'")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        self.max_buffer_bytes = _positive(max_buffer_bytes, "max_buffer_bytes")
        self.max_cached_shapes = _positive(max_cached_shapes, "max_cached_shapes")
        try:
            from pypto.runtime import RunConfig

            from .kernels import make_bf16_matmul_kernel
        except ImportError as exc:
            raise RuntimeError(
                "PyptoMatmulOps requires an installed PyPTO compiler and Ascend runtime"
            ) from exc
        self._config = RunConfig(platform=platform, device_id=device_id, save_kernels=True)
        if self._config.platform != platform:
            raise RuntimeError(f"PyPTO did not select the requested {platform!r} backend")
        self._make_kernel = make_bf16_matmul_kernel
        self._artifacts = tempfile.TemporaryDirectory(prefix="v41-matmul-", dir=build_dir)
        self._cache: OrderedDict[tuple[int, int, int], tuple[object, tempfile.TemporaryDirectory]] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self._closed = False

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the actual PyPTO kernel and return an owned, cropped FP32 result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("matmul provider is closed")
            rows, columns, inner = _shape(a, b)
            padded = ((rows + 15) // 16 * 16, (columns + 63) // 64 * 64, (inner + 63) // 64 * 64)
            pm, pn, pk = padded
            buffers = 2 * (pm * pk + pk * pn) + 4 * pm * pn
            _budget(
                2 * (rows * inner + inner * columns)
                + 2 * buffers
                + 4 * rows * columns
                + 16 * max(pm * pk, pk * pn, pm * pn),
                self.max_buffer_bytes,
            )
            _finite(a)
            _finite(b)
            kernel, directory = self._entry(padded)
            lhs = self._pad(a, (pm, pk))
            rhs = self._pad(b, (pk, pn))
            output = torch.empty((pm, pn), dtype=torch.float32)
            config = replace(self._config, save_kernels_dir=directory.name)
            kernel(lhs, rhs, output, config=config)
            _finite(output)
            return output[:rows, :columns].clone(memory_format=torch.contiguous_format)

    def _entry(self, padded: tuple[int, int, int]) -> tuple[object, tempfile.TemporaryDirectory]:
        if padded in self._cache:
            self._cache.move_to_end(padded)
            return self._cache[padded]
        if len(self._cache) == self.max_cached_shapes:
            _, (_, directory) = self._cache.popitem(last=False)
            directory.cleanup()
        kernel = self._make_kernel()
        pm, pn, pk = padded
        directory = tempfile.TemporaryDirectory(prefix=f"m{pm}_n{pn}_k{pk}-", dir=self._artifacts.name)
        # Each JIT object owns exactly one shape, so eviction needs no private
        # compiler-cache mutation. The dispatch lock guarantees it is inactive.
        self._cache[padded] = (kernel, directory)
        return kernel, directory

    @staticmethod
    def _pad(tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if tuple(tensor.shape) == shape and tensor.is_contiguous():
            return tensor.detach()
        padded = torch.zeros(shape, dtype=torch.bfloat16)
        padded[: tensor.shape[0], : tensor.shape[1]].copy_(tensor.detach())
        return padded

    def close(self) -> None:
        """Wait for an active dispatch, then release only this instance's artifacts/cache."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._cache.clear()
                self._artifacts.cleanup()
