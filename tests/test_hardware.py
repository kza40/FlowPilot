"""
Phase 1 hardware smoke tests.

Non-CUDA tests run everywhere (CI, dev laptops).
Tests marked @pytest.mark.cuda require a CUDA-capable GPU.
Run with:
    pytest -m "not cuda"    # fast, no GPU needed
    pytest -m cuda          # on Jetson or CUDA dev machine
"""

from __future__ import annotations

import importlib

import pytest


# ── Package import smoke tests (no GPU needed) ────────────────────────────────

def test_flowpilot_importable() -> None:
    import flowpilot
    assert flowpilot.__version__ == "0.1.0"


@pytest.mark.parametrize("module", [
    "flowpilot.sensors",
    "flowpilot.perception",
    "flowpilot.agent",
    "flowpilot.safety",
    "flowpilot.benchmarks",
    "flowpilot.benchmarks.hardware",
    "flowpilot.benchmarks.compare",
])
def test_subpackage_importable(module: str) -> None:
    importlib.import_module(module)


def test_numpy_importable() -> None:
    import numpy as np
    assert np.__version__


def test_pydantic_importable() -> None:
    import pydantic
    assert int(pydantic.VERSION.split(".")[0]) >= 2


# ── CUDA presence and device properties ───────────────────────────────────────

@pytest.mark.cuda
def test_torch_cuda_available() -> None:
    import torch
    assert torch.cuda.is_available(), (
        "CUDA not available. Ensure JetPack + PyTorch Jetson wheel are installed."
    )


@pytest.mark.cuda
def test_device_is_ampere_or_newer() -> None:
    """Orin Nano is sm87. Guard against accidentally running on an old GPU."""
    import torch
    props = torch.cuda.get_device_properties(0)
    major = props.major
    assert major >= 8, (
        f"Expected Ampere (sm8x) or newer, got sm{major}.{props.minor} on {props.name}. "
        "Int8 kernels and DLA offload require sm80+."
    )


@pytest.mark.cuda
def test_device_has_sufficient_memory() -> None:
    """Orin Nano has 8 GB shared; require at least 4 GB visible to CUDA."""
    import torch
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 1024**3
    assert gb >= 4.0, f"Only {gb:.1f} GB CUDA memory detected — expected ≥ 4 GB on Orin Nano."


@pytest.mark.cuda
def test_sm_count() -> None:
    """Orin Nano has 16 SMs (1024 CUDA cores). Warn if fewer."""
    import torch
    props = torch.cuda.get_device_properties(0)
    # Not a hard failure — allow smaller GPUs for dev, but warn.
    if props.multi_processor_count < 16:
        pytest.warns(
            UserWarning,
            match="fewer SMs than Orin Nano",
        )


# ── Tensor precision smoke tests ─────────────────────────────────────────────

@pytest.mark.cuda
def test_fp16_matmul_correct() -> None:
    import torch
    a = torch.eye(64, dtype=torch.float16, device="cuda")
    b = torch.ones(64, 64, dtype=torch.float16, device="cuda")
    result = torch.matmul(a, b)
    torch.cuda.synchronize()
    expected = torch.ones(64, 64, dtype=torch.float16, device="cuda")
    assert torch.allclose(result, expected, atol=1e-2), "fp16 matmul identity check failed"


@pytest.mark.cuda
def test_int8_matmul_available() -> None:
    """torch._int_mm is required for int8 inference on Orin Nano (sm87)."""
    import torch
    assert hasattr(torch, "_int_mm"), "torch._int_mm not available — update PyTorch"
    a = torch.zeros(32, 32, dtype=torch.int8, device="cuda")
    b = torch.zeros(32, 32, dtype=torch.int8, device="cuda")
    out = torch._int_mm(a, b)
    torch.cuda.synchronize()
    assert out.shape == (32, 32)


# ── Latency floor: single matmul within budget ────────────────────────────────

@pytest.mark.cuda
def test_fp16_matmul_latency_within_budget() -> None:
    """
    A 2048×2048 fp16 matmul should complete well under the 80 ms per-cycle
    budget. If this fails, the GPU is severely throttled or misconfigured.
    """
    import time
    import torch

    BUDGET_MS = 80.0
    SIZE = 2048
    WARMUP = 5
    ITERS = 20

    a = torch.randn(SIZE, SIZE, dtype=torch.float16, device="cuda")
    b = torch.randn(SIZE, SIZE, dtype=torch.float16, device="cuda")

    for _ in range(WARMUP):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    median_ms = sorted(times)[len(times) // 2]
    assert median_ms < BUDGET_MS, (
        f"fp16 2048×2048 matmul median {median_ms:.1f} ms exceeds {BUDGET_MS} ms budget. "
        "Check GPU clock, thermal throttling, or power mode (sudo nvpmodel -m 0)."
    )


# ── Benchmark harness integration ─────────────────────────────────────────────

@pytest.mark.cuda
def test_hardware_benchmark_runs() -> None:
    from flowpilot.benchmarks.hardware import run
    report = run(out=None)
    assert report.device.available
    assert len(report.matmul) >= 2
