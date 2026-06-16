"""
CUDA device query and tensor microbenchmark.

Run directly:  python -m flowpilot.benchmarks.hardware [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DeviceInfo:
    available: bool
    name: str = ""
    compute_capability: str = ""
    sm_count: int = 0
    total_memory_gb: float = 0.0
    # Theoretical peak bandwidth GB/s derived from device props
    memory_bandwidth_gbs: float = 0.0
    cuda_version: str = ""
    torch_version: str = ""


@dataclass
class MatmulResult:
    dtype: str
    size: int
    median_ms: float
    p99_ms: float
    tflops: float


@dataclass
class BenchmarkReport:
    timestamp: str
    device: DeviceInfo
    latency_budget_ms: float = 80.0
    matmul: list[MatmulResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _query_device() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(available=False, notes="torch not installed")  # type: ignore[call-arg]

    if not torch.cuda.is_available():
        return DeviceInfo(available=False, torch_version=torch.__version__)

    props = torch.cuda.get_device_properties(0)
    # Theoretical peak bandwidth: 2 * memory_clock_rate (kHz) * bus_width (bits) / 8 / 1e6 GB/s
    bw_gbs = (
        2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 / 1e9
    )
    return DeviceInfo(
        available=True,
        name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        sm_count=props.multi_processor_count,
        total_memory_gb=round(props.total_memory / 1024**3, 2),
        memory_bandwidth_gbs=round(bw_gbs, 1),
        cuda_version=torch.version.cuda or "",
        torch_version=torch.__version__,
    )


def _bench_matmul(dtype_str: str, size: int, warmup: int = 10, iters: int = 100) -> MatmulResult:
    import torch

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "int8": None}
    dtype = dtype_map[dtype_str]

    if dtype is None:
        # int8 matmul via torch._int_mm (requires sm80+)
        a = torch.randint(-128, 127, (size, size), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (size, size), dtype=torch.int8, device="cuda")
        op = lambda: torch._int_mm(a, b)  # noqa: E731
    else:
        a = torch.randn(size, size, dtype=dtype, device="cuda")
        b = torch.randn(size, size, dtype=dtype, device="cuda")
        op = lambda: torch.matmul(a, b)  # noqa: E731

    # Warmup
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        op()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)

    med = statistics.median(samples)
    p99 = sorted(samples)[int(0.99 * len(samples))]
    # FLOPS for matmul(N,N) @ (N,N) = 2*N^3
    tflops = (2 * size**3) / (med * 1e-3) / 1e12

    return MatmulResult(
        dtype=dtype_str,
        size=size,
        median_ms=round(med, 3),
        p99_ms=round(p99, 3),
        tflops=round(tflops, 3),
    )


def run(out: Optional[Path] = None) -> BenchmarkReport:
    import datetime

    report = BenchmarkReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        device=_query_device(),
    )

    if not report.device.available:
        report.notes.append("CUDA not available — skipping matmul benchmarks")
        _print_report(report)
        if out:
            _save(report, out)
        return report

    print(f"\nDevice : {report.device.name}")
    print(f"SMs    : {report.device.sm_count}")
    print(f"Memory : {report.device.total_memory_gb} GB")
    print(f"BW est.: {report.device.memory_bandwidth_gbs} GB/s")
    print(f"Compute: sm{report.device.compute_capability}\n")

    configs = [
        ("fp16", 2048),
        ("fp16", 4096),
        ("fp32", 1024),
        ("fp32", 2048),
    ]

    # int8 only on sm80+ (Ampere and newer — Orin Nano is sm87)
    if report.device.compute_capability >= "8.0":
        configs += [("int8", 2048), ("int8", 4096)]
    else:
        report.notes.append("int8 _int_mm skipped: requires sm80+")

    for dtype, size in configs:
        r = _bench_matmul(dtype, size)
        report.matmul.append(r)
        print(f"  matmul {dtype} {size}×{size}: median={r.median_ms:.2f} ms  "
              f"p99={r.p99_ms:.2f} ms  {r.tflops:.2f} TFLOPS")

    # Flag if any benchmark exceeds the per-decision budget
    for r in report.matmul:
        if r.median_ms > report.latency_budget_ms:
            report.notes.append(
                f"WARNING: {r.dtype} {r.size}×{r.size} median {r.median_ms} ms "
                f"exceeds {report.latency_budget_ms} ms budget"
            )

    if not report.notes:
        report.notes.append("All benchmarks within latency budget.")

    print()
    for note in report.notes:
        print(f"  [{note}]")

    if out:
        _save(report, out)

    return report


def _print_report(report: BenchmarkReport) -> None:
    print(json.dumps(asdict(report), indent=2))


def _save(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))
    print(f"\nReport saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowPilot hardware benchmark")
    parser.add_argument("--out", type=Path, default=None, help="Save JSON report to this path")
    args = parser.parse_args()
    run(out=args.out)
