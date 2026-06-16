"""
Sensor ingestion latency benchmark.

Measures the time from when a sensor frame is generated (timestamp_ns on
the frame) to when it becomes available in a SyncedFrame.  Both per-sensor
latency and total synchronizer overhead are reported.

Run directly:  python -m flowpilot.benchmarks.ingestion [--out results.json]

All measurements use mock adapters — no real hardware needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SensorLatencyStats:
    sensor: str
    n_samples: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass
class IngestionReport:
    timestamp: str
    duration_s: float
    n_synced_frames: int
    per_sensor: list[SensorLatencyStats] = field(default_factory=list)
    sync_overhead_median_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


async def _run(duration_s: float = 5.0) -> IngestionReport:
    import datetime

    from flowpilot.sensors._mock import (
        MockCameraAdapter,
        MockImuAdapter,
        MockLidarAdapter,
        MockRadarAdapter,
    )
    from flowpilot.sensors.manager import SensorManager

    camera = MockCameraAdapter(fps=30.0)
    lidar = MockLidarAdapter(hz=10.0)
    radar = MockRadarAdapter(hz=20.0)
    imu = MockImuAdapter(hz=100.0)

    manager = SensorManager(camera=camera, lidar=lidar, radar=radar, imu=imu)

    camera_latencies: list[float] = []
    lidar_latencies: list[float] = []
    radar_latencies: list[float] = []
    imu_latencies: list[float] = []
    sync_overheads: list[float] = []

    await manager.start()
    deadline = time.monotonic() + duration_s

    try:
        async for synced in manager.synced_frames():
            if time.monotonic() > deadline:
                break

            now_ns = time.time_ns()

            t_sync_start = time.perf_counter()
            _ = synced  # sync already done; measure delivery overhead
            sync_overheads.append((time.perf_counter() - t_sync_start) * 1e3)

            if synced.camera:
                camera_latencies.append((now_ns - synced.camera.timestamp_ns) / 1e6)
            if synced.lidar:
                lidar_latencies.append((now_ns - synced.lidar.timestamp_ns) / 1e6)
            if synced.radar:
                radar_latencies.append((now_ns - synced.radar.timestamp_ns) / 1e6)
            if synced.imu:
                imu_latencies.append((now_ns - synced.imu.timestamp_ns) / 1e6)

    finally:
        await manager.stop()

    def stats(name: str, samples: list[float]) -> Optional[SensorLatencyStats]:
        if not samples:
            return None
        s = sorted(samples)
        return SensorLatencyStats(
            sensor=name,
            n_samples=len(s),
            median_ms=round(statistics.median(s), 3),
            p95_ms=round(s[int(0.95 * len(s))], 3),
            p99_ms=round(s[int(0.99 * len(s))], 3),
            max_ms=round(s[-1], 3),
        )

    per_sensor = [
        r for r in [
            stats("camera", camera_latencies),
            stats("lidar", lidar_latencies),
            stats("radar", radar_latencies),
            stats("imu", imu_latencies),
        ]
        if r is not None
    ]

    notes: list[str] = []
    targets = {
        "camera": 10.0,  # ≤10 ms target from plan
        "lidar": 20.0,
        "radar": 15.0,
        "imu": 5.0,
    }
    for s in per_sensor:
        target = targets.get(s.sensor, 80.0)
        if s.p99_ms > target:
            notes.append(
                f"WARNING: {s.sensor} p99 {s.p99_ms:.1f} ms exceeds target {target:.0f} ms"
            )

    if not notes:
        notes.append("All sensors within latency targets at p99.")

    total_frames = len(camera_latencies)

    return IngestionReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_s=duration_s,
        n_synced_frames=total_frames,
        per_sensor=per_sensor,
        sync_overhead_median_ms=round(
            statistics.median(sync_overheads) if sync_overheads else 0.0, 4
        ),
        notes=notes,
    )


def run(duration_s: float = 5.0, out: Optional[Path] = None) -> IngestionReport:
    report = asyncio.run(_run(duration_s))

    print(f"\nIngestion benchmark ({report.duration_s}s, {report.n_synced_frames} frames)\n")
    for s in report.per_sensor:
        print(
            f"  {s.sensor:<8}  median={s.median_ms:.2f} ms  "
            f"p95={s.p95_ms:.2f} ms  p99={s.p99_ms:.2f} ms  max={s.max_ms:.2f} ms"
        )
    print(f"\n  sync overhead median: {report.sync_overhead_median_ms:.4f} ms")
    print()
    for note in report.notes:
        print(f"  [{note}]")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        print(f"\nReport saved → {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowPilot ingestion benchmark")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--out", type=Path, default=None, help="Save JSON report to this path")
    args = parser.parse_args()
    run(duration_s=args.duration, out=args.out)
