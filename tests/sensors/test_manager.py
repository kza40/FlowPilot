"""Integration tests for SensorManager with mock adapters."""

import asyncio
import time

import pytest

from flowpilot.sensors._mock import (
    MockCameraAdapter,
    MockImuAdapter,
    MockLidarAdapter,
    MockRadarAdapter,
)
from flowpilot.sensors.manager import SensorManager
from flowpilot.sensors.types import SyncedFrame


@pytest.mark.asyncio
async def test_manager_emits_synced_frames() -> None:
    camera = MockCameraAdapter(fps=30.0)
    manager = SensorManager(camera=camera)

    frames: list[SyncedFrame] = []
    await manager.start()
    try:
        async for sf in manager.synced_frames():
            frames.append(sf)
            if len(frames) >= 5:
                break
    finally:
        await manager.stop()

    assert len(frames) == 5
    assert all(isinstance(sf, SyncedFrame) for sf in frames)


@pytest.mark.asyncio
async def test_manager_frame_ids_monotonic() -> None:
    camera = MockCameraAdapter(fps=30.0)
    manager = SensorManager(camera=camera)

    ids: list[int] = []
    await manager.start()
    try:
        async for sf in manager.synced_frames():
            ids.append(sf.frame_id)
            if len(ids) >= 10:
                break
    finally:
        await manager.stop()

    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)  # no duplicates


@pytest.mark.asyncio
async def test_manager_all_sensors_present() -> None:
    manager = SensorManager(
        camera=MockCameraAdapter(fps=30.0),
        lidar=MockLidarAdapter(hz=10.0),
        radar=MockRadarAdapter(hz=20.0),
        imu=MockImuAdapter(hz=100.0),
    )

    await manager.start()
    # Give sensors time to produce at least one frame each
    await asyncio.sleep(0.15)

    sf = None
    async for sf in manager.synced_frames():
        break

    await manager.stop()

    assert sf is not None
    # After 150 ms, all sensors should have produced at least one frame
    assert sf.camera is not None, "Camera frame missing"
    assert sf.lidar is not None, "LiDAR frame missing"
    assert sf.radar is not None, "Radar frame missing"
    assert sf.imu is not None, "IMU frame missing"


@pytest.mark.asyncio
async def test_manager_camera_only() -> None:
    """Manager should work with only a camera (most common dev config)."""
    manager = SensorManager(camera=MockCameraAdapter(fps=30.0))
    frames: list[SyncedFrame] = []

    await manager.start()
    try:
        async for sf in manager.synced_frames():
            frames.append(sf)
            if len(frames) >= 3:
                break
    finally:
        await manager.stop()

    assert all(sf.lidar is None for sf in frames)
    assert all(sf.radar is None for sf in frames)
    assert all(sf.imu is None for sf in frames)


@pytest.mark.asyncio
async def test_manager_no_camera_uses_timer() -> None:
    """Without a camera, the manager falls back to a 30 Hz timer."""
    manager = SensorManager(lidar=MockLidarAdapter(hz=10.0))
    frames: list[SyncedFrame] = []

    await manager.start()
    await asyncio.sleep(0.15)

    try:
        async for sf in manager.synced_frames():
            frames.append(sf)
            if len(frames) >= 3:
                break
    finally:
        await manager.stop()

    assert len(frames) >= 3


@pytest.mark.asyncio
async def test_ingestion_benchmark_runs() -> None:
    from flowpilot.benchmarks.ingestion import _run

    report = await _run(duration_s=1.0)
    assert report.n_synced_frames > 0
    assert any(s.sensor == "camera" for s in report.per_sensor)
