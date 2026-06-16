"""
Synthetic sensor adapters for testing and benchmarking.

These generate realistic-sized data at configurable rates without any
real hardware. All adapters implement the SensorAdapter protocol.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from .types import (
    CameraFrame,
    ImuFrame,
    LidarFrame,
    RadarDetection,
    RadarFrame,
)


class MockCameraAdapter:
    """Emits synthetic 1920×1080 BGR frames at 30 Hz."""

    def __init__(
        self,
        sensor_id: str = "camera_0",
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._period = 1.0 / fps
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[CameraFrame]:
        while self._running:
            t0 = time.monotonic()
            image = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            yield CameraFrame(
                sensor_id=self.sensor_id,
                timestamp_ns=time.time_ns(),
                image=image,
                width=self.width,
                height=self.height,
                frame_index=self._frame_index,
            )
            self._frame_index += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._period - elapsed))


class MockLidarAdapter:
    """Emits synthetic point clouds (~30 000 points) at 10 Hz."""

    def __init__(
        self,
        sensor_id: str = "lidar_0",
        n_points: int = 30_000,
        hz: float = 10.0,
    ) -> None:
        self.sensor_id = sensor_id
        self._n_points = n_points
        self._period = 1.0 / hz
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[LidarFrame]:
        while self._running:
            t0 = time.monotonic()
            xyz = np.random.uniform(-50.0, 50.0, (self._n_points, 3)).astype(np.float32)
            intensity = np.random.uniform(0.0, 1.0, (self._n_points, 1)).astype(np.float32)
            points = np.hstack([xyz, intensity])
            yield LidarFrame(
                sensor_id=self.sensor_id,
                timestamp_ns=time.time_ns(),
                points=points,
                frame_index=self._frame_index,
            )
            self._frame_index += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._period - elapsed))


class MockRadarAdapter:
    """Emits synthetic radar detections at 20 Hz."""

    def __init__(
        self,
        sensor_id: str = "radar_0",
        n_detections: int = 8,
        hz: float = 20.0,
    ) -> None:
        self.sensor_id = sensor_id
        self._n_detections = n_detections
        self._period = 1.0 / hz
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[RadarFrame]:
        while self._running:
            t0 = time.monotonic()
            detections = [
                RadarDetection(
                    range_m=float(np.random.uniform(5.0, 150.0)),
                    azimuth_rad=float(np.random.uniform(-0.5, 0.5)),
                    elevation_rad=float(np.random.uniform(-0.1, 0.1)),
                    velocity_ms=float(np.random.uniform(-30.0, 30.0)),
                    rcs_db=float(np.random.uniform(-10.0, 20.0)),
                )
                for _ in range(self._n_detections)
            ]
            yield RadarFrame(
                sensor_id=self.sensor_id,
                timestamp_ns=time.time_ns(),
                detections=detections,
                frame_index=self._frame_index,
            )
            self._frame_index += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._period - elapsed))


class MockImuAdapter:
    """Emits synthetic IMU readings at 100 Hz."""

    def __init__(
        self,
        sensor_id: str = "imu_0",
        hz: float = 100.0,
    ) -> None:
        self.sensor_id = sensor_id
        self._period = 1.0 / hz
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[ImuFrame]:
        while self._running:
            t0 = time.monotonic()
            yield ImuFrame(
                sensor_id=self.sensor_id,
                timestamp_ns=time.time_ns(),
                accel_ms2=np.array([
                    np.random.normal(0.0, 0.5),
                    np.random.normal(0.0, 0.5),
                    np.random.normal(9.81, 0.1),
                ]),
                gyro_rads=np.random.normal(0.0, 0.01, 3),
                frame_index=self._frame_index,
            )
            self._frame_index += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._period - elapsed))
