"""
SensorManager — orchestrates all sensor adapters and the synchronizer.

Usage
-----
    manager = SensorManager(
        camera=CameraAdapter(...),
        lidar=OusterLidarAdapter(...),
        radar=SocketCanRadarAdapter(...),
        imu=SerialImuAdapter(...),
    )

    async with manager:
        async for synced_frame in manager.synced_frames():
            process(synced_frame)

All sensors are optional; SensorManager works with any subset.
The camera is the primary clock: one SyncedFrame is emitted per
CameraFrame.  If no camera is provided, frames are emitted on a 30 Hz
timer instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from .sync import TemporalSynchronizer
from .types import (
    CameraFrame,
    ImuFrame,
    LidarFrame,
    RadarFrame,
    SyncedFrame,
)

log = logging.getLogger(__name__)

_FALLBACK_HZ = 30.0  # used when no camera adapter is present


class SensorManager:
    def __init__(
        self,
        camera=None,
        lidar=None,
        radar=None,
        imu=None,
        synchronizer: Optional[TemporalSynchronizer] = None,
    ) -> None:
        self._camera = camera
        self._lidar = lidar
        self._radar = radar
        self._imu = imu
        self._sync = synchronizer or TemporalSynchronizer()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True

        adapters = {
            "camera": self._camera,
            "lidar": self._lidar,
            "radar": self._radar,
            "imu": self._imu,
        }
        for name, adapter in adapters.items():
            if adapter is not None:
                await adapter.start()
                log.info("SensorManager: started %s adapter", name)

        if self._camera:
            self._tasks.append(asyncio.create_task(self._feed_camera(), name="feed-camera"))
        if self._lidar:
            self._tasks.append(asyncio.create_task(self._feed_lidar(), name="feed-lidar"))
        if self._radar:
            self._tasks.append(asyncio.create_task(self._feed_radar(), name="feed-radar"))
        if self._imu:
            self._tasks.append(asyncio.create_task(self._feed_imu(), name="feed-imu"))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for adapter in [self._camera, self._lidar, self._radar, self._imu]:
            if adapter is not None:
                await adapter.stop()

        log.info("SensorManager: all adapters stopped")

    @asynccontextmanager
    async def run(self):
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    # ── feeder coroutines (push into synchronizer) ───────────────────────────

    async def _feed_camera(self) -> None:
        async for frame in self._camera.frames():
            self._sync.update_camera(frame)

    async def _feed_lidar(self) -> None:
        async for frame in self._lidar.frames():
            self._sync.update_lidar(frame)

    async def _feed_radar(self) -> None:
        async for frame in self._radar.frames():
            self._sync.update_radar(frame)

    async def _feed_imu(self) -> None:
        async for frame in self._imu.frames():
            self._sync.update_imu(frame)

    # ── primary output ────────────────────────────────────────────────────────

    async def synced_frames(self) -> AsyncIterator[SyncedFrame]:
        """
        Yields one SyncedFrame per camera frame (30 Hz), or at 30 Hz on a
        timer if no camera adapter is configured.
        """
        if self._camera:
            async for _ in self._camera.frames():
                yield self._sync.sync()
        else:
            period = 1.0 / _FALLBACK_HZ
            while self._running:
                t0 = time.monotonic()
                yield self._sync.sync()
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, period - elapsed))

    # ── convenience accessors ─────────────────────────────────────────────────

    def latest_camera(self) -> Optional[CameraFrame]:
        return self._sync._cameras[-1] if self._sync._cameras else None

    def latest_lidar(self) -> Optional[LidarFrame]:
        return self._sync._lidars[-1] if self._sync._lidars else None

    def latest_radar(self) -> Optional[RadarFrame]:
        return self._sync._radars[-1] if self._sync._radars else None

    def latest_imu(self) -> Optional[ImuFrame]:
        return self._sync._imus[-1] if self._sync._imus else None
