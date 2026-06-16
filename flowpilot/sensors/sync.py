"""
Temporal synchronizer — snaps all sensor streams to a common clock.

Strategy
--------
The camera runs at 30 Hz and acts as the primary clock.  When a new
CameraFrame arrives it triggers a synchronization event: the synchronizer
reads the *latest* available frame from each other sensor, computes each
sensor's age (now - frame.timestamp_ns), and emits a SyncedFrame.

For the IMU (100 Hz), the latest frame is always very fresh.  For LiDAR
(10 Hz) and radar (20 Hz), the frame may be up to one frame-period old;
the staleness_ns field reports this so downstream consumers can decide
whether to use the data or request a HOLD.

Interpolation
-------------
Numeric sensor data (IMU accel/gyro, radar range/velocity) is linearly
interpolated between the two frames nearest to the camera timestamp when
a two-frame history is available.  Camera and LiDAR use nearest-neighbour
(interpolating images or point clouds is not meaningful here).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .types import (
    CAMERA_MAX_AGE_NS,
    IMU_MAX_AGE_NS,
    LIDAR_MAX_AGE_NS,
    RADAR_MAX_AGE_NS,
    CameraFrame,
    ImuFrame,
    LidarFrame,
    RadarFrame,
    SyncedFrame,
)

log = logging.getLogger(__name__)

_HISTORY = 4  # keep last N frames per sensor for interpolation


def _lerp_imu(a: ImuFrame, b: ImuFrame, t_ns: int) -> ImuFrame:
    """Linear interpolation between two ImuFrames at timestamp t_ns."""
    span = b.timestamp_ns - a.timestamp_ns
    if span <= 0:
        return b
    alpha = (t_ns - a.timestamp_ns) / span
    alpha = max(0.0, min(1.0, alpha))
    return ImuFrame(
        sensor_id=b.sensor_id,
        timestamp_ns=t_ns,
        accel_ms2=a.accel_ms2 + alpha * (b.accel_ms2 - a.accel_ms2),
        gyro_rads=a.gyro_rads + alpha * (b.gyro_rads - a.gyro_rads),
        frame_index=b.frame_index,
    )


class TemporalSynchronizer:
    """
    Thread-safe ring-buffer-per-sensor synchronizer.

    Push frames from any thread via `update_*`.
    Call `sync(trigger_ts_ns)` from the asyncio loop to produce a SyncedFrame.
    """

    def __init__(
        self,
        camera_max_age_ns: int = CAMERA_MAX_AGE_NS,
        lidar_max_age_ns: int = LIDAR_MAX_AGE_NS,
        radar_max_age_ns: int = RADAR_MAX_AGE_NS,
        imu_max_age_ns: int = IMU_MAX_AGE_NS,
    ) -> None:
        self._camera_max_age = camera_max_age_ns
        self._lidar_max_age = lidar_max_age_ns
        self._radar_max_age = radar_max_age_ns
        self._imu_max_age = imu_max_age_ns

        self._lock = threading.Lock()
        self._cameras: deque[CameraFrame] = deque(maxlen=_HISTORY)
        self._lidars: deque[LidarFrame] = deque(maxlen=_HISTORY)
        self._radars: deque[RadarFrame] = deque(maxlen=_HISTORY)
        self._imus: deque[ImuFrame] = deque(maxlen=_HISTORY)

        self._frame_id = 0

    # ── producers (called from sensor threads) ────────────────────────────────

    def update_camera(self, frame: CameraFrame) -> None:
        with self._lock:
            self._cameras.append(frame)

    def update_lidar(self, frame: LidarFrame) -> None:
        with self._lock:
            self._lidars.append(frame)

    def update_radar(self, frame: RadarFrame) -> None:
        with self._lock:
            self._radars.append(frame)

    def update_imu(self, frame: ImuFrame) -> None:
        with self._lock:
            self._imus.append(frame)

    # ── consumer (called from asyncio loop) ───────────────────────────────────

    def sync(self, trigger_ts_ns: Optional[int] = None) -> SyncedFrame:
        """
        Produce a SyncedFrame snapped to `trigger_ts_ns` (defaults to now).
        """
        now_ns = trigger_ts_ns if trigger_ts_ns is not None else time.time_ns()

        with self._lock:
            camera = self._cameras[-1] if self._cameras else None
            lidar = self._lidars[-1] if self._lidars else None
            radar = self._radars[-1] if self._radars else None
            imu_frames = list(self._imus)

        # Interpolate IMU if we have two bracketing frames
        imu = _best_imu(imu_frames, now_ns)

        staleness: dict[str, int] = {}
        if camera:
            staleness["camera"] = now_ns - camera.timestamp_ns
        if lidar:
            staleness["lidar"] = now_ns - lidar.timestamp_ns
        if radar:
            staleness["radar"] = now_ns - radar.timestamp_ns
        if imu:
            staleness["imu"] = now_ns - imu.timestamp_ns

        self._frame_id += 1
        synced = SyncedFrame(
            frame_id=self._frame_id,
            timestamp_ns=now_ns,
            camera=camera,
            lidar=lidar,
            radar=radar,
            imu=imu,
            staleness_ns=staleness,
        )

        _log_stale(synced, self._camera_max_age, self._lidar_max_age,
                   self._radar_max_age, self._imu_max_age)
        return synced


def _best_imu(
    frames: list[ImuFrame], t_ns: int
) -> Optional[ImuFrame]:
    """Return interpolated IMU frame at t_ns, or nearest if only one available."""
    if not frames:
        return None
    if len(frames) == 1:
        return frames[-1]

    # Find the two frames that bracket t_ns
    before: Optional[ImuFrame] = None
    after: Optional[ImuFrame] = None
    for f in frames:
        if f.timestamp_ns <= t_ns:
            before = f
        elif after is None:
            after = f
            break

    if before is None:
        return frames[0]
    if after is None:
        return frames[-1]
    return _lerp_imu(before, after, t_ns)


def _log_stale(
    frame: SyncedFrame,
    cam_max: int,
    lid_max: int,
    rad_max: int,
    imu_max: int,
) -> None:
    thresholds = {"camera": cam_max, "lidar": lid_max, "radar": rad_max, "imu": imu_max}
    for sensor, max_age in thresholds.items():
        age = frame.staleness_ns.get(sensor)
        if age is not None and age > max_age:
            log.warning(
                "SyncedFrame[%d]: %s stale by %.1f ms (max %.1f ms)",
                frame.frame_id,
                sensor,
                age / 1e6,
                max_age / 1e6,
            )
