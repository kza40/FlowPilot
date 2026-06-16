"""Sensor frame types shared across all ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Frame-period constants in nanoseconds (used by staleness checks)
CAMERA_PERIOD_NS = 33_333_333   # 30 Hz
LIDAR_PERIOD_NS  = 100_000_000  # 10 Hz
RADAR_PERIOD_NS  = 50_000_000   # 20 Hz
IMU_PERIOD_NS    = 10_000_000   # 100 Hz

# Staleness threshold: 2 missed frames before a sensor is considered stale
CAMERA_MAX_AGE_NS = 2 * CAMERA_PERIOD_NS
LIDAR_MAX_AGE_NS  = 2 * LIDAR_PERIOD_NS
RADAR_MAX_AGE_NS  = 2 * RADAR_PERIOD_NS
IMU_MAX_AGE_NS    = 2 * IMU_PERIOD_NS


@dataclass(slots=True)
class CameraFrame:
    sensor_id: str
    timestamp_ns: int
    # (H, W, 3) BGR uint8 on CPU; torch.Tensor (3, H, W) fp16 on CUDA Jetson path
    image: np.ndarray
    width: int
    height: int
    frame_index: int = 0


@dataclass(slots=True)
class LidarFrame:
    sensor_id: str
    timestamp_ns: int
    # (N, 4) float32: x, y, z, intensity  —  ego-centric coordinates, metres
    points: np.ndarray
    frame_index: int = 0


@dataclass(slots=True)
class RadarDetection:
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    velocity_ms: float   # radial velocity; positive = target approaching
    rcs_db: float        # radar cross-section


@dataclass(slots=True)
class RadarFrame:
    sensor_id: str
    timestamp_ns: int
    detections: list[RadarDetection] = field(default_factory=list)
    frame_index: int = 0


@dataclass(slots=True)
class ImuFrame:
    sensor_id: str
    timestamp_ns: int
    accel_ms2: np.ndarray   # (3,) float64: x, y, z  m/s²
    gyro_rads: np.ndarray   # (3,) float64: x, y, z  rad/s
    frame_index: int = 0


@dataclass
class SyncedFrame:
    """All active sensors snapped to a common camera-clock timestamp."""

    frame_id: int
    timestamp_ns: int
    camera: Optional[CameraFrame] = None
    lidar: Optional[LidarFrame] = None
    radar: Optional[RadarFrame] = None
    imu: Optional[ImuFrame] = None
    # Age of each sensor's latest frame at sync time (ns)
    staleness_ns: dict[str, int] = field(default_factory=dict)

    def is_stale(self, sensor_id: str, max_age_ns: int) -> bool:
        return self.staleness_ns.get(sensor_id, max_age_ns + 1) > max_age_ns

    def any_stale(self) -> bool:
        checks = {
            "camera": CAMERA_MAX_AGE_NS,
            "lidar": LIDAR_MAX_AGE_NS,
            "radar": RADAR_MAX_AGE_NS,
            "imu": IMU_MAX_AGE_NS,
        }
        return any(
            self.is_stale(k, v)
            for k, v in checks.items()
            if k in self.staleness_ns
        )
