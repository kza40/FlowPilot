"""Tests for TemporalSynchronizer."""

import time

import numpy as np
import pytest

from flowpilot.sensors.sync import TemporalSynchronizer, _lerp_imu
from flowpilot.sensors.types import (
    CAMERA_MAX_AGE_NS,
    CameraFrame,
    ImuFrame,
    LidarFrame,
    RadarDetection,
    RadarFrame,
)


def _camera(ts_ns: int) -> CameraFrame:
    return CameraFrame(
        sensor_id="cam",
        timestamp_ns=ts_ns,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        width=2,
        height=2,
    )


def _lidar(ts_ns: int) -> LidarFrame:
    return LidarFrame(
        sensor_id="lid",
        timestamp_ns=ts_ns,
        points=np.zeros((4, 4), dtype=np.float32),
    )


def _imu(ts_ns: int, accel_z: float = 9.81) -> ImuFrame:
    return ImuFrame(
        sensor_id="imu",
        timestamp_ns=ts_ns,
        accel_ms2=np.array([0.0, 0.0, accel_z]),
        gyro_rads=np.zeros(3),
    )


class TestLerpImu:
    def test_at_start(self) -> None:
        a = _imu(0, accel_z=0.0)
        b = _imu(1_000_000, accel_z=10.0)
        r = _lerp_imu(a, b, t_ns=0)
        assert r.accel_ms2[2] == pytest.approx(0.0)

    def test_at_end(self) -> None:
        a = _imu(0, accel_z=0.0)
        b = _imu(1_000_000, accel_z=10.0)
        r = _lerp_imu(a, b, t_ns=1_000_000)
        assert r.accel_ms2[2] == pytest.approx(10.0)

    def test_midpoint(self) -> None:
        a = _imu(0, accel_z=0.0)
        b = _imu(1_000_000, accel_z=10.0)
        r = _lerp_imu(a, b, t_ns=500_000)
        assert r.accel_ms2[2] == pytest.approx(5.0)

    def test_clamp_below(self) -> None:
        a = _imu(100, accel_z=1.0)
        b = _imu(200, accel_z=2.0)
        r = _lerp_imu(a, b, t_ns=0)  # before a
        assert r.accel_ms2[2] == pytest.approx(1.0)


class TestTemporalSynchronizer:
    def test_empty_sync_returns_nones(self) -> None:
        sync = TemporalSynchronizer()
        sf = sync.sync(trigger_ts_ns=time.time_ns())
        assert sf.camera is None
        assert sf.lidar is None
        assert sf.imu is None

    def test_camera_latest_returned(self) -> None:
        sync = TemporalSynchronizer()
        t1 = time.time_ns()
        t2 = t1 + 33_000_000
        sync.update_camera(_camera(t1))
        sync.update_camera(_camera(t2))
        sf = sync.sync(trigger_ts_ns=t2)
        assert sf.camera is not None
        assert sf.camera.timestamp_ns == t2

    def test_staleness_computed(self) -> None:
        sync = TemporalSynchronizer()
        old_ts = time.time_ns() - 500_000_000  # 500 ms ago
        sync.update_lidar(_lidar(old_ts))
        sf = sync.sync()
        assert "lidar" in sf.staleness_ns
        assert sf.staleness_ns["lidar"] > 400_000_000  # at least 400 ms

    def test_imu_interpolated(self) -> None:
        sync = TemporalSynchronizer()
        base = time.time_ns()
        sync.update_imu(_imu(base, accel_z=0.0))
        sync.update_imu(_imu(base + 10_000_000, accel_z=10.0))
        sf = sync.sync(trigger_ts_ns=base + 5_000_000)
        assert sf.imu is not None
        assert sf.imu.accel_ms2[2] == pytest.approx(5.0, abs=0.1)

    def test_frame_ids_increment(self) -> None:
        sync = TemporalSynchronizer()
        ids = [sync.sync().frame_id for _ in range(5)]
        assert ids == list(range(1, 6))

    def test_any_stale_fresh_sensors(self) -> None:
        sync = TemporalSynchronizer()
        now = time.time_ns()
        sync.update_camera(_camera(now - 10_000_000))   # 10 ms ago
        sync.update_lidar(_lidar(now - 50_000_000))     # 50 ms ago
        sf = sync.sync(trigger_ts_ns=now)
        # camera (10 ms) is well within 2-frame budget (66 ms)
        assert not sf.is_stale("camera", max_age_ns=CAMERA_MAX_AGE_NS)
