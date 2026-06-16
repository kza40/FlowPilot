"""Unit tests for sensor frame types and SyncedFrame helpers."""

import numpy as np
import pytest

from flowpilot.sensors.types import (
    CAMERA_MAX_AGE_NS,
    LIDAR_MAX_AGE_NS,
    CameraFrame,
    ImuFrame,
    LidarFrame,
    RadarDetection,
    RadarFrame,
    SyncedFrame,
)


def make_camera(ts_ns: int = 1_000_000_000) -> CameraFrame:
    return CameraFrame(
        sensor_id="camera_0",
        timestamp_ns=ts_ns,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        width=4,
        height=4,
    )


def make_lidar(ts_ns: int = 1_000_000_000) -> LidarFrame:
    return LidarFrame(
        sensor_id="lidar_0",
        timestamp_ns=ts_ns,
        points=np.zeros((10, 4), dtype=np.float32),
    )


def make_synced(**kwargs) -> SyncedFrame:
    return SyncedFrame(frame_id=1, timestamp_ns=1_000_000_000, **kwargs)


class TestCameraFrame:
    def test_fields(self) -> None:
        f = make_camera(ts_ns=42)
        assert f.sensor_id == "camera_0"
        assert f.timestamp_ns == 42
        assert f.image.shape == (4, 4, 3)

    def test_default_frame_index(self) -> None:
        assert make_camera().frame_index == 0


class TestLidarFrame:
    def test_points_shape(self) -> None:
        f = make_lidar()
        assert f.points.shape == (10, 4)
        assert f.points.dtype == np.float32


class TestRadarFrame:
    def test_empty_detections(self) -> None:
        f = RadarFrame(sensor_id="radar_0", timestamp_ns=0)
        assert f.detections == []

    def test_detection_fields(self) -> None:
        d = RadarDetection(
            range_m=50.0,
            azimuth_rad=0.1,
            elevation_rad=0.0,
            velocity_ms=-5.0,
            rcs_db=10.0,
        )
        assert d.range_m == 50.0
        assert d.velocity_ms == -5.0


class TestSyncedFrame:
    def test_is_stale_missing_sensor(self) -> None:
        sf = make_synced()
        # Missing sensor has no entry → treated as stale
        assert sf.is_stale("lidar", max_age_ns=LIDAR_MAX_AGE_NS)

    def test_is_stale_fresh(self) -> None:
        sf = make_synced(staleness_ns={"camera": 10_000_000})  # 10 ms
        assert not sf.is_stale("camera", max_age_ns=CAMERA_MAX_AGE_NS)

    def test_is_stale_old(self) -> None:
        sf = make_synced(staleness_ns={"camera": CAMERA_MAX_AGE_NS + 1})
        assert sf.is_stale("camera", max_age_ns=CAMERA_MAX_AGE_NS)

    def test_any_stale_all_fresh(self) -> None:
        sf = make_synced(staleness_ns={
            "camera": 10_000_000,
            "lidar": 50_000_000,
            "radar": 20_000_000,
            "imu": 3_000_000,
        })
        assert not sf.any_stale()

    def test_any_stale_one_old(self) -> None:
        sf = make_synced(staleness_ns={
            "camera": 10_000_000,
            "lidar": LIDAR_MAX_AGE_NS + 1,
        })
        assert sf.any_stale()
