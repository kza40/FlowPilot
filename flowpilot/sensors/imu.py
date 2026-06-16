"""
IMU ingestion adapter — serial port (UART) interface.

Parses a simple binary frame format suitable for common IMU modules
(e.g., VectorNav VN-100, u-blox IMU, or any device outputting the
minimal frame defined below).

Frame format (16 bytes, little-endian):
    [uint32  timestamp_us]
    [float32 accel_x]  [float32 accel_y]  [float32 accel_z]   (m/s²)
    [float16 gyro_x]   [float16 gyro_y]   [float16 gyro_z]    (rad/s)

A `NmeaImuAdapter` sub-variant parses the $PMIMU NMEA sentence for
devices that stream ASCII.

`ImuIntegrator` converts a stream of ImuFrames into dead-reckoning
velocity and position estimates (useful as a fallback when GPS/odometry
is unavailable).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from typing import AsyncIterator, Optional

import numpy as np

from .base import FrameQueue
from .types import ImuFrame

log = logging.getLogger(__name__)

# Minimal binary frame: uint32 ts_us + 3×float32 accel + 3×float16 gyro
_FRAME_MAGIC = b"\xAA\x55"
_PAYLOAD_SIZE = 4 + 3 * 4 + 3 * 2  # 22 bytes
_FRAME_SIZE = len(_FRAME_MAGIC) + _PAYLOAD_SIZE + 1  # +1 checksum byte


def _parse_binary_frame(buf: bytes) -> Optional[tuple[int, np.ndarray, np.ndarray]]:
    """Returns (timestamp_ns, accel_ms2, gyro_rads) or None if malformed."""
    if len(buf) < _FRAME_SIZE or buf[:2] != _FRAME_MAGIC:
        return None
    payload = buf[2: 2 + _PAYLOAD_SIZE]
    checksum = buf[2 + _PAYLOAD_SIZE]
    if (sum(payload) & 0xFF) != checksum:
        return None
    ts_us, ax, ay, az = struct.unpack_from("<Ifff", payload, 0)
    # float16 gyro
    gx, gy, gz = np.frombuffer(payload[16:22], dtype=np.float16).astype(np.float64)
    ts_ns = ts_us * 1_000
    accel = np.array([ax, ay, az], dtype=np.float64)
    gyro = np.array([gx, gy, gz], dtype=np.float64)
    return ts_ns, accel, gyro


class SerialImuAdapter:
    """
    Reads IMU data from a UART serial port.

    The device must stream the binary frame format described above.
    For NMEA-based devices use NmeaImuAdapter instead.
    """

    def __init__(
        self,
        sensor_id: str = "imu_0",
        port: str = "/dev/ttyTHS0",   # Jetson Orin Nano UART
        baudrate: int = 921_600,
        queue_maxsize: int = 200,
    ) -> None:
        self.sensor_id = sensor_id
        self._port = port
        self._baudrate = baudrate
        self._queue: FrameQueue[ImuFrame] = FrameQueue(maxsize=queue_maxsize)
        self._serial = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError("pyserial is required: pip install pyserial") from exc

        self._queue.bind_loop(asyncio.get_running_loop())
        self._serial = serial.Serial(self._port, self._baudrate, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"imu-{self.sensor_id}"
        )
        self._thread.start()
        log.info("IMU[%s]: listening on %s @ %d baud", self.sensor_id, self._port, self._baudrate)

    def _recv_loop(self) -> None:
        buf = b""
        while self._running:
            chunk = self._serial.read(64)
            if not chunk:
                continue
            buf += chunk

            # Scan for frame magic
            while len(buf) >= _FRAME_SIZE:
                idx = buf.find(_FRAME_MAGIC)
                if idx == -1:
                    buf = buf[-1:]
                    break
                if idx > 0:
                    buf = buf[idx:]

                result = _parse_binary_frame(buf)
                if result is None:
                    buf = buf[1:]
                    continue

                ts_ns, accel, gyro = result
                self._queue.put_threadsafe(
                    ImuFrame(
                        sensor_id=self.sensor_id,
                        timestamp_ns=ts_ns,
                        accel_ms2=accel,
                        gyro_rads=gyro,
                        frame_index=self._frame_index,
                    )
                )
                self._frame_index += 1
                buf = buf[_FRAME_SIZE:]

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("IMU[%s]: stopped", self.sensor_id)

    async def frames(self) -> AsyncIterator[ImuFrame]:
        while self._running:
            yield await self._queue.get()


class ImuIntegrator:
    """
    Dead-reckoning position and velocity estimator from IMU frames.

    Integrates acceleration (after gravity removal) to velocity, then
    velocity to position.  Drift accumulates over time — this is a
    short-horizon fallback only, not a navigation solution.
    """

    GRAVITY = np.array([0.0, 0.0, 9.81])

    def __init__(self) -> None:
        self.velocity_ms: np.ndarray = np.zeros(3)
        self.position_m: np.ndarray = np.zeros(3)
        self._last_ts_ns: Optional[int] = None

    def update(self, frame: ImuFrame) -> None:
        if self._last_ts_ns is None:
            self._last_ts_ns = frame.timestamp_ns
            return

        dt = (frame.timestamp_ns - self._last_ts_ns) / 1e9
        self._last_ts_ns = frame.timestamp_ns

        # Remove gravity (assumes sensor is nominally upright)
        linear_accel = frame.accel_ms2 - self.GRAVITY
        self.velocity_ms = self.velocity_ms + linear_accel * dt
        self.position_m = self.position_m + self.velocity_ms * dt

    def reset(self) -> None:
        self.velocity_ms = np.zeros(3)
        self.position_m = np.zeros(3)
        self._last_ts_ns = None
