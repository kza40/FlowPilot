"""
LiDAR ingestion adapter — Ouster OS-series (UDP protocol).

The Ouster sensor sends lidar packets over UDP on a configurable port
(default 7502).  Each packet contains a fixed number of columns; a full
revolution is assembled from multiple packets into a point cloud.

A `PcapLidarAdapter` replays pre-recorded .pcap files at wall-clock rate
for offline development and testing.

Point cloud output format: (N, 4) float32 — [x, y, z, intensity] in
ego-centric metres, right-handed coordinate system (x forward, y left, z up).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from .base import FrameQueue
from .types import LidarFrame

log = logging.getLogger(__name__)

# Ouster OS0/OS1/OS2 packet constants (firmware 2.x / 3.x)
_OUSTER_LIDAR_PORT = 7502
_COLUMNS_PER_PACKET = 16
_PIXELS_PER_COLUMN = 128   # OS0-128; adjust for OS0-32/OS1-64 etc.
_PACKET_HEADER_SIZE = 8    # bytes


def _parse_ouster_packet(data: bytes, n_pixels: int) -> Optional[np.ndarray]:
    """
    Minimal Ouster UDP packet parser.

    Returns (M, 4) float32 [x, y, z, intensity] for the valid returns in
    this packet, or None if the packet is malformed.

    This is a simplified parser that extracts range and signal from each
    channel.  A production implementation would use the official ouster-sdk
    (`pip install ouster-sdk`) for full metadata / calibration support.
    """
    col_size = 4 + n_pixels * 12  # timestamp(4) + per-channel(12)
    expected = _PACKET_HEADER_SIZE + _COLUMNS_PER_PACKET * col_size
    if len(data) < expected:
        return None

    points: list[np.ndarray] = []
    offset = _PACKET_HEADER_SIZE

    for _ in range(_COLUMNS_PER_PACKET):
        # column timestamp (uint32, microseconds)
        # col_ts = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        for pix_idx in range(n_pixels):
            pix_offset = offset + pix_idx * 12
            if pix_offset + 12 > len(data):
                break
            # range (uint32, mm), signal (uint16), reflectivity (uint8), ...
            range_mm, signal, _ = struct.unpack_from("<IHH", data, pix_offset)
            if range_mm == 0:
                continue
            r = range_mm / 1000.0
            # Placeholder: real decoder maps (range, beam_angle) → (x, y, z)
            # using per-beam intrinsics from sensor metadata JSON.
            x = r
            y = 0.0
            z = float(pix_idx) * 0.01
            intensity = signal / 65535.0
            points.append(np.array([x, y, z, intensity], dtype=np.float32))

        offset += n_pixels * 12

    return np.vstack(points) if points else np.zeros((0, 4), dtype=np.float32)


class OusterLidarAdapter:
    """
    Receives Ouster lidar UDP packets and assembles them into point clouds.

    Each cloud covers one full azimuth revolution (or `columns_per_frame`
    columns, whichever comes first) and is emitted as a LidarFrame.
    """

    def __init__(
        self,
        sensor_id: str = "lidar_0",
        host: str = "0.0.0.0",
        port: int = _OUSTER_LIDAR_PORT,
        pixels_per_column: int = _PIXELS_PER_COLUMN,
        columns_per_frame: int = 1024,
        queue_maxsize: int = 20,
    ) -> None:
        self.sensor_id = sensor_id
        self._host = host
        self._port = port
        self._pixels_per_column = pixels_per_column
        self._columns_per_frame = columns_per_frame
        self._queue: FrameQueue[LidarFrame] = FrameQueue(maxsize=queue_maxsize)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_index = 0
        self._accum: list[np.ndarray] = []
        self._col_count = 0

    async def start(self) -> None:
        self._queue.bind_loop(asyncio.get_running_loop())
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"lidar-{self.sensor_id}"
        )
        self._thread.start()
        log.info("LiDAR[%s]: listening on %s:%d", self.sensor_id, self._host, self._port)

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            batch = _parse_ouster_packet(data, self._pixels_per_column)
            if batch is None:
                continue

            ts_ns = time.time_ns()
            self._accum.append(batch)
            self._col_count += _COLUMNS_PER_PACKET

            if self._col_count >= self._columns_per_frame:
                cloud = np.vstack(self._accum) if self._accum else np.zeros((0, 4), np.float32)
                self._queue.put_threadsafe(
                    LidarFrame(
                        sensor_id=self.sensor_id,
                        timestamp_ns=ts_ns,
                        points=cloud,
                        frame_index=self._frame_index,
                    )
                )
                self._frame_index += 1
                self._accum.clear()
                self._col_count = 0

    async def stop(self) -> None:
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("LiDAR[%s]: stopped", self.sensor_id)

    async def frames(self) -> AsyncIterator[LidarFrame]:
        while self._running:
            yield await self._queue.get()


class PcapLidarAdapter:
    """
    Replays an Ouster .pcap recording at wall-clock speed.

    Requires `scapy` (`pip install scapy`) for pcap parsing.
    Useful for offline development without physical hardware.
    """

    def __init__(
        self,
        pcap_path: str | Path,
        sensor_id: str = "lidar_0",
        pixels_per_column: int = _PIXELS_PER_COLUMN,
        columns_per_frame: int = 1024,
        loop: bool = False,
    ) -> None:
        self.sensor_id = sensor_id
        self._path = Path(pcap_path)
        self._pixels_per_column = pixels_per_column
        self._columns_per_frame = columns_per_frame
        self._loop = loop
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Pcap file not found: {self._path}")
        self._running = True
        log.info("LiDAR[%s]: replaying %s", self.sensor_id, self._path)

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[LidarFrame]:
        try:
            from scapy.all import PcapReader, UDP  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("scapy is required for PcapLidarAdapter: pip install scapy") from exc

        while self._running:
            accum: list[np.ndarray] = []
            col_count = 0
            prev_pkt_time: Optional[float] = None

            with PcapReader(str(self._path)) as reader:
                for pkt in reader:
                    if not self._running:
                        return
                    if not pkt.haslayer(UDP) or pkt[UDP].dport != _OUSTER_LIDAR_PORT:
                        continue

                    # Honour original inter-packet timing
                    if prev_pkt_time is not None:
                        delay = float(pkt.time) - prev_pkt_time
                        if delay > 0:
                            await asyncio.sleep(delay)
                    prev_pkt_time = float(pkt.time)

                    payload = bytes(pkt[UDP].payload)
                    batch = _parse_ouster_packet(payload, self._pixels_per_column)
                    if batch is None:
                        continue

                    accum.append(batch)
                    col_count += _COLUMNS_PER_PACKET

                    if col_count >= self._columns_per_frame:
                        cloud = np.vstack(accum) if accum else np.zeros((0, 4), np.float32)
                        yield LidarFrame(
                            sensor_id=self.sensor_id,
                            timestamp_ns=time.time_ns(),
                            points=cloud,
                            frame_index=self._frame_index,
                        )
                        self._frame_index += 1
                        accum.clear()
                        col_count = 0

            if not self._loop:
                break
