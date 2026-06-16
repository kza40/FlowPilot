"""
Radar ingestion adapter — SocketCAN interface.

Reads CAN frames from a SocketCAN network interface (Linux only) using
the `python-can` library.  CAN message parsing is delegated to a
pluggable `RadarParser` so vendor-specific protocols can be swapped
without changing the adapter.

Design note: SocketCAN is preferred over vendor SDKs because it is
kernel-native, requires no proprietary license, and its latency is
directly measurable with standard tools (candump, python-can).

Usage:
    adapter = SocketCanRadarAdapter(
        channel="can0",
        parser=ContiArs430Parser(),   # your radar's CAN DBC parser
    )
    await adapter.start()
    async for frame in adapter.frames():
        process(frame)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncIterator, Optional, Protocol

import numpy as np

from .base import FrameQueue
from .types import RadarDetection, RadarFrame

log = logging.getLogger(__name__)


class RadarParser(Protocol):
    """
    Pluggable CAN message → RadarDetection converter.

    Implement this for your specific radar's DBC spec.
    """

    def parse(self, arbitration_id: int, data: bytes) -> Optional[RadarDetection]:
        """
        Returns a RadarDetection if `data` on `arbitration_id` encodes one,
        otherwise returns None (message is for a different signal / ignored).
        """
        ...

    def is_frame_boundary(self, arbitration_id: int) -> bool:
        """
        Returns True when this CAN ID signals the end of one radar scan
        cycle (i.e., all detections for this frame have been sent).
        """
        ...


class PassthroughRadarParser:
    """
    Minimal parser that decodes a flat 8-byte CAN message with the layout:
        [range_m (f32)] [velocity_ms (f32)]   — azimuth/elevation = 0
    Useful for loopback / synthetic testing.
    """

    DETECTION_ID = 0x100
    BOUNDARY_ID = 0x1FF

    def parse(self, arbitration_id: int, data: bytes) -> Optional[RadarDetection]:
        if arbitration_id != self.DETECTION_ID or len(data) < 8:
            return None
        import struct
        range_m, velocity_ms = struct.unpack_from("<ff", data)
        return RadarDetection(
            range_m=range_m,
            azimuth_rad=0.0,
            elevation_rad=0.0,
            velocity_ms=velocity_ms,
            rcs_db=0.0,
        )

    def is_frame_boundary(self, arbitration_id: int) -> bool:
        return arbitration_id == self.BOUNDARY_ID


class SocketCanRadarAdapter:
    """
    Reads radar CAN frames from a SocketCAN interface.

    Detections within one scan cycle are accumulated until a frame-boundary
    message arrives (as defined by the parser), then emitted as a RadarFrame.

    Requires Linux + SocketCAN driver:
        sudo ip link set can0 up type can bitrate 500000
    """

    def __init__(
        self,
        sensor_id: str = "radar_0",
        channel: str = "can0",
        bitrate: int = 500_000,
        parser: Optional[RadarParser] = None,
        queue_maxsize: int = 40,
    ) -> None:
        self.sensor_id = sensor_id
        self._channel = channel
        self._bitrate = bitrate
        self._parser = parser or PassthroughRadarParser()
        self._queue: FrameQueue[RadarFrame] = FrameQueue(maxsize=queue_maxsize)
        self._bus = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        try:
            import can
        except ImportError as exc:
            raise ImportError("python-can is required: pip install python-can") from exc

        self._queue.bind_loop(asyncio.get_running_loop())
        self._bus = can.interface.Bus(
            channel=self._channel,
            interface="socketcan",
            bitrate=self._bitrate,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"radar-{self.sensor_id}"
        )
        self._thread.start()
        log.info("Radar[%s]: listening on %s @ %d bps", self.sensor_id, self._channel, self._bitrate)

    def _recv_loop(self) -> None:
        pending: list[RadarDetection] = []

        while self._running:
            msg = self._bus.recv(timeout=0.1)
            if msg is None:
                continue

            if self._parser.is_frame_boundary(msg.arbitration_id):
                self._queue.put_threadsafe(
                    RadarFrame(
                        sensor_id=self.sensor_id,
                        timestamp_ns=time.time_ns(),
                        detections=list(pending),
                        frame_index=self._frame_index,
                    )
                )
                self._frame_index += 1
                pending.clear()
            else:
                detection = self._parser.parse(msg.arbitration_id, bytes(msg.data))
                if detection is not None:
                    pending.append(detection)

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._bus:
            self._bus.shutdown()
            self._bus = None
        log.info("Radar[%s]: stopped", self.sensor_id)

    async def frames(self) -> AsyncIterator[RadarFrame]:
        while self._running:
            yield await self._queue.get()
