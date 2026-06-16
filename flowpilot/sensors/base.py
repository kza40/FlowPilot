"""SensorAdapter protocol and thread-safe FrameQueue used by all adapters."""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Generic, Protocol, TypeVar, runtime_checkable

from .types import CameraFrame, ImuFrame, LidarFrame, RadarFrame

AnyFrame = CameraFrame | LidarFrame | RadarFrame | ImuFrame
F = TypeVar("F", CameraFrame, LidarFrame, RadarFrame, ImuFrame)


@runtime_checkable
class SensorAdapter(Protocol):
    """Common interface every sensor adapter must satisfy."""

    sensor_id: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def frames(self) -> AsyncIterator[AnyFrame]: ...


class FrameQueue(Generic[F]):
    """
    Delivers frames produced by a background thread to asyncio consumers.

    The producer calls `put_threadsafe` from any thread.
    The consumer calls `get` from an asyncio coroutine.

    When the queue is full (slow consumer), the oldest frame is dropped so
    real-time latency is preserved over completeness.
    """

    def __init__(self, maxsize: int = 60) -> None:
        self._queue: asyncio.Queue[F] = asyncio.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._latest: F | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def put_threadsafe(self, frame: F) -> None:
        with self._lock:
            self._latest = frame
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._enqueue, frame)

    def _enqueue(self, frame: F) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()   # drop oldest
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(frame)

    def latest(self) -> F | None:
        with self._lock:
            return self._latest

    async def get(self) -> F:
        return await self._queue.get()
