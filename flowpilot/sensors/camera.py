"""
Camera ingestion adapter.

Capture paths (selected at construction time):
  1. Jetson CSI  — nvarguscamerasrc → NVMM → nvvidconv → BGR → appsink
  2. Jetson USB  — nvv4l2src        → NVMM → nvvidconv → BGR → appsink
  3. Desktop     — standard OpenCV VideoCapture (V4L2 / DirectShow)

Paths 1 and 2 keep decoding on the Jetson ISP/VIC, avoiding a CPU-side
decode. The frame arrives at appsink as BGR in CPU memory; from there it
is optionally uploaded to CUDA (torch.Tensor on device).

True zero-copy (NVMM → CUDA without CPU touch) requires DeepStream pyds
or jetson-utils — that path is deferred to the Phase 7 optimisation pass.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

import numpy as np

from .base import FrameQueue
from .types import CameraFrame

log = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="camera")


def _build_jetson_pipeline(
    width: int, height: int, fps: int, source: str
) -> str:
    if source == "csi":
        src = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1,format=NV12"
        )
    else:
        src = (
            f"nvv4l2src device=/dev/video0 ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1"
        )
    return (
        f"{src} ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


def _build_desktop_pipeline(width: int, height: int, fps: int) -> str:
    return (
        f"v4l2src ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


class CameraAdapter:
    """
    Async camera adapter.  Capture runs in a daemon thread; frames are
    delivered to async consumers via FrameQueue.
    """

    def __init__(
        self,
        sensor_id: str = "camera_0",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        device_index: int = 0,
        # Set one of these to True on Jetson
        jetson_csi: bool = False,
        jetson_usb: bool = False,
        # Upload captured frame to CUDA after capture
        cuda_upload: bool = False,
        queue_maxsize: int = 60,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self._device_index = device_index
        self._jetson_csi = jetson_csi
        self._jetson_usb = jetson_usb
        self._cuda_upload = cuda_upload
        self._queue: FrameQueue[CameraFrame] = FrameQueue(maxsize=queue_maxsize)
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_index = 0

    async def start(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python-headless is required for CameraAdapter") from e

        self._queue.bind_loop(asyncio.get_running_loop())

        if self._jetson_csi:
            pipeline = _build_jetson_pipeline(self.width, self.height, self.fps, "csi")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            log.info("Camera[%s]: Jetson CSI GStreamer pipeline", self.sensor_id)
        elif self._jetson_usb:
            pipeline = _build_jetson_pipeline(self.width, self.height, self.fps, "usb")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            log.info("Camera[%s]: Jetson USB GStreamer pipeline", self.sensor_id)
        else:
            self._cap = cv2.VideoCapture(self._device_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            log.info("Camera[%s]: OpenCV capture (device %d)", self.sensor_id, self._device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"CameraAdapter[{self.sensor_id}]: failed to open capture device"
            )

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"camera-{self.sensor_id}"
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        import cv2  # noqa: F401 — already imported in start()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("Camera[%s]: read failed", self.sensor_id)
                continue

            ts_ns = time.time_ns()

            if self._cuda_upload:
                try:
                    import torch
                    # Upload to GPU; non_blocking keeps the CPU thread moving
                    image: np.ndarray = torch.from_numpy(  # type: ignore[assignment]
                        frame.copy()
                    ).cuda(non_blocking=True)
                except Exception as exc:
                    log.warning("Camera CUDA upload failed (%s), using CPU frame", exc)
                    image = frame
            else:
                image = frame

            self._queue.put_threadsafe(
                CameraFrame(
                    sensor_id=self.sensor_id,
                    timestamp_ns=ts_ns,
                    image=image,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    frame_index=self._frame_index,
                )
            )
            self._frame_index += 1

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        log.info("Camera[%s]: stopped", self.sensor_id)

    async def frames(self) -> AsyncIterator[CameraFrame]:
        while self._running:
            yield await self._queue.get()
