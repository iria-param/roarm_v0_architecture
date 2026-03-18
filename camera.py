"""
camera.py — OpenCV webcam capture and base64 JPEG encoding.

Usage:
    cam = CameraCapture(device_index=0, width=640, height=480, jpeg_quality=85)
    frame_b64, frame_np = cam.grab_frame_b64()
    cam.release()
"""

import base64
import logging
import time
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class CameraCapture:
    """Wraps an OpenCV VideoCapture for single-frame grabs with base64 encoding."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 5,
        jpeg_quality: int = 85,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._cap: cv2.VideoCapture | None = None
        self._open()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the capture device and configure resolution / fps."""
        logger.info("Opening camera device index=%d", self.device_index)
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
        if not cap.isOpened():
            # Fallback without backend hint
            cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera at device index {self.device_index}. "
                "Check USB connection and device_index in config.yaml."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        self._cap = cap
        logger.info(
            "Camera opened: %dx%d @ %d fps",
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grab_frame(self) -> np.ndarray:
        """
        Capture and return a single BGR frame as a numpy array.

        Raises:
            CameraError: If the frame cannot be read.
        """
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Camera is not open. Call _open() first.")

        # Drain the buffer so we get a fresh frame (important at low fps)
        for _ in range(2):
            ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to read frame from camera.")

        return frame  # BGR uint8 ndarray

    def grab_frame_b64(self) -> Tuple[str, np.ndarray]:
        """
        Capture a JPEG-encoded frame and return:
            (base64_string, raw_bgr_ndarray)

        The base64 string is suitable for direct inclusion in a Gemini API
        `inline_data` part with mime_type="image/jpeg".
        """
        frame = self.grab_frame()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            raise CameraError("JPEG encoding failed.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        logger.debug("Frame captured: %d bytes base64", len(b64))
        return b64, frame

    def save_frame(self, frame: np.ndarray, path: str) -> None:
        """Save a raw BGR frame to disk (useful for debug / calibration)."""
        cv2.imwrite(path, frame)
        logger.debug("Frame saved to %s", path)

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python camera.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Camera smoke-test — press any key to exit.")
    with CameraCapture(device_index=0) as cam:
        b64, frame = cam.grab_frame_b64()
        print(f"  Frame shape : {frame.shape}")
        print(f"  Base64 size : {len(b64)} chars")
        cv2.imshow("RoArm Camera Test", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.")
