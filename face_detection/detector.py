"""Face detection helpers built on top of OpenCV Haar cascades."""

from __future__ import annotations

from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def detect_faces_opencv(image: Any) -> list[tuple[int, int, int, int]]:
    """Detect faces in an image using OpenCV's built-in Haar cascade detector."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt first.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
