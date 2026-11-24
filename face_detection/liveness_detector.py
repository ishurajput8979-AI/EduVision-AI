"""Face liveness detection helpers to reduce spoofing risk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import dlib
except ImportError:  # pragma: no cover
    dlib = None


class LivenessDetector:
    """Estimate whether a face likely belongs to a live person."""

    def __init__(self) -> None:
        """Initialize thresholds and optional dlib resources."""
        self.eye_aspect_ratio_threshold = 0.2
        self.mouth_aspect_ratio_threshold = 0.3
        self.min_frames_for_analysis = 5
        self.min_movement_for_liveness = 10
        self._eye_cascade = None
        self._landmark_detector = None
        self._landmark_predictor = None
        self.landmark_model_path = Path(
            os.environ.get("DLIB_LANDMARK_MODEL", "shape_predictor_68_face_landmarks.dat")
        )

    def _get_eye_cascade(self):
        """Load and cache the OpenCV eye detector."""
        if cv2 is None:
            return None
        if self._eye_cascade is None:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            if cascade.empty():
                return None
            self._eye_cascade = cascade
        return self._eye_cascade

    def detect_blink(self, frame: Any, face_landmarks: list[tuple] | None = None) -> bool:
        """Detect simple blink evidence by checking whether eyes are visible."""
        if cv2 is None or frame is None:
            return False

        cascade = self._get_eye_cascade()
        if cascade is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray, 1.3, 5)
        return len(eyes) > 0

    def detect_head_movement(self, frames: list[Any]) -> dict[str, Any]:
        """Detect coarse movement across consecutive frames."""
        if cv2 is None or len(frames) < 2:
            return {"movement_detected": False, "total_movement": 0.0, "movement_frames": 0}

        orb = cv2.ORB_create()
        total_movement = 0.0
        movement_frames = 0

        try:
            for current_frame, next_frame in zip(frames, frames[1:]):
                kp1, des1 = orb.detectAndCompute(current_frame, None)
                kp2, des2 = orb.detectAndCompute(next_frame, None)
                if des1 is None or des2 is None:
                    continue

                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                if not matches:
                    continue

                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                mean_movement = float(np.mean(np.linalg.norm(dst_pts - src_pts, axis=2)))
                total_movement += mean_movement
                if mean_movement > self.min_movement_for_liveness:
                    movement_frames += 1
        except Exception as exc:
            print(f"Movement detection error: {exc}")
            return {
                "movement_detected": False,
                "total_movement": 0.0,
                "movement_frames": 0,
                "error": str(exc),
            }

        return {
            "movement_detected": movement_frames >= 2,
            "total_movement": round(total_movement, 3),
            "movement_frames": movement_frames,
        }

    def detect_face_landmarks(self, frame: Any) -> list[tuple[int, int]] | None:
        """Detect facial landmarks when dlib and the model file are available."""
        if dlib is None or cv2 is None or frame is None or not self.landmark_model_path.exists():
            return None

        try:
            if self._landmark_detector is None:
                self._landmark_detector = dlib.get_frontal_face_detector()
            if self._landmark_predictor is None:
                self._landmark_predictor = dlib.shape_predictor(str(self.landmark_model_path))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._landmark_detector(gray)
            if not faces:
                return None
            landmarks = self._landmark_predictor(gray, faces[0])
            return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        except Exception as exc:
            print(f"Landmark detection error: {exc}")
            return None

    def analyze_texture(self, frame: Any) -> dict[str, Any]:
        """Estimate whether the face region has enough texture to be plausible."""
        if cv2 is None or frame is None:
            return {}

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return {
                "laplacian_variance": laplacian_var,
                "blur_detected": laplacian_var < 100,
                "texture_score": min(1.0, laplacian_var / 500),
            }
        except Exception as exc:
            print(f"Texture analysis error: {exc}")
            return {}

    def check_liveness_score(self, frames: list[Any]) -> dict[str, Any]:
        """Combine blink, movement, and texture signals into one score."""
        if len(frames) < self.min_frames_for_analysis:
            return {
                "liveness_score": 0.0,
                "is_likely_live": False,
                "error": f"Need at least {self.min_frames_for_analysis} frames",
            }

        reasons: list[str] = []
        scores: dict[str, float] = {}

        blink_count = sum(1 for frame in frames if self.detect_blink(frame))
        scores["blink"] = float(min(40, blink_count * 8))
        reasons.append(
            f"Blinks detected: {blink_count}" if blink_count >= 2 else f"Few blinks detected: {blink_count}"
        )

        movement = self.detect_head_movement(frames)
        scores["movement"] = 30.0 if movement.get("movement_detected") else 0.0
        reasons.append(
            "Natural head movement detected"
            if movement.get("movement_detected")
            else "No clear head movement detected"
        )

        texture_scores = [self.analyze_texture(frame) for frame in frames]
        avg_texture = (
            sum(item.get("texture_score", 0.0) for item in texture_scores) / len(texture_scores)
        )
        scores["texture"] = float(min(30, avg_texture * 40))
        reasons.append(
            "Good frame texture"
            if not any(item.get("blur_detected", False) for item in texture_scores)
            else "Some frames appear blurry"
        )

        total_score = sum(scores.values())
        liveness_score = min(1.0, total_score / 100.0)
        is_likely_live = liveness_score >= 0.6

        return {
            "liveness_score": float(liveness_score),
            "is_likely_live": is_likely_live,
            "confidence": (
                "high" if liveness_score > 0.8 else "medium" if liveness_score > 0.5 else "low"
            ),
            "scores": scores,
            "reasons": reasons,
            "movement_info": movement,
            "recommendation": (
                "PASS - Person appears to be live."
                if is_likely_live
                else "REJECT - Likely spoof or poor capture; request a retake."
            ),
        }


"""
# Example usage
detector = LivenessDetector()
result = detector.check_liveness_score(frames)
print(result)
"""
