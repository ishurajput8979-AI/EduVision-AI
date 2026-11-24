"""Face enrollment, recognition, and liveness helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from database import IMAGES_DIR, VIDEOS_DIR
from face_detection.liveness_detector import LivenessDetector

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_FACE_SIZE = 60
MIN_IMAGE_SHARPNESS = 35.0

_liveness_detector: LivenessDetector | None = None


def get_liveness_detector() -> LivenessDetector:
    """Return a shared liveness detector instance."""
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector


def _require_face_dependencies() -> None:
    """Fail fast with a clear message when face libraries are missing."""
    if cv2 is None or face_recognition is None:
        raise RuntimeError(
            "OpenCV and face_recognition are required for face enrollment and attendance."
        )


def ensure_student_media_directories(student_erp: str) -> tuple[Path, Path]:
    """Create per-student image and video folders."""
    image_dir = IMAGES_DIR / student_erp
    video_dir = VIDEOS_DIR / student_erp
    image_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, video_dir


def allowed_image(filename: str) -> bool:
    """Validate still-image upload extension."""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video(filename: str) -> bool:
    """Validate video upload extension."""
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def save_photo_samples(student_erp: str, photos: list[FileStorage]) -> list[Path]:
    """Persist uploaded student photos."""
    _require_face_dependencies()
    image_dir, _ = ensure_student_media_directories(student_erp)
    saved_paths: list[Path] = []
    for index, photo in enumerate(photos, start=1):
        if not photo or not photo.filename or not allowed_image(photo.filename):
            continue
        destination = image_dir / secure_filename(f"{index:02d}_{photo.filename}")
        photo.save(destination)
        saved_paths.append(destination)
    return saved_paths


def save_video_sample(student_erp: str, video: FileStorage) -> Path:
    """Persist uploaded enrollment video."""
    _require_face_dependencies()
    _, video_dir = ensure_student_media_directories(student_erp)
    destination = video_dir / secure_filename(video.filename or "enrollment_video.mp4")
    video.save(destination)
    return destination


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 15,
    max_frames: int = 25,
) -> list[Path]:
    """Extract evenly spaced frames from a video."""
    _require_face_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    saved_frames: list[Path] = []
    frame_index = 0

    while capture.isOpened():
        success, frame = capture.read()
        if not success or len(saved_frames) >= max_frames:
            break
        if frame_index % frame_interval == 0:
            frame_name = output_dir / f"video_frame_{len(saved_frames) + 1:02d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved_frames.append(frame_name)
        frame_index += 1

    capture.release()
    return saved_frames


def extract_face_embedding(image_path: Path) -> list[float] | None:
    """Generate a 128D embedding from a single clear face image."""
    _require_face_dependencies()
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(grayscale, cv2.CV_64F).var()) < MIN_IMAGE_SHARPNESS:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_image, model="hog")
    if len(locations) != 1:
        return None
    top, right, bottom, left = locations[0]
    if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
        return None
    encodings = face_recognition.face_encodings(rgb_image, locations, num_jitters=1)
    if len(encodings) != 1:
        return None

    vector = np.array(encodings[0], dtype=np.float32)
    if vector.size != 128 or not np.isfinite(vector).all():
        return None
    return vector.tolist()


def build_embeddings_from_samples(sample_paths: list[Path]) -> list[dict[str, Any]]:
    """Generate valid embeddings from photo and frame samples."""
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for sample_path in sample_paths:
        normalized_path = str(sample_path)
        if normalized_path in seen_paths:
            continue
        embedding = extract_face_embedding(sample_path)
        if embedding is None:
            continue
        seen_paths.add(normalized_path)
        records.append({"sample_path": normalized_path, "embedding": embedding})
    return records


def build_embedding_index(face_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert stored embeddings into a fast NumPy index."""
    if not face_records:
        return {"matrix": np.empty((0, 128), dtype=np.float32), "labels": []}

    rows: list[list[float]] = []
    labels: list[dict[str, Any]] = []
    for record in face_records:
        embedding = json.loads(record["embedding_json"])
        vector = np.array(embedding, dtype=np.float32)
        if vector.size != 128 or not np.isfinite(vector).all():
            continue
        rows.append(vector.tolist())
        labels.append(
            {
                "student_erp": record["student_erp"],
                "name": record["name"],
                "sample_path": record["sample_path"],
            }
        )
    if not rows:
        return {"matrix": np.empty((0, 128), dtype=np.float32), "labels": []}
    return {"matrix": np.array(rows, dtype=np.float32), "labels": labels}


def extract_encodings_from_frame(
    frame: Any,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Extract face locations and embeddings from a frame."""
    _require_face_dependencies()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_frame, locations, num_jitters=1)
    return locations, encodings


def match_embedding(
    encoding: np.ndarray,
    index: dict[str, Any],
    threshold: float = 0.5,
    min_margin: float = 0.03,
) -> dict[str, Any]:
    """Match a live embedding against known faces."""
    matrix = index["matrix"]
    labels = index["labels"]
    if matrix.size == 0:
        return {"matched": False, "name": "Unknown", "student_erp": None, "distance": None}

    distances = np.linalg.norm(matrix - encoding, axis=1)
    student_distances: dict[str, dict[str, Any]] = {}
    for index_position, distance in enumerate(distances):
        label = labels[index_position]
        erp_id = label["student_erp"]
        if erp_id not in student_distances or distance < student_distances[erp_id]["distance"]:
            student_distances[erp_id] = {"distance": float(distance), "label": label}

    ranked = sorted(student_distances.values(), key=lambda item: item["distance"])
    best = ranked[0]
    best_distance = float(best["distance"])
    second_best = float(ranked[1]["distance"]) if len(ranked) > 1 else 1.0
    margin = second_best - best_distance

    if best_distance >= threshold or margin < min_margin:
        return {
            "matched": False,
            "name": "Unknown",
            "student_erp": None,
            "distance": best_distance,
            "confidence": 0.0,
        }

    confidence = max(0.0, min(1.0, 1 - (best_distance / max(threshold, 1e-6))))
    return {
        "matched": True,
        "name": best["label"]["name"],
        "student_erp": best["label"]["student_erp"],
        "distance": best_distance,
        "confidence": round(confidence, 3),
    }


def recognize_faces_in_frame(
    frame: Any,
    index: dict[str, Any],
    threshold: float = 0.5,
    check_liveness: bool = False,
    liveness_threshold: float = 0.6,
    enforce_liveness: bool = True,
) -> list[dict[str, Any]]:
    """Recognize faces in a frame and optionally perform single-frame liveness quality checks."""
    locations, encodings = extract_encodings_from_frame(frame)
    matches: list[dict[str, Any]] = []
    liveness_detector = get_liveness_detector() if check_liveness else None

    for location, encoding in zip(locations, encodings):
        result = match_embedding(encoding, index, threshold=threshold)
        top, right, bottom, left = location
        liveness_score = 1.0
        is_live = True

        if check_liveness and result.get("matched") and liveness_detector:
            face_region = frame[top:bottom, left:right]
            texture = liveness_detector.analyze_texture(face_region)
            liveness_score = float(texture.get("texture_score", 0.0))
            is_live = liveness_score >= liveness_threshold and not texture.get(
                "blur_detected", True
            )
            if not is_live and enforce_liveness:
                result["matched"] = False
                result["reason"] = "Face quality or spoof check failed."
            elif not is_live:
                result["reason"] = "Face quality is low; recognition kept but recapture is recommended."

        matches.append(
            {
                **result,
                "liveness_score": liveness_score,
                "is_live": is_live,
                "box": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                },
            }
        )
    return matches


def extract_face_region(frame: Any, box: dict[str, int]) -> Any | None:
    """Safely crop a face region from a frame."""
    top = max(0, int(box["top"]))
    right = max(0, int(box["right"]))
    bottom = max(0, int(box["bottom"]))
    left = max(0, int(box["left"]))
    face_region = frame[top:bottom, left:right]
    if face_region is None or getattr(face_region, "size", 0) == 0:
        return None
    return face_region.copy()


def append_face_sample(
    buffers: dict[str, list[Any]],
    student_erp: str,
    face_region: Any | None,
    max_samples: int = 8,
) -> None:
    """Append face crops for movement/blink-based liveness analysis."""
    if face_region is None:
        return
    bucket = buffers.setdefault(student_erp, [])
    bucket.append(face_region)
    if len(bucket) > max_samples:
        del bucket[:-max_samples]


def evaluate_frame_sequence_liveness(
    frames: list[Any],
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Run a simple blink and movement-based liveness check on a face frame sequence."""
    detector = get_liveness_detector()
    if len(frames) < detector.min_frames_for_analysis:
        return {
            "liveness_score": 0.0,
            "is_likely_live": False,
            "confidence": "low",
            "reasons": [
                f"Need at least {detector.min_frames_for_analysis} face frames for liveness."
            ],
        }

    result = detector.check_liveness_score(frames[-detector.min_frames_for_analysis :])
    score = float(result.get("liveness_score", 0.0))
    movement_detected = bool(result.get("movement_info", {}).get("movement_detected"))
    result["is_likely_live"] = bool(
        score >= threshold or (movement_detected and score >= max(0.25, threshold - 0.1))
    )
    if result["is_likely_live"]:
        result["recommendation"] = "PASS - Movement or blink evidence looks live enough."
    return result


def evaluate_saved_frame_paths(
    frame_paths: list[Path],
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Run liveness on saved frame paths extracted from an enrollment video."""
    _require_face_dependencies()
    frames = [cv2.imread(str(path)) for path in frame_paths]
    valid_frames = [frame for frame in frames if frame is not None]
    return evaluate_frame_sequence_liveness(valid_frames, threshold=threshold)
