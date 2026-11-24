"""Attendance capture and face-marking services."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from cache_manager import cache_manager
from database import (
    ACADEMIC_EXCLUDED_DAY_TYPES,
    ATTENDANCE_MEDIA_DIR,
    get_academic_calendar_day_status,
    get_assignment_by_id,
    get_face_embeddings,
    get_section_roster,
    teacher_can_take_attendance,
    upsert_attendance,
)
from services.face import (
    allowed_image,
    allowed_video,
    append_face_sample,
    build_embedding_index,
    evaluate_frame_sequence_liveness,
    extract_face_region,
    extract_frames_from_video,
    recognize_faces_in_frame,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def save_attendance_media(
    upload: FileStorage,
    teacher_id: int,
    assignment_id: int,
    attendance_date: str,
) -> Path:
    """Persist teacher-uploaded attendance media."""
    teacher_dir = ATTENDANCE_MEDIA_DIR / f"teacher_{teacher_id}" / f"assignment_{assignment_id}"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    file_name = secure_filename(upload.filename or f"{attendance_date}_capture.jpg")
    file_path = teacher_dir / file_name
    upload.save(file_path)
    return file_path


def _serialize_embedding_index(index: dict[str, Any]) -> dict[str, Any]:
    """Convert the face index to a cache-safe structure."""
    return {
        "matrix": index["matrix"].tolist(),
        "labels": index["labels"],
    }


def _hydrate_embedding_index(cached_index: Any) -> dict[str, Any] | None:
    """Restore a cached face index into the NumPy structure used by recognition."""
    if not cached_index:
        return None
    matrix = cached_index.get("matrix", [])
    labels = cached_index.get("labels", [])
    if hasattr(matrix, "size"):
        return {"matrix": matrix, "labels": labels}
    if isinstance(matrix, str):
        return None
    return {
        "matrix": np.array(matrix, dtype=np.float32),
        "labels": labels,
    }


def _build_roster_context(
    assignment_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load assignment, roster, and face index."""
    assignment = get_assignment_by_id(assignment_id)
    if not assignment:
        raise ValueError("Teacher-subject assignment not found.")

    roster = get_section_roster(
        assignment["course"],
        assignment["branch"],
        int(assignment["year"]),
        int(assignment["semester"]),
        assignment["section"],
    )
    if not roster:
        raise ValueError("No students found in the selected subject section.")

    cache_key = f"embedding_index_{assignment['subject_id']}_{assignment['section']}"
    index = _hydrate_embedding_index(cache_manager.get(cache_key))
    if index is None:
        face_records = get_face_embeddings([student["erp_id"] for student in roster])
        index = build_embedding_index(face_records)
        cache_manager.set(cache_key, _serialize_embedding_index(index), ttl_seconds=3600)
    return assignment, roster, index


def _record_match(
    match: dict[str, Any],
    recognized_map: dict[str, float],
) -> None:
    """Store the best distance seen for a recognized student."""
    current_best = recognized_map.get(match["student_erp"], 1.0)
    recognized_map[match["student_erp"]] = min(current_best, float(match["distance"]))


def _collect_matches_from_frame(
    frame: Any,
    recognized_map: dict[str, float],
    unknown_count: int,
    index: dict[str, Any],
    threshold: float,
    face_buffers: dict[str, list[Any]] | None = None,
    check_liveness: bool = False,
    liveness_threshold: float = 0.6,
    liveness_details: dict[str, Any] | None = None,
    enforce_liveness: bool = True,
) -> int:
    """Update recognition results from a single frame."""
    matches = recognize_faces_in_frame(
        frame,
        index,
        threshold=threshold,
        check_liveness=check_liveness,
        liveness_threshold=liveness_threshold,
        enforce_liveness=enforce_liveness,
    )
    for match in matches:
        if match["matched"]:
            _record_match(match, recognized_map)
            if face_buffers is not None:
                append_face_sample(
                    face_buffers,
                    match["student_erp"],
                    extract_face_region(frame, match["box"]),
                )
        else:
            unknown_count += 1
        if liveness_details is not None and match.get("student_erp"):
            liveness_details[match["student_erp"]] = {
                "liveness_score": float(match.get("liveness_score", 0.0) or 0.0),
                "is_likely_live": bool(match.get("is_live")),
                "confidence": "medium" if match.get("is_live") else "low",
                "reasons": [match["reason"]] if match.get("reason") else [],
            }
    return unknown_count


def _ensure_attendance_date_allowed(
    assignment: dict[str, Any],
    attendance_date: str,
) -> None:
    """Block attendance capture only for academic dates that explicitly exclude classes."""
    excluded_day = get_academic_calendar_day_status(
        assignment["course"],
        assignment["branch"],
        int(assignment["year"]),
        int(assignment["semester"]),
        attendance_date,
    )
    if not excluded_day:
        return
    if excluded_day["day_type"] not in ACADEMIC_EXCLUDED_DAY_TYPES:
        return
    day_label = excluded_day["day_type"].replace("_", " ")
    reason = (excluded_day.get("reason") or "").strip()
    reason_suffix = f" Reason: {reason}." if reason else ""
    raise ValueError(
        f"Attendance cannot be marked on {attendance_date} because it is declared as {day_label}.{reason_suffix}"
    )


def _apply_liveness_filter(
    recognized_map: dict[str, float],
    face_buffers: dict[str, list[Any]],
    liveness_threshold: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Keep only identities that pass sequence liveness."""
    filtered_map: dict[str, float] = {}
    liveness_details: dict[str, Any] = {}
    for student_erp, distance in recognized_map.items():
        frames = face_buffers.get(student_erp, [])
        result = evaluate_frame_sequence_liveness(frames, threshold=liveness_threshold)
        liveness_details[student_erp] = result
        if result.get("is_likely_live"):
            filtered_map[student_erp] = distance
    return filtered_map, liveness_details


def process_media_file(
    file_path: Path,
    index: dict[str, Any],
    threshold: float = 0.5,
    enable_liveness: bool = True,
    liveness_threshold: float = 0.35,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    """Recognize faces from a still image or video clip."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt first.")

    recognized_map: dict[str, float] = {}
    unknown_count = 0
    liveness_details: dict[str, Any] = {}

    if allowed_image(file_path.name):
        frame = cv2.imread(str(file_path))
        if frame is None:
            raise ValueError("Uploaded image could not be processed.")
        unknown_count = _collect_matches_from_frame(
            frame,
            recognized_map,
            unknown_count,
            index,
            threshold,
            check_liveness=enable_liveness,
            liveness_threshold=liveness_threshold,
            liveness_details=liveness_details,
            enforce_liveness=False,
        )
        return recognized_map, unknown_count, liveness_details

    if allowed_video(file_path.name):
        face_buffers: dict[str, list[Any]] = {}
        extracted_dir = file_path.parent / f"{file_path.stem}_frames"
        frame_paths = extract_frames_from_video(file_path, extracted_dir)
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            unknown_count = _collect_matches_from_frame(
                frame,
                recognized_map,
                unknown_count,
                index,
                threshold,
                face_buffers=face_buffers,
            )
        if enable_liveness:
            recognized_map, liveness_details = _apply_liveness_filter(
                recognized_map,
                face_buffers,
                liveness_threshold=liveness_threshold,
            )
        return recognized_map, unknown_count, liveness_details

    raise ValueError(f"Unsupported attendance media format: {file_path.suffix.lower()}")


def _write_attendance_rows(
    assignment: dict[str, Any],
    roster: list[dict[str, Any]],
    recognized_map: dict[str, float],
    manual_present: list[str],
    attendance_date: str,
    source: str,
    effective_teacher_id: int,
) -> dict[str, Any]:
    """Persist attendance rows after automated recognition and manual overrides."""
    final_present = set(recognized_map.keys()) | set(manual_present)
    present_students: list[dict[str, Any]] = []
    absent_students: list[dict[str, Any]] = []

    for student in roster:
        erp_id = student["erp_id"]
        manual_override = int(erp_id in manual_present and erp_id not in recognized_map)
        if erp_id in final_present:
            present_students.append(student)
            upsert_attendance(
                student_erp=erp_id,
                subject_id=int(assignment["subject_id"]),
                teacher_id=int(effective_teacher_id),
                attendance_date=attendance_date,
                status="Present",
                confidence=recognized_map.get(erp_id),
                source=source,
                manual_override=manual_override,
            )
        else:
            absent_students.append(student)
            upsert_attendance(
                student_erp=erp_id,
                subject_id=int(assignment["subject_id"]),
                teacher_id=int(effective_teacher_id),
                attendance_date=attendance_date,
                status="Absent",
                confidence=None,
                source=source,
                manual_override=0,
            )

    cache_manager.invalidate(f"assignment_attendance_{int(assignment['id'])}")
    return {
        "subject_name": assignment["subject_name"],
        "subject_code": assignment["subject_code"],
        "attendance_date": attendance_date,
        "present_count": len(present_students),
        "absent_count": len(absent_students),
        "recognized_students": present_students,
        "absent_students": absent_students,
        "recognized_distances": recognized_map,
    }


def mark_attendance_from_upload(
    assignment_id: int,
    teacher_id: int,
    attendance_date: str | None,
    media_upload: FileStorage | None,
    manual_present: list[str] | None = None,
    threshold: float = 0.5,
    liveness_threshold: float = 0.35,
) -> dict[str, Any]:
    """Run attendance recognition from uploaded image or video."""
    assignment, roster, index = _build_roster_context(assignment_id)
    resolved_date = attendance_date or date.today().isoformat()
    if not teacher_can_take_attendance(int(teacher_id), int(assignment_id), resolved_date):
        raise PermissionError("You are not assigned to this subject.")
    _ensure_attendance_date_allowed(assignment, resolved_date)

    recognized_map: dict[str, float] = {}
    unknown_count = 0
    liveness_details: dict[str, Any] = {}
    manual_present = manual_present or []
    source = "manual"

    if media_upload and media_upload.filename:
        if index["matrix"].size == 0:
            raise ValueError(
                "No enrolled face data is available for this section. Enroll students first or use manual override only."
            )
        file_path = save_attendance_media(media_upload, teacher_id, assignment_id, resolved_date)
        recognized_map, unknown_count, liveness_details = process_media_file(
            file_path=file_path,
            index=index,
            threshold=threshold,
            enable_liveness=True,
            liveness_threshold=liveness_threshold,
        )
        source = file_path.suffix.lower().replace(".", "") or "upload"

    result = _write_attendance_rows(
        assignment=assignment,
        roster=roster,
        recognized_map=recognized_map,
        manual_present=manual_present,
        attendance_date=resolved_date,
        source=source,
        effective_teacher_id=int(teacher_id),
    )
    result["unknown_faces"] = unknown_count
    result["liveness"] = liveness_details
    return result


def run_live_camera_attendance(
    assignment_id: int,
    teacher_id: int,
    attendance_date: str | None = None,
    threshold: float = 0.5,
    frame_skip: int = 3,
    max_frames: int = 180,
    liveness_threshold: float = 0.6,
) -> dict[str, Any]:
    """Open the webcam, verify liveness, and mark attendance."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install requirements.txt first.")

    assignment, roster, index = _build_roster_context(assignment_id)
    resolved_date = attendance_date or date.today().isoformat()
    if not teacher_can_take_attendance(int(teacher_id), int(assignment_id), resolved_date):
        raise PermissionError("You are not assigned to this subject.")
    _ensure_attendance_date_allowed(assignment, resolved_date)
    if index["matrix"].size == 0:
        raise ValueError("No enrolled face data is available for this section. Enroll students first.")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to access the system camera.")

    recognized_map: dict[str, float] = {}
    face_buffers: dict[str, list[Any]] = {}
    liveness_details: dict[str, Any] = {}
    unknown_count = 0
    frame_counter = 0

    while capture.isOpened() and frame_counter < max_frames:
        success, frame = capture.read()
        if not success:
            break

        frame_counter += 1
        if frame_counter % frame_skip == 0:
            matches = recognize_faces_in_frame(frame, index, threshold=threshold)
            for match in matches:
                label = "Unknown"
                color = (0, 0, 255)
                if match["matched"]:
                    student_erp = match["student_erp"]
                    append_face_sample(
                        face_buffers,
                        student_erp,
                        extract_face_region(frame, match["box"]),
                    )
                    liveness_result = evaluate_frame_sequence_liveness(
                        face_buffers.get(student_erp, []),
                        threshold=liveness_threshold,
                    )
                    liveness_details[student_erp] = liveness_result
                    if liveness_result.get("is_likely_live"):
                        _record_match(match, recognized_map)
                        label = f"{match['name']} live"
                        color = (0, 180, 0)
                    else:
                        label = f"{match['name']} check movement/blink"
                        color = (0, 165, 255)
                else:
                    unknown_count += 1

                box = match["box"]
                cv2.rectangle(
                    frame,
                    (box["left"], box["top"]),
                    (box["right"], box["bottom"]),
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (box["left"], max(box["top"] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Live Attendance - press Q to finish", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

    result = _write_attendance_rows(
        assignment=assignment,
        roster=roster,
        recognized_map=recognized_map,
        manual_present=[],
        attendance_date=resolved_date,
        source="camera",
        effective_teacher_id=int(teacher_id),
    )
    result["unknown_faces"] = unknown_count
    result["liveness"] = liveness_details
    return result
