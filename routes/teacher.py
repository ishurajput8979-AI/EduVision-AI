"""Teacher dashboard and attendance routes."""

from __future__ import annotations

from datetime import date
import sqlite3
from typing import Any

from flask import Blueprint, flash, render_template, request, session

from auth_utils import parse_csv_list
from database import get_teacher_by_id, get_teacher_effective_assignments, get_teacher_live_schedule
from services.analytics import calculate_assignment_attendance
from services.attendance import mark_attendance_from_upload, run_live_camera_attendance
from services.auth import login_required, roles_required

teacher_bp = Blueprint("teacher", __name__, url_prefix="/teacher")


def _load_teacher_context() -> dict[str, Any]:
    teacher_id = int(session["linked_teacher_id"])
    today = date.today().isoformat()
    teacher = get_teacher_by_id(teacher_id)
    assignments = get_teacher_effective_assignments(teacher_id, assignment_date=today)
    summaries = {
        assignment["id"]: calculate_assignment_attendance(assignment["id"])
        for assignment in assignments
    }
    live_schedule = get_teacher_live_schedule(teacher_id)
    assignment_averages = []
    all_student_percentages: list[float] = []
    for assignment in assignments:
        rows = summaries.get(assignment["id"], [])
        percentages = [float(student["percentage"]) for student in rows]
        average_percentage = round(sum(percentages) / len(percentages), 2) if percentages else 0.0
        assignment_averages.append(
            {
                "label": f"{assignment['subject_code']} - {assignment['section']}",
                "value": average_percentage,
            }
        )
        all_student_percentages.extend(percentages)

    metrics = {
        "assignment_count": len(assignments),
        "active_class_count": len(live_schedule["active_slots"]),
        "student_count": sum(len(rows) for rows in summaries.values()),
        "average_attendance": round(sum(all_student_percentages) / len(all_student_percentages), 2)
        if all_student_percentages
        else 0.0,
    }
    return {
        "teacher": teacher,
        "assignments": assignments,
        "summaries": summaries,
        "live_schedule": live_schedule,
        "metrics": metrics,
        "charts": {"assignment_averages": assignment_averages},
    }


@teacher_bp.route("/dashboard", methods=["GET", "POST"])
@login_required
@roles_required("teacher")
def dashboard():
    """Render teacher dashboard and process uploaded attendance."""
    context = _load_teacher_context()
    result = None
    if request.method == "POST":
        assignment_id = int(request.form["assignment_id"])
        attendance_date = request.form.get("attendance_date") or None
        media_upload = request.files.get("media")
        manual_present = parse_csv_list(request.form.get("manual_present", ""))
        try:
            result = mark_attendance_from_upload(
                assignment_id=assignment_id,
                teacher_id=int(session["linked_teacher_id"]),
                attendance_date=attendance_date,
                media_upload=media_upload,
                manual_present=manual_present,
            )
            flash("Attendance processed successfully.", "success")
            context = _load_teacher_context()
        except (PermissionError, RuntimeError, ValueError, sqlite3.Error) as exc:
            flash(f"Unable to mark attendance: {exc}", "danger")
    return render_template("teacher_dashboard.html", data=context, result=result)


@teacher_bp.route("/live-attendance", methods=["POST"])
@login_required
@roles_required("teacher")
def live_attendance():
    """Start live camera attendance for an active class."""
    context = _load_teacher_context()
    result = None
    assignment_id = int(request.form["assignment_id"])
    attendance_date = request.form.get("attendance_date") or date.today().isoformat()

    try:
        result = run_live_camera_attendance(
            assignment_id=assignment_id,
            teacher_id=int(session["linked_teacher_id"]),
            attendance_date=attendance_date,
        )
        flash("Live camera attendance completed successfully.", "success")
        context = _load_teacher_context()
    except (PermissionError, RuntimeError, ValueError, sqlite3.Error) as exc:
        flash(f"Unable to start live camera attendance: {exc}", "danger")

    return render_template("teacher_dashboard.html", data=context, result=result)
