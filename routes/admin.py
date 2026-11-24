"""Admin dashboard and management routes."""

from __future__ import annotations

from datetime import datetime
import mimetypes
from pathlib import Path
import sqlite3

from flask import Blueprint, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from auth_utils import ValidationError
from cache_manager import cache_manager
from database import (
    ACADEMIC_CALENDAR_DIR,
    assign_teacher_subject,
    create_academic_calendar_file,
    create_section,
    create_semester_timeline,
    create_student,
    create_subject,
    create_substitute_assignment,
    create_teacher,
    create_timetable_entry,
    get_dashboard_snapshot,
    get_academic_calendar_file,
    upsert_academic_calendar_day,
    update_timetable_entry,
)
from services.auth import login_required, roles_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")
ALLOWED_CALENDAR_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


def _load_admin_context() -> dict:
    data = get_dashboard_snapshot()
    data["summary"] = {
        "students": len(data["students"]),
        "teachers": len(data["teachers"]),
        "subjects": len(data["subjects"]),
        "sections": len(data["sections"]),
        "academic_calendars": len(data.get("academic_calendars", [])),
        "academic_calendar_days": len(data.get("academic_calendar_days", [])),
    }
    return data


def _invalidate_assignment_dashboard_cache() -> None:
    """Refresh cached teacher assignment analytics after admin data changes."""
    cache_manager.invalidate("assignment_attendance_*")


def _resolve_post_redirect() -> str:
    """Return a safe relative redirect target for admin POST forms."""
    next_url = (request.form.get("next") or "").strip()
    if next_url.startswith("/") and not next_url.startswith("//"):
        return next_url
    return url_for("admin.dashboard")


@admin_bp.route("/dashboard")
@login_required
@roles_required("admin")
def dashboard():
    """Render admin dashboard."""
    return render_template("admin_dashboard.html", data=_load_admin_context())


@admin_bp.route("/students/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_student():
    """Create a new student master record."""
    try:
        create_student(
            {
                "erp_id": request.form["erp_id"].strip(),
                "name": request.form["name"].strip(),
                "roll_no": request.form["roll_no"].strip(),
                "course": request.form["course"].strip(),
                "branch": request.form["branch"].strip(),
                "year": int(request.form["year"]),
                "semester": int(request.form["semester"]),
                "section": request.form["section"].strip().upper(),
                "email": request.form.get("email", "").strip() or None,
                "phone": request.form.get("phone", "").strip() or None,
                "cgpa": request.form.get("cgpa", "").strip() or None,
                "assignments_percentage": request.form.get("assignments_percentage", "").strip() or None,
            }
        )
        _invalidate_assignment_dashboard_cache()
        flash("Student added successfully.", "success")
    except (KeyError, ValidationError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to add student: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/teachers/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_teacher():
    """Create a teacher or HOD and provision their login account."""
    try:
        create_teacher(
            {
                "teacher_code": request.form["teacher_code"].strip(),
                "name": request.form["name"].strip(),
                "department": request.form["department"].strip(),
                "email": request.form.get("email", "").strip() or None,
                "phone": request.form.get("phone", "").strip() or None,
                "role": request.form["role"].strip(),
                "password": request.form["password"],
            }
        )
        flash("Teacher account created successfully.", "success")
    except (KeyError, ValidationError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to add teacher: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/subjects/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_subject():
    """Create a subject for a given course and semester."""
    try:
        create_subject(
            {
                "subject_code": request.form["subject_code"].strip(),
                "subject_name": request.form["subject_name"].strip(),
                "course": request.form["course"].strip(),
                "branch": request.form["branch"].strip(),
                "year": int(request.form["year"]),
                "semester": int(request.form["semester"]),
            }
        )
        flash("Subject created successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to add subject: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/sections/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_section():
    """Create a section record."""
    try:
        create_section(
            request.form["course"].strip(),
            request.form["branch"].strip(),
            int(request.form["year"]),
            int(request.form["semester"]),
            request.form["section"].strip().upper(),
        )
        flash("Section created successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to add section: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/assignments/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_assignment():
    """Assign a teacher to a subject and section."""
    try:
        assign_teacher_subject(
            int(request.form["teacher_id"]),
            int(request.form["subject_id"]),
            request.form["section"].strip().upper(),
        )
        flash("Teacher assigned to subject successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to assign teacher: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/timetable/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_timetable():
    """Create a timetable slot."""
    try:
        create_timetable_entry(request.form.to_dict())
        _invalidate_assignment_dashboard_cache()
        flash("Timetable entry created successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to save timetable: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/timetable/update", methods=["POST"])
@login_required
@roles_required("admin")
def update_timetable():
    """Update an existing timetable row."""
    try:
        timetable_id = int(request.form["timetable_id"])
        update_timetable_entry(
            timetable_id,
            {
                "teacher_id": request.form["teacher_id"],
                "subject_id": request.form["subject_id"],
                "section": request.form["section"].strip().upper(),
                "day_of_week": request.form["day_of_week"],
                "start_time": request.form["start_time"],
                "end_time": request.form["end_time"],
                "room_no": request.form.get("room_no", "").strip() or None,
            },
        )
        _invalidate_assignment_dashboard_cache()
        flash("Timetable updated successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to update timetable: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/substitutes/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_substitute():
    """Assign a one-day substitute teacher."""
    try:
        create_substitute_assignment(
            {
                "assignment_id": int(request.form["assignment_id"]),
                "substitute_teacher_id": int(request.form["substitute_teacher_id"]),
                "assignment_date": request.form["assignment_date"],
                "note": request.form.get("note", "").strip() or None,
            }
        )
        flash("One-day substitute teacher saved successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to save substitute assignment: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/timelines/add", methods=["POST"])
@login_required
@roles_required("admin")
def add_timeline():
    """Create or update the semester attendance policy."""
    try:
        create_semester_timeline(request.form.to_dict())
        _invalidate_assignment_dashboard_cache()
        flash("Semester timeline saved successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to save timeline: {exc}", "danger")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/academic-calendar/upload", methods=["POST"])
@login_required
@roles_required("admin")
def upload_academic_calendar():
    """Upload an academic calendar reference file for a specific academic group."""
    upload = request.files.get("calendar_pdf")
    stored_path: Path | None = None
    try:
        if upload is None or not upload.filename:
            raise ValueError("Academic calendar file is required.")

        safe_filename = secure_filename(upload.filename)
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in ALLOWED_CALENDAR_EXTENSIONS:
            raise ValueError("Only PDF, JPG, JPEG, PNG, or WEBP files are allowed.")

        title = (request.form.get("title") or "").strip() or Path(safe_filename).stem.replace("_", " ")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        stored_filename = f"{timestamp}_{safe_filename}"
        stored_path = ACADEMIC_CALENDAR_DIR / stored_filename
        upload.save(stored_path)

        create_academic_calendar_file(
            {
                "course": request.form["course"],
                "branch": request.form["branch"],
                "year": request.form["year"],
                "semester": request.form["semester"],
                "title": title,
                "original_filename": upload.filename,
                "stored_filename": stored_filename,
            }
        )
        flash("Academic calendar file uploaded successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        if stored_path and stored_path.exists():
            stored_path.unlink(missing_ok=True)
        flash(f"Unable to upload academic calendar: {exc}", "danger")
    return redirect(_resolve_post_redirect())


@admin_bp.route("/academic-calendar/day", methods=["POST"])
@login_required
@roles_required("admin")
def add_academic_calendar_day():
    """Create or update a manual academic calendar day marker."""
    try:
        upsert_academic_calendar_day(request.form.to_dict())
        _invalidate_assignment_dashboard_cache()
        flash("Academic calendar day saved successfully.", "success")
    except (KeyError, ValueError, sqlite3.IntegrityError) as exc:
        flash(f"Unable to save academic calendar day: {exc}", "danger")
    return redirect(_resolve_post_redirect())


@admin_bp.route("/academic-calendar/<int:calendar_file_id>")
@login_required
@roles_required("admin")
def download_academic_calendar(calendar_file_id: int):
    """Open a previously uploaded academic calendar reference file."""
    calendar_file = get_academic_calendar_file(calendar_file_id)
    if not calendar_file:
        flash("Academic calendar file not found.", "danger")
        return redirect(url_for("admin.dashboard"))

    target_path = ACADEMIC_CALENDAR_DIR / calendar_file["stored_filename"]
    if not target_path.exists():
        flash("Academic calendar file is missing from storage.", "danger")
        return redirect(url_for("admin.dashboard"))

    mimetype = mimetypes.guess_type(target_path.name)[0] or "application/octet-stream"
    return send_from_directory(
        str(ACADEMIC_CALENDAR_DIR),
        calendar_file["stored_filename"],
        mimetype=mimetype,
        as_attachment=False,
        download_name=calendar_file["original_filename"],
    )
