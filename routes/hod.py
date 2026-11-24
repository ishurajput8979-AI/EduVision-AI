"""HOD dashboard routes."""

from __future__ import annotations

from datetime import date, datetime

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from alert_service import alert_service
from database import create_alert, get_student_by_erp, get_teacher_by_id
from services.auth import login_required, roles_required
from services.analytics import (
    _department_matches_branch,
    get_department_overview,
    get_department_overall_attendance,
    get_student_attendance_overview,
)

hod_bp = Blueprint("hod", __name__, url_prefix="/hod")


def _default_alert_form() -> dict[str, object]:
    """Return the default HOD SMS preference state."""
    return {
        "message_mode": "default",
        "custom_message": "",
    }


def _read_alert_form() -> dict[str, object]:
    """Read HOD SMS preferences from a submitted form."""
    message_mode = "custom" if request.form.get("message_mode") == "custom" else "default"
    custom_message = (request.form.get("custom_message") or "").strip()
    if message_mode == "custom" and not custom_message:
        flash("Custom SMS blank tha, isliye default message use kiya jayega.", "warning")
        message_mode = "default"
    return {
        "message_mode": message_mode,
        "custom_message": custom_message,
    }


def _flash_sms_summary(summary: dict[str, int | str]) -> None:
    """Flash manual HOD SMS outcome."""
    if summary["status"] == "unconfigured":
        flash("Twilio SMS abhi configured nahi hai.", "warning")
    elif summary["status"] == "missing_phone":
        flash("Student ka phone number missing hai.", "warning")
    elif summary["status"] == "not_low":
        flash("Manual SMS sirf low-attendance student ke liye bheja ja sakta hai.", "warning")
    elif summary["status"] == "sent":
        flash("Manual low-attendance SMS sent successfully.", "success")
    else:
        flash("Manual SMS send nahi ho saka.", "warning")


def _build_message_context(
    teacher_name: str,
    student_erp: str,
    attendance_data: dict[str, object],
) -> dict[str, object]:
    """Build a consistent overall-attendance SMS context for HOD alerts."""
    return {
        "teacher_name": teacher_name,
        "subject_name": "overall attendance",
        "subject_code": "OVERALL",
        "attendance_date": date.today().isoformat(),
        "student_erp": student_erp,
        "total_classes": int(
            attendance_data.get("total_scheduled_classes")
            or attendance_data.get("total_classes")
            or 0
        ),
        "present_classes": int(
            attendance_data.get("total_present_classes")
            or attendance_data.get("present_classes")
            or 0
        ),
    }


def _flash_bulk_sms_summary(summary: dict[str, object]) -> None:
    """Flash aggregate HOD bulk-SMS results."""
    if summary["status"] == "unconfigured":
        flash("Twilio SMS abhi configured nahi hai.", "warning")
        return
    if summary["status"] == "no_low_attendance":
        flash("Department me koi low-attendance student abhi available nahi mila.", "warning")
        return

    if int(summary["sent_count"] or 0):
        flash(
            f"Manual low-attendance SMS sent to {summary['sent_count']} student(s).",
            "success",
        )
    elif int(summary["eligible_count"] or 0):
        flash("Low-attendance SMS bhejne ki koshish hui, lekin send nahi ho saka.", "warning")

    if int(summary["missing_phone_count"] or 0):
        flash(
            f"{summary['missing_phone_count']} low-attendance student(s) ke phone number missing the.",
            "warning",
        )

    if int(summary["failed_count"] or 0):
        template_error = summary.get("template_error")
        if template_error:
            flash(f"Some SMS could not be sent: {template_error}", "warning")
        elif not int(summary["sent_count"] or 0):
            flash("Manual SMS send nahi ho saka.", "warning")


@hod_bp.route("/dashboard")
@login_required
@roles_required("hod")
def dashboard():
    """Render HOD dashboard with department risk analytics."""
    alert_form = _default_alert_form()
    teacher = get_teacher_by_id(int(session["linked_teacher_id"]))
    if not teacher:
        flash("HOD account mapping missing. Dashboard load nahi ho saka.", "danger")
        return redirect(url_for("auth.login"))
    overview = get_department_overview(teacher["department"])
    overall_attendance = get_department_overall_attendance(teacher["department"])
    return render_template(
        "hod_dashboard.html",
        teacher=teacher,
        overview=overview,
        sms_status=alert_service.get_sms_status(),
        alert_form=alert_form,
        overall_attendance=overall_attendance,
    )


@hod_bp.route("/send-alert", methods=["POST"])
@login_required
@roles_required("hod")
def send_alert():
    """Allow the HOD to send a manual low-attendance SMS to any student in the department."""
    alert_form = _read_alert_form()
    teacher = get_teacher_by_id(int(session["linked_teacher_id"]))
    if not teacher:
        flash("HOD account mapping missing. Alert send nahi ho saka.", "danger")
        return redirect(url_for("auth.login"))
    student_erp = (request.form.get("student_erp") or "").strip().upper()
    student = get_student_by_erp(student_erp)
    if not student or not _department_matches_branch(teacher["department"], student["branch"]):
        flash("Student department mismatch. Alert send nahi ho saka.", "danger")
        return redirect(url_for("hod.dashboard"))

    if not alert_service.get_sms_status()["configured"]:
        _flash_sms_summary({"status": "unconfigured"})
        return redirect(url_for("hod.dashboard"))
    if not student.get("phone"):
        _flash_sms_summary({"status": "missing_phone"})
        return redirect(url_for("hod.dashboard"))

    attendance_overview = get_student_attendance_overview(student_erp)
    percentage = float(attendance_overview["overall_attendance"] or 0.0)
    minimum = float(attendance_overview["min_attendance"] or 75.0)
    if percentage >= minimum:
        _flash_sms_summary({"status": "not_low"})
        return redirect(url_for("hod.dashboard"))

    custom_template = (
        alert_form["custom_message"]
        if alert_form["message_mode"] == "custom"
        else None
    )
    try:
        sent = alert_service.send_low_attendance_alert(
            phone_number=student["phone"],
            student_name=student["name"],
            attendance_percentage=percentage,
            threshold=minimum,
            required_classes=int(attendance_overview.get("required_classes") or 0),
            custom_message_template=custom_template,
            message_context=_build_message_context(
                teacher_name=teacher["name"],
                student_erp=student_erp,
                attendance_data=attendance_overview,
            ),
        )
    except ValueError as exc:
        flash(f"Custom SMS template invalid hai: {exc}", "warning")
        return redirect(url_for("hod.dashboard"))
    if sent:
        create_alert(
            student_erp,
            "manual_sms",
            (
                f"Manual SMS sent by HOD {teacher['name']} at "
                f"{datetime.now().strftime('%H:%M:%S')}. Overall attendance: {percentage:.2f}%."
            ),
        )
        _flash_sms_summary({"status": "sent"})
    else:
        _flash_sms_summary({"status": "failed"})
    return redirect(url_for("hod.dashboard"))


@hod_bp.route("/manual-alert-all", methods=["POST"])
@login_required
@roles_required("hod")
def manual_alert_all():
    """Send SMS alerts to all low-attendance students in the HOD's department."""
    alert_form = _read_alert_form()
    teacher = get_teacher_by_id(int(session["linked_teacher_id"]))
    if not teacher:
        flash("HOD account mapping missing. Bulk alert send nahi ho saka.", "danger")
        return redirect(url_for("auth.login"))
    students = get_department_overall_attendance(teacher["department"])

    if not alert_service.get_sms_status()["configured"]:
        _flash_bulk_sms_summary({"status": "unconfigured"})
        return redirect(url_for("hod.dashboard"))

    custom_template = (
        alert_form["custom_message"]
        if alert_form["message_mode"] == "custom"
        else None
    )
    timestamp = datetime.now().strftime("%H:%M:%S")
    summary = {
        "status": "processed",
        "low_count": 0,
        "eligible_count": 0,
        "sent_count": 0,
        "failed_count": 0,
        "missing_phone_count": 0,
        "template_error": None,
    }

    for student in students:
        if not student["is_below_threshold"]:
            continue

        summary["low_count"] += 1
        if not student.get("phone"):
            summary["missing_phone_count"] += 1
            continue

        summary["eligible_count"] += 1
        try:
            sent = alert_service.send_low_attendance_alert(
                phone_number=student["phone"],
                student_name=student["name"],
                attendance_percentage=float(student["percentage"] or 0.0),
                threshold=float(student["min_attendance"] or 75.0),
                required_classes=int(student.get("required_classes") or 0),
                custom_message_template=custom_template,
                message_context=_build_message_context(
                    teacher_name=teacher["name"],
                    student_erp=student["erp_id"],
                    attendance_data=student,
                ),
            )
        except ValueError as exc:
            summary["failed_count"] += 1
            summary["template_error"] = summary["template_error"] or str(exc)
            continue

        if sent:
            summary["sent_count"] += 1
            create_alert(
                student["erp_id"],
                "manual_sms",
                (
                    f"Manual SMS sent by HOD {teacher['name']} at {timestamp}. "
                    f"Overall attendance: {float(student['percentage'] or 0.0):.2f}%."
                ),
            )
        else:
            summary["failed_count"] += 1

    if not summary["low_count"]:
        summary["status"] = "no_low_attendance"
    _flash_bulk_sms_summary(summary)
    return redirect(url_for("hod.dashboard"))
