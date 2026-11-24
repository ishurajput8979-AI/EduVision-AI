"""Authentication and account routes."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
import mimetypes
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    abort,
    flash,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from auth_utils import ValidationError, hash_password, validate_password_strength, verify_password
from database import (
    ACADEMIC_CALENDAR_DIR,
    consume_password_reset_otp,
    create_password_reset_otp,
    get_academic_calendar_file,
    get_academic_calendar_files,
    get_academic_calendar_days,
    get_all_academic_groups,
    get_attendance_daily_summary,
    get_attendance_rows_for_date,
    get_active_password_reset,
    get_semester_timeline,
    get_scheduled_weekdays,
    get_student_by_erp,
    get_teacher_by_id,
    get_teacher_effective_assignments,
    get_user_by_id,
    get_user_by_username,
    update_user_contact,
    update_user_password,
)
from security_config import limiter
from services.analytics import _department_matches_branch
from services.auth import (
    attach_auth_cookie,
    authenticate_user,
    clear_auth_state,
    generate_otp,
    login_required,
    redirect_to_dashboard,
    start_user_session,
)

auth_bp = Blueprint("auth", __name__)
CALENDAR_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_REGULAR_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)
DEFAULT_WEEKLY_HOLIDAYS = ("Sunday",)

ROLE_PORTALS = [
    {
        "key": "student",
        "label": "Student",
        "tagline": "Track overall attendance, subject analytics, and alerts.",
    },
    {
        "key": "teacher",
        "label": "Teacher",
        "tagline": "Mark attendance, manage live sessions, and review class coverage.",
    },
    {
        "key": "hod",
        "label": "HOD",
        "tagline": "Monitor department-wide attendance risk and send interventions.",
    },
    {
        "key": "admin",
        "label": "Admin",
        "tagline": "Configure academic master data, schedules, and assignments.",
    },
]
ROLE_PORTAL_MAP = {portal["key"]: portal for portal in ROLE_PORTALS}


def _resolve_selected_role(raw_role: str | None) -> str:
    """Normalize a requested public login portal role."""
    role = (raw_role or "").strip().lower()
    return role if role in ROLE_PORTAL_MAP else ""


def _parse_iso_date(raw_value: str | None) -> date | None:
    """Parse a YYYY-MM-DD date value and return None when invalid."""
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_month_start(raw_value: str | None, fallback: date) -> date:
    """Parse a YYYY-MM month value and return the first day of that month."""
    if raw_value:
        try:
            return datetime.strptime(raw_value, "%Y-%m").date().replace(day=1)
        except ValueError:
            pass
    return fallback.replace(day=1)


def _scope_key(scope: dict[str, Any]) -> tuple[str, str, int, int]:
    """Return the unique key for an academic scope row."""
    return (
        str(scope["course"]),
        str(scope["branch"]),
        int(scope["year"]),
        int(scope["semester"]),
    )


def _scope_token(scope: dict[str, Any]) -> str:
    """Encode an academic scope into a compact query-string token."""
    course, branch, year, semester = _scope_key(scope)
    return f"{course}||{branch}||{year}||{semester}"


def _scope_label(scope: dict[str, Any]) -> str:
    """Build a user-facing label for an academic scope."""
    course, branch, year, semester = _scope_key(scope)
    return f"{course} | {branch} | Year {year} | Semester {semester}"


def _unique_scopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and de-duplicate academic scope rows."""
    unique_map: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        if not row:
            continue
        scope = {
            "course": (row.get("course") or "").strip(),
            "branch": (row.get("branch") or "").strip(),
            "year": int(row["year"]),
            "semester": int(row["semester"]),
        }
        if not scope["course"] or not scope["branch"]:
            continue
        unique_map[_scope_key(scope)] = scope
    return sorted(unique_map.values(), key=_scope_key)


def _get_accessible_calendar_scopes() -> list[dict[str, Any]]:
    """Return the academic groups visible to the currently logged-in user."""
    current_role = session.get("role")
    if current_role == "admin":
        return _unique_scopes(get_all_academic_groups())

    if current_role == "student":
        student = get_student_by_erp(session.get("linked_erp", ""))
        if not student:
            return []
        return _unique_scopes([student])

    teacher_id = session.get("linked_teacher_id")
    if not teacher_id:
        return []

    if current_role == "teacher":
        return _unique_scopes(get_teacher_effective_assignments(int(teacher_id)))

    if current_role == "hod":
        teacher = get_teacher_by_id(int(teacher_id))
        if not teacher:
            return []
        return [
            scope
            for scope in _unique_scopes(get_all_academic_groups())
            if _department_matches_branch(teacher.get("department"), scope["branch"])
        ]

    return []


def _resolve_selected_scope(scopes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the requested academic scope when it is available to the user."""
    if not scopes:
        return None
    requested_token = (request.args.get("scope") or "").strip()
    for scope in scopes:
        if _scope_token(scope) == requested_token:
            return scope
    return scopes[0]


def _resolve_calendar_filters() -> dict[str, Any]:
    """Return role-aware attendance filters for shared calendar views."""
    current_role = session.get("role")
    if current_role == "teacher":
        return {"teacher_id": int(session["linked_teacher_id"]), "student_erp": None}
    if current_role == "student":
        return {"teacher_id": None, "student_erp": session["linked_erp"]}
    return {"teacher_id": None, "student_erp": None}


def _timeline_window(timeline: dict[str, Any] | None) -> tuple[date | None, date | None]:
    """Parse the semester timeline start and end dates."""
    if not timeline:
        return None, None
    return _parse_iso_date(timeline.get("start_date")), _parse_iso_date(timeline.get("end_date"))


def _build_day_metadata(
    day_value: date,
    calendar_days_by_date: dict[str, dict[str, Any]],
    timeline: dict[str, Any] | None,
    scheduled_weekdays: set[str],
) -> dict[str, Any]:
    """Describe one academic calendar date for rendering."""
    override = calendar_days_by_date.get(day_value.isoformat())
    timeline_start, timeline_end = _timeline_window(timeline)
    within_timeline = True
    if timeline_start and timeline_end:
        within_timeline = timeline_start <= day_value <= timeline_end

    status = None
    reason = ""
    weekday_label = day_value.strftime("%A")
    if override:
        status = override["day_type"]
        reason = (override.get("reason") or "").strip()
    elif (
        within_timeline
        and weekday_label in DEFAULT_WEEKLY_HOLIDAYS
        and weekday_label not in scheduled_weekdays
    ):
        status = "holiday"
        reason = f"Weekly off ({weekday_label})"
    elif scheduled_weekdays and within_timeline and weekday_label in scheduled_weekdays:
        status = "regular"

    return {
        "status": status,
        "status_label": status.replace("_", " ").title() if status else "No Class Plan",
        "reason": reason,
        "within_timeline": within_timeline,
    }


@auth_bp.route("/academic-calendar")
@login_required
def academic_calendar():
    """Render the shared academic calendar with role-aware attendance visibility."""
    accessible_scopes = _get_accessible_calendar_scopes()
    selected_scope = _resolve_selected_scope(accessible_scopes)
    current_role = session.get("role")
    today = date.today()

    if not selected_scope:
        return render_template(
            "academic_calendar.html",
            current_role=current_role,
            scope_options=[],
            selected_scope=None,
            selected_scope_token="",
            month_label=today.strftime("%B %Y"),
            selected_month=today.strftime("%Y-%m"),
            calendar_weeks=[],
            weekday_labels=list(calendar.day_abbr),
            selected_date=None,
            selected_day_summary=None,
            selected_day_status=None,
            selected_day_rows=[],
            calendar_day_count=0,
            timeline=None,
            scheduled_weekdays=[],
            calendar_files=[],
            preview_file=None,
            return_to=url_for("auth.academic_calendar"),
            prev_month_url=None,
            next_month_url=None,
        )

    filters = _resolve_calendar_filters()
    timeline = get_semester_timeline(
        selected_scope["course"],
        selected_scope["branch"],
        selected_scope["year"],
        selected_scope["semester"],
    )
    scheduled_weekdays = set(
        get_scheduled_weekdays(
            selected_scope["course"],
            selected_scope["branch"],
            selected_scope["year"],
            selected_scope["semester"],
        )
    )
    if timeline and not scheduled_weekdays:
        scheduled_weekdays = set(DEFAULT_REGULAR_WEEKDAYS)

    calendar_days = get_academic_calendar_days(
        selected_scope["course"],
        selected_scope["branch"],
        selected_scope["year"],
        selected_scope["semester"],
    )
    calendar_days_by_date = {row["calendar_date"]: row for row in calendar_days}

    summary_rows = get_attendance_daily_summary(
        selected_scope["course"],
        selected_scope["branch"],
        selected_scope["year"],
        selected_scope["semester"],
        teacher_id=filters["teacher_id"],
        student_erp=filters["student_erp"],
    )
    attendance_summary_by_date = {
        row["attendance_date"]: {
            "marked_count": int(row["marked_count"] or 0),
            "present_count": int(row["present_count"] or 0),
            "absent_count": int(row["absent_count"] or 0),
            "manual_count": int(row["manual_count"] or 0),
        }
        for row in summary_rows
    }

    selected_date = _parse_iso_date(request.args.get("date"))
    default_month = selected_date or today
    timeline_start, timeline_end = _timeline_window(timeline)
    if (
        not selected_date
        and timeline_start
        and timeline_end
        and not (timeline_start <= default_month <= timeline_end)
    ):
        default_month = timeline_start
    month_start = _parse_month_start(request.args.get("month"), default_month)
    month_value = month_start.strftime("%Y-%m")
    scope_token = _scope_token(selected_scope)

    calendar_builder = calendar.Calendar(firstweekday=0)
    calendar_weeks: list[list[dict[str, Any]]] = []
    for week in calendar_builder.monthdatescalendar(month_start.year, month_start.month):
        week_entries: list[dict[str, Any]] = []
        for day_value in week:
            day_metadata = _build_day_metadata(
                day_value,
                calendar_days_by_date,
                timeline,
                scheduled_weekdays,
            )
            day_summary = attendance_summary_by_date.get(day_value.isoformat())
            if day_summary and not day_metadata["status"]:
                day_metadata["status"] = "regular"
                day_metadata["status_label"] = "Attendance Recorded"
            week_entries.append(
                {
                    "day_number": day_value.day,
                    "iso_date": day_value.isoformat(),
                    "in_month": day_value.month == month_start.month,
                    "is_today": day_value == today,
                    "is_selected": selected_date == day_value,
                    "summary": day_summary,
                    "href": url_for(
                        "auth.academic_calendar",
                        scope=scope_token,
                        month=day_value.replace(day=1).strftime("%Y-%m"),
                        date=day_value.isoformat(),
                    ),
                    **day_metadata,
                }
            )
        calendar_weeks.append(week_entries)

    selected_day_rows: list[dict[str, Any]] = []
    selected_day_summary = None
    selected_day_status = None
    if selected_date:
        selected_day_summary = attendance_summary_by_date.get(selected_date.isoformat())
        selected_day_status = _build_day_metadata(
            selected_date,
            calendar_days_by_date,
            timeline,
            scheduled_weekdays,
        )
        if selected_day_summary and not selected_day_status["status"]:
            selected_day_status["status"] = "regular"
            selected_day_status["status_label"] = "Attendance Recorded"
        selected_day_rows = get_attendance_rows_for_date(
            selected_scope["course"],
            selected_scope["branch"],
            selected_scope["year"],
            selected_scope["semester"],
            selected_date.isoformat(),
            teacher_id=filters["teacher_id"],
            student_erp=filters["student_erp"],
        )

    calendar_files = get_academic_calendar_files(
        selected_scope["course"],
        selected_scope["branch"],
        selected_scope["year"],
        selected_scope["semester"],
    )
    preview_file = None
    for calendar_file in calendar_files:
        suffix = Path(calendar_file["stored_filename"]).suffix.lower()
        calendar_file["is_image"] = suffix in CALENDAR_IMAGE_EXTENSIONS
        calendar_file["view_url"] = url_for(
            "auth.view_academic_calendar_file",
            calendar_file_id=calendar_file["id"],
        )
        if preview_file is None and calendar_file["is_image"]:
            preview_file = calendar_file

    prev_month = (month_start - timedelta(days=1)).replace(day=1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    scope_options = [
        {
            **scope,
            "label": _scope_label(scope),
            "token": _scope_token(scope),
            "selected": _scope_key(scope) == _scope_key(selected_scope),
            "href": url_for("auth.academic_calendar", scope=_scope_token(scope)),
        }
        for scope in accessible_scopes
    ]

    return render_template(
        "academic_calendar.html",
        current_role=current_role,
        scope_options=scope_options,
        selected_scope=selected_scope,
        selected_scope_token=scope_token,
        selected_scope_label=_scope_label(selected_scope),
        month_label=month_start.strftime("%B %Y"),
        selected_month=month_value,
        calendar_weeks=calendar_weeks,
        weekday_labels=list(calendar.day_abbr),
        selected_date=selected_date.isoformat() if selected_date else "",
        selected_day_summary=selected_day_summary,
        selected_day_status=selected_day_status,
        selected_day_rows=selected_day_rows,
        calendar_day_count=len(calendar_days),
        timeline=timeline,
        scheduled_weekdays=sorted(
            scheduled_weekdays,
            key=lambda label: DEFAULT_REGULAR_WEEKDAYS.index(label)
            if label in DEFAULT_REGULAR_WEEKDAYS
            else 7,
        ),
        calendar_files=calendar_files,
        preview_file=preview_file,
        return_to=request.full_path[:-1] if request.full_path.endswith("?") else request.full_path,
        prev_month_url=url_for(
            "auth.academic_calendar",
            scope=scope_token,
            month=prev_month.strftime("%Y-%m"),
        ),
        next_month_url=url_for(
            "auth.academic_calendar",
            scope=scope_token,
            month=next_month.strftime("%Y-%m"),
        ),
    )


@auth_bp.route("/academic-calendar/file/<int:calendar_file_id>")
@login_required
def view_academic_calendar_file(calendar_file_id: int):
    """Serve an uploaded academic calendar file to authorized users."""
    calendar_file = get_academic_calendar_file(calendar_file_id)
    if not calendar_file:
        abort(404)

    allowed_scope_keys = {_scope_key(scope) for scope in _get_accessible_calendar_scopes()}
    file_scope = {
        "course": calendar_file["course"],
        "branch": calendar_file["branch"],
        "year": int(calendar_file["year"]),
        "semester": int(calendar_file["semester"]),
    }
    if _scope_key(file_scope) not in allowed_scope_keys:
        abort(403)

    target_path = ACADEMIC_CALENDAR_DIR / calendar_file["stored_filename"]
    if not target_path.exists():
        flash("Academic calendar file is missing from storage.", "danger")
        return redirect(url_for("auth.academic_calendar", scope=_scope_token(file_scope)))

    mimetype = mimetypes.guess_type(target_path.name)[0] or "application/octet-stream"
    return send_from_directory(
        str(ACADEMIC_CALENDAR_DIR),
        calendar_file["stored_filename"],
        mimetype=mimetype,
        as_attachment=False,
        download_name=calendar_file["original_filename"],
    )


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    """Render login page and create authenticated session."""
    selected_role = _resolve_selected_role(request.args.get("role"))
    if request.method == "POST":
        selected_role = _resolve_selected_role(request.form.get("expected_role"))
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template(
                "login.html",
                username=username,
                selected_role=selected_role,
                role_portals=ROLE_PORTALS,
            )

        user = authenticate_user(username, password)
        if not user:
            flash("Invalid username or password.", "danger")
            return render_template(
                "login.html",
                username=username,
                selected_role=selected_role,
                role_portals=ROLE_PORTALS,
            )
        if selected_role and user["role"] != selected_role:
            flash(
                f"This login is for the {ROLE_PORTAL_MAP[selected_role]['label']} portal. "
                "Please use the correct account for this role.",
                "danger",
            )
            return render_template(
                "login.html",
                username=username,
                selected_role=selected_role,
                role_portals=ROLE_PORTALS,
            )

        token = start_user_session(user)
        response = make_response(redirect(redirect_to_dashboard(user["role"])))
        attach_auth_cookie(response, token)
        flash(f"Welcome back, {user['username']}.", "success")
        return response

    return render_template(
        "login.html",
        username="",
        selected_role=selected_role,
        role_portals=ROLE_PORTALS,
    )


@auth_bp.route("/logout")
def logout():
    """Clear session and token cookie."""
    response = make_response(redirect(url_for("auth.login")))
    clear_auth_state(response)
    flash("You have been logged out.", "info")
    return response


@auth_bp.route("/account/settings", methods=["GET", "POST"])
@login_required
def account_settings():
    """Allow a logged-in user to update contact details and password."""
    user = get_user_by_id(int(session["user_id"]))
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "update_contact":
                email = request.form.get("email", "").strip() or None
                phone = request.form.get("phone", "").strip() or None
                update_user_contact(int(session["user_id"]), email, phone)
                flash("Contact details updated successfully.", "success")
                user = get_user_by_id(int(session["user_id"]))
            elif action == "change_password":
                current_password = request.form.get("current_password", "")
                new_password = request.form.get("new_password", "")
                confirm_password = request.form.get("confirm_password", "")
                if not verify_password(current_password, user["password"]):
                    flash("Current password is incorrect.", "danger")
                elif new_password != confirm_password:
                    flash("New password and confirm password do not match.", "danger")
                else:
                    validate_password_strength(new_password)
                    update_user_password(int(session["user_id"]), hash_password(new_password))
                    flash("Password changed successfully.", "success")
                    user = get_user_by_id(int(session["user_id"]))
        except ValidationError as exc:
            flash(str(exc), "danger")

    return render_template("account_settings.html", user=user)


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def forgot_password():
    """Handle OTP-based password reset using email or phone."""
    if request.method == "POST":
        step = request.form.get("step", "request")
        username = request.form.get("username", "").strip()
        user = get_user_by_username(username) if username else None

        if not user:
            flash("Username not found.", "danger")
            return render_template("forgot_password.html", step="request")

        if step == "request":
            contact_method = request.form.get("contact_method", "").strip()
            destination = (user.get(contact_method) or "").strip() if contact_method else ""
            if contact_method not in {"email", "phone"} or not destination:
                flash("Selected contact method is not configured for this account.", "danger")
                return render_template(
                    "forgot_password.html",
                    step="request",
                    username=username,
                    user=user,
                )

            otp = generate_otp()
            expires_at = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            create_password_reset_otp(
                user_id=int(user["id"]),
                otp_hash=hash_password(otp),
                contact_method=contact_method,
                destination=destination,
                expires_at=expires_at,
            )
            flash(
                f"Demo OTP sent to your {contact_method}: {otp}. Integrate real SMS/email service for production.",
                "warning",
            )
            return render_template(
                "forgot_password.html",
                step="verify",
                username=username,
                contact_method=contact_method,
            )

        if step == "verify":
            otp = request.form.get("otp", "").strip()
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")
            active_reset = get_active_password_reset(int(user["id"]))
            if not active_reset:
                flash("No active OTP found. Request a new OTP.", "danger")
                return render_template("forgot_password.html", step="request")
            if datetime.strptime(active_reset["expires_at"], "%Y-%m-%d %H:%M:%S") < datetime.now():
                consume_password_reset_otp(int(active_reset["id"]))
                flash("OTP expired. Request a new one.", "danger")
                return render_template("forgot_password.html", step="request")
            if not verify_password(otp, active_reset["otp_hash"]):
                flash("Invalid OTP.", "danger")
                return render_template(
                    "forgot_password.html",
                    step="verify",
                    username=username,
                    contact_method=active_reset["contact_method"],
                )
            if new_password != confirm_password:
                flash("New password and confirm password do not match.", "danger")
                return render_template(
                    "forgot_password.html",
                    step="verify",
                    username=username,
                    contact_method=active_reset["contact_method"],
                )
            try:
                validate_password_strength(new_password)
            except ValidationError as exc:
                flash(str(exc), "danger")
                return render_template(
                    "forgot_password.html",
                    step="verify",
                    username=username,
                    contact_method=active_reset["contact_method"],
                )

            update_user_password(int(user["id"]), hash_password(new_password))
            consume_password_reset_otp(int(active_reset["id"]))
            flash("Password reset successful. Please log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("forgot_password.html", step="request")
