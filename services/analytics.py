"""Attendance analytics and dashboard context builders."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re
from statistics import mean
from typing import Any

from cache_manager import cache_manager
from database import (
    ACADEMIC_EXCLUDED_DAY_TYPES,
    create_alert,
    get_academic_calendar_day_status,
    get_academic_calendar_days,
    get_assignment_by_id,
    get_recent_alerts,
    get_student_by_erp,
    predict_student_risk,
    query_all,
    query_one,
)

WEEKDAY_TO_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def _label_tokens(raw_value: str | None) -> set[str]:
    """Split a department or branch label into uppercase alphanumeric tokens."""
    return {token for token in re.findall(r"[A-Z0-9]+", (raw_value or "").upper()) if token}


def _department_matches_branch(department: str | None, branch: str | None) -> bool:
    """Return True when a HOD department label should cover a student/subject branch label."""
    normalized_department = (department or "").strip().upper()
    normalized_branch = (branch or "").strip().upper()
    if not normalized_department or not normalized_branch:
        return False
    if normalized_department == normalized_branch:
        return True

    department_tokens = _label_tokens(normalized_department)
    branch_tokens = _label_tokens(normalized_branch)
    if department_tokens and branch_tokens and department_tokens.issubset(branch_tokens):
        return True

    compact_department = "".join(sorted(department_tokens))
    compact_branch = "".join(sorted(branch_tokens))
    return bool(compact_department and compact_department == compact_branch)


def _parse_iso_date(raw_value: str | None) -> date | None:
    """Safely parse a YYYY-MM-DD date string."""
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_date_within_window(
    target_date: date | None,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Return whether a date falls inside the active attendance window."""
    if target_date is None:
        return False
    if start_date and target_date < start_date:
        return False
    if end_date and target_date > end_date:
        return False
    return True


def _get_excluded_calendar_dates(
    course: str,
    branch: str,
    year: int,
    semester: int,
) -> dict[date, dict[str, Any]]:
    """Return holiday and half-day dates that should be excluded from attendance logic."""
    excluded_dates: dict[date, dict[str, Any]] = {}
    for row in get_academic_calendar_days(course, branch, year, semester):
        if row.get("day_type") not in ACADEMIC_EXCLUDED_DAY_TYPES:
            continue
        calendar_date = _parse_iso_date(row.get("calendar_date"))
        if calendar_date is not None:
            excluded_dates[calendar_date] = row
    return excluded_dates


def _filter_recorded_attendance_rows(
    rows: list[dict[str, Any]],
    start_date: date | None,
    end_date: date | None,
    excluded_dates: set[date],
) -> list[dict[str, Any]]:
    """Drop attendance rows that fall outside the timeline or on excluded academic dates."""
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        attendance_date = _parse_iso_date(row.get("attendance_date"))
        if not _is_date_within_window(attendance_date, start_date, end_date):
            continue
        if attendance_date in excluded_dates:
            continue
        filtered_rows.append(row)
    return filtered_rows


def _count_weekday_occurrences(start_date: date, end_date: date, weekday_name: str) -> int:
    """Count how many times a weekday appears within an inclusive date window."""
    if end_date < start_date:
        return 0
    weekday_index = WEEKDAY_TO_INDEX.get(weekday_name)
    if weekday_index is None:
        return 0
    days_until_first = (weekday_index - start_date.weekday()) % 7
    first_match = start_date + timedelta(days=days_until_first)
    if first_match > end_date:
        return 0
    return 1 + ((end_date - first_match).days // 7)


def _get_timeline_config(course: str, branch: str, year: int, semester: int) -> dict[str, Any]:
    """Return semester timeline boundaries and attendance threshold for an academic group."""
    timeline = query_one(
        """
        SELECT *
        FROM semester_timelines
        WHERE course = ? AND branch = ? AND year = ? AND semester = ?
        """,
        (course, branch, int(year), int(semester)),
    )
    today = date.today()
    start_date = _parse_iso_date((timeline or {}).get("start_date"))
    end_date = _parse_iso_date((timeline or {}).get("end_date"))
    effective_end = min(end_date or today, today)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "effective_end": effective_end,
        "min_attendance": float((timeline or {}).get("min_attendance") or 75.0),
        "timeline_available": bool(start_date),
    }


def _get_section_schedule_rows(
    course: str,
    branch: str,
    year: int,
    semester: int,
    section: str,
    subject_id: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch timetable slots for one academic section, optionally for one subject."""
    params: list[Any] = [course, branch, int(year), int(semester), section.upper()]
    subject_filter = ""
    if subject_id is not None:
        subject_filter = "AND tt.subject_id = ?"
        params.append(int(subject_id))
    return query_all(
        f"""
        SELECT
            tt.subject_id,
            tt.day_of_week,
            tt.start_time,
            tt.end_time,
            s.subject_name,
            s.subject_code
        FROM timetable tt
        JOIN subjects s ON s.id = tt.subject_id
        WHERE s.course = ? AND s.branch = ? AND s.year = ? AND s.semester = ?
          AND tt.section = ?
          {subject_filter}
        ORDER BY tt.day_of_week, tt.start_time
        """,
        params,
    )


def _build_scheduled_class_counts(
    schedule_rows: list[dict[str, Any]],
    start_date: date | None,
    end_date: date | None,
    excluded_dates: set[date] | None = None,
) -> dict[int, dict[str, Any]]:
    """Aggregate scheduled lecture counts per subject within the active timeline window."""
    counts: dict[int, dict[str, Any]] = {}
    if not start_date or not end_date or end_date < start_date:
        return counts
    excluded_dates = excluded_dates or set()

    for row in schedule_rows:
        subject_id = int(row["subject_id"])
        subject_counts = counts.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "subject_name": row["subject_name"],
                "subject_code": row["subject_code"],
                "scheduled_classes": 0,
            },
        )
        scheduled_classes = _count_weekday_occurrences(
            start_date,
            end_date,
            row["day_of_week"],
        )
        weekday_index = WEEKDAY_TO_INDEX.get(row["day_of_week"])
        if weekday_index is not None and excluded_dates:
            scheduled_classes -= sum(
                1
                for excluded_date in excluded_dates
                if start_date <= excluded_date <= end_date and excluded_date.weekday() == weekday_index
            )
        subject_counts["scheduled_classes"] += max(scheduled_classes, 0)
    return counts


def _finalize_attendance_rows(
    rows: list[dict[str, Any]],
    default_threshold: float,
) -> list[dict[str, Any]]:
    """Fill derived lecture metrics after scheduled and recorded counts are known."""
    finalized_rows: list[dict[str, Any]] = []
    for row in rows:
        present = int(row.get("present_classes") or 0)
        recorded = int(row.get("recorded_classes") or 0)
        scheduled = int(row.get("scheduled_classes") or 0)
        total_classes = max(scheduled, recorded)
        if total_classes <= 0:
            total_classes = recorded

        row["min_attendance"] = float(row.get("min_attendance") or default_threshold or 75.0)
        row["total_classes"] = total_classes
        row["recorded_classes"] = recorded
        row["present_classes"] = present
        row["unmarked_classes"] = max(total_classes - recorded, 0)
        row["missed_classes"] = max(total_classes - present, 0)
        row["explicit_absent_classes"] = max(recorded - present, 0)
        row["percentage"] = round((present / total_classes) * 100, 2) if total_classes else 0.0
        row["required_classes"] = _classes_needed_to_reach_threshold(
            total_classes=total_classes,
            present_classes=present,
            threshold=float(row["min_attendance"]),
        )
        finalized_rows.append(row)
    return sorted(finalized_rows, key=lambda item: (item["subject_name"], item["subject_code"]))


def get_student_attendance_overview(student_erp: str) -> dict[str, Any]:
    """Return subject-wise and overall attendance using semester timeline and timetable totals."""
    student = get_student_by_erp(student_erp)
    if not student:
        return {
            "subject_percentages": [],
            "overall_attendance": 0.0,
            "total_scheduled_classes": 0,
            "total_present_classes": 0,
            "total_recorded_classes": 0,
            "total_unmarked_classes": 0,
            "total_missed_classes": 0,
            "min_attendance": 75.0,
            "required_classes": 0,
            "weakest_subject": None,
        }

    timeline = _get_timeline_config(
        student["course"],
        student["branch"],
        int(student["year"]),
        int(student["semester"]),
    )
    excluded_calendar_dates = set(
        _get_excluded_calendar_dates(
            student["course"],
            student["branch"],
            int(student["year"]),
            int(student["semester"]),
        ).keys()
    )
    schedule_counts = _build_scheduled_class_counts(
        _get_section_schedule_rows(
            student["course"],
            student["branch"],
            int(student["year"]),
            int(student["semester"]),
            student["section"],
        ),
        start_date=timeline["start_date"],
        end_date=timeline["effective_end"],
        excluded_dates=excluded_calendar_dates,
    )
    recorded_rows = query_all(
        """
        SELECT
            a.subject_id,
            s.subject_name,
            s.subject_code,
            a.attendance_date,
            a.status
        FROM attendance a
        JOIN subjects s ON s.id = a.subject_id
        WHERE a.student_erp = ?
          AND s.course = ?
          AND s.branch = ?
          AND s.year = ?
          AND s.semester = ?
        ORDER BY s.subject_name, a.attendance_date
        """,
        (
            student_erp,
            student["course"],
            student["branch"],
            int(student["year"]),
            int(student["semester"]),
        ),
    )
    recorded_rows = _filter_recorded_attendance_rows(
        recorded_rows,
        start_date=timeline["start_date"],
        end_date=timeline["effective_end"],
        excluded_dates=excluded_calendar_dates,
    )

    subject_map: dict[int, dict[str, Any]] = {
        int(subject_id): {
            "subject_id": int(subject_id),
            "subject_name": item["subject_name"],
            "subject_code": item["subject_code"],
            "scheduled_classes": int(item.get("scheduled_classes") or 0),
            "recorded_classes": 0,
            "present_classes": 0,
            "min_attendance": timeline["min_attendance"],
        }
        for subject_id, item in schedule_counts.items()
    }
    for row in recorded_rows:
        subject_id = int(row["subject_id"])
        subject_entry = subject_map.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "subject_name": row["subject_name"],
                "subject_code": row["subject_code"],
                "scheduled_classes": 0,
                "recorded_classes": 0,
                "present_classes": 0,
                "min_attendance": timeline["min_attendance"],
            },
        )
        subject_entry["recorded_classes"] += 1
        if row.get("status") == "Present":
            subject_entry["present_classes"] += 1

    subject_percentages = _finalize_attendance_rows(
        list(subject_map.values()),
        default_threshold=timeline["min_attendance"],
    )
    total_scheduled_classes = sum(int(item["total_classes"]) for item in subject_percentages)
    total_present_classes = sum(int(item["present_classes"]) for item in subject_percentages)
    total_recorded_classes = sum(int(item["recorded_classes"]) for item in subject_percentages)
    total_unmarked_classes = sum(int(item["unmarked_classes"]) for item in subject_percentages)
    total_missed_classes = sum(int(item["missed_classes"]) for item in subject_percentages)
    overall_attendance = (
        round((total_present_classes / total_scheduled_classes) * 100, 2)
        if total_scheduled_classes
        else 0.0
    )
    required_classes = _classes_needed_to_reach_threshold(
        total_classes=total_scheduled_classes,
        present_classes=total_present_classes,
        threshold=float(timeline["min_attendance"]),
    )
    weakest_subject = (
        min(
            subject_percentages,
            key=lambda item: (float(item["percentage"]), item["subject_name"]),
        )
        if subject_percentages
        else None
    )
    return {
        "subject_percentages": subject_percentages,
        "overall_attendance": overall_attendance,
        "total_scheduled_classes": total_scheduled_classes,
        "total_present_classes": total_present_classes,
        "total_recorded_classes": total_recorded_classes,
        "total_unmarked_classes": total_unmarked_classes,
        "total_missed_classes": total_missed_classes,
        "min_attendance": float(timeline["min_attendance"]),
        "required_classes": required_classes,
        "weakest_subject": weakest_subject,
    }


def calculate_student_percentages(student_erp: str) -> list[dict[str, Any]]:
    """Return subject-wise attendance percentages for a student."""
    return get_student_attendance_overview(student_erp)["subject_percentages"]


def calculate_assignment_attendance(assignment_id: int) -> list[dict[str, Any]]:
    """Calculate subject attendance for the roster mapped to an assignment."""
    cache_key = f"assignment_attendance_{int(assignment_id)}"
    cached_rows = cache_manager.get(cache_key)
    if cached_rows is not None:
        return cached_rows

    assignment = get_assignment_by_id(assignment_id)
    if not assignment:
        return []
    timeline = _get_timeline_config(
        assignment["course"],
        assignment["branch"],
        int(assignment["year"]),
        int(assignment["semester"]),
    )
    excluded_calendar_dates = set(
        _get_excluded_calendar_dates(
            assignment["course"],
            assignment["branch"],
            int(assignment["year"]),
            int(assignment["semester"]),
        ).keys()
    )
    scheduled_counts = _build_scheduled_class_counts(
        _get_section_schedule_rows(
            assignment["course"],
            assignment["branch"],
            int(assignment["year"]),
            int(assignment["semester"]),
            assignment["section"],
            subject_id=int(assignment["subject_id"]),
        ),
        start_date=timeline["start_date"],
        end_date=timeline["effective_end"],
        excluded_dates=excluded_calendar_dates,
    )
    scheduled_total = int(
        (scheduled_counts.get(int(assignment["subject_id"])) or {}).get("scheduled_classes") or 0
    )

    raw_rows = query_all(
        """
        SELECT
            st.erp_id AS student_erp,
            st.name,
            st.roll_no,
            st.email,
            st.phone,
            COALESCE(sl.min_attendance, 75.0) AS min_attendance,
            a.attendance_date,
            a.status
        FROM teacher_subjects ts
        JOIN subjects s ON s.id = ts.subject_id
        JOIN students st
            ON st.course = s.course
            AND st.branch = s.branch
            AND st.year = s.year
            AND st.semester = s.semester
            AND st.section = ts.section
        LEFT JOIN semester_timelines sl
            ON sl.course = st.course
            AND sl.branch = st.branch
            AND sl.year = st.year
            AND sl.semester = st.semester
        LEFT JOIN attendance a
            ON a.student_erp = st.erp_id
            AND a.subject_id = ts.subject_id
        WHERE ts.id = ?
        ORDER BY st.roll_no, a.attendance_date
        """,
        (assignment_id,),
    )
    rows_by_student: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        student_erp = row["student_erp"]
        student_row = rows_by_student.setdefault(
            student_erp,
            {
                "student_erp": student_erp,
                "name": row["name"],
                "roll_no": row["roll_no"],
                "email": row["email"],
                "phone": row["phone"],
                "min_attendance": float(row.get("min_attendance") or timeline["min_attendance"]),
                "recorded_classes": 0,
                "present_classes": 0,
            },
        )
        attendance_date = _parse_iso_date(row.get("attendance_date"))
        if not _is_date_within_window(
            attendance_date,
            timeline["start_date"],
            timeline["effective_end"],
        ):
            continue
        if attendance_date in excluded_calendar_dates:
            continue
        student_row["recorded_classes"] += 1
        if row.get("status") == "Present":
            student_row["present_classes"] += 1

    rows = list(rows_by_student.values())
    for row in rows:
        row["subject_name"] = assignment["subject_name"]
        row["subject_code"] = assignment["subject_code"]
        row["scheduled_classes"] = scheduled_total
    finalized_rows = _finalize_attendance_rows(rows, default_threshold=timeline["min_attendance"])
    cache_manager.set(cache_key, finalized_rows, ttl_seconds=300)
    return finalized_rows


def _classes_needed_to_reach_threshold(
    total_classes: int,
    present_classes: int,
    threshold: float,
) -> int:
    """Return the minimum extra attended classes needed to recover attendance."""
    if total_classes <= 0:
        return 0
    threshold_ratio = max(0.0, min(float(threshold) / 100.0, 0.99))
    if total_classes and (present_classes / total_classes) >= threshold_ratio:
        return 0
    required = (threshold_ratio * total_classes - present_classes) / (1 - threshold_ratio)
    return max(1, math.ceil(required))


def _build_low_attendance_message(row: dict[str, Any]) -> str:
    """Create the stored dashboard alert copy for a low-attendance student."""
    return (
        f"Low attendance in {row['subject_name']} ({row['subject_code']}): "
        f"{row['percentage']}% against required {row['min_attendance']}%. "
        f"Need at least {row['required_classes']} more classes to recover."
    )


def create_low_attendance_alerts(assignment_id: int | None = None) -> list[dict[str, Any]]:
    """Create unique low-attendance alerts and return the newly-created rows."""
    created_rows: list[dict[str, Any]] = []
    if assignment_id is not None:
        assignment = get_assignment_by_id(assignment_id)
        if not assignment:
            return created_rows

        for row in calculate_assignment_attendance(assignment_id):
            total_classes = int(row["total_classes"] or 0)
            present_classes = int(row["present_classes"] or 0)
            threshold = float(row.get("min_attendance") or 75.0)
            percentage = float(row.get("percentage") or 0.0)
            if total_classes <= 0 or percentage >= threshold:
                continue

            alert_row = dict(row)
            alert_row["subject_name"] = assignment["subject_name"]
            alert_row["subject_code"] = assignment["subject_code"]
            alert_row["min_attendance"] = threshold
            alert_row["required_classes"] = _classes_needed_to_reach_threshold(
                total_classes=total_classes,
                present_classes=present_classes,
                threshold=threshold,
            )
            alert_row["alert_message"] = _build_low_attendance_message(alert_row)
            if create_alert(
                alert_row["student_erp"],
                "low_attendance",
                alert_row["alert_message"],
            ):
                created_rows.append(alert_row)
        return created_rows

    rows = query_all(
        """
        SELECT
            a.student_erp,
            st.name,
            st.email,
            st.phone,
            s.subject_name,
            s.subject_code,
            COALESCE(sl.min_attendance, 75.0) AS min_attendance,
            COUNT(a.id) AS total_classes,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS present_classes,
            ROUND(
                SUM(CASE WHEN a.status = 'Present' THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(a.id),
                2
            ) AS percentage
        FROM attendance a
        JOIN students st ON st.erp_id = a.student_erp
        JOIN subjects s ON s.id = a.subject_id
        LEFT JOIN semester_timelines sl
            ON sl.course = st.course
            AND sl.branch = st.branch
            AND sl.year = st.year
            AND sl.semester = st.semester
        GROUP BY a.student_erp, s.subject_name, s.subject_code, sl.min_attendance
        HAVING percentage < COALESCE(sl.min_attendance, 75.0)
        """
    )
    for row in rows:
        row["required_classes"] = _classes_needed_to_reach_threshold(
            total_classes=int(row["total_classes"] or 0),
            present_classes=int(row["present_classes"] or 0),
            threshold=float(row["min_attendance"] or 75.0),
        )
        row["alert_message"] = _build_low_attendance_message(row)
        if create_alert(row["student_erp"], "low_attendance", row["alert_message"]):
            created_rows.append(row)
    return created_rows


def _build_department_today_schedule(department: str) -> dict[str, Any]:
    """Return today's department timetable with active/upcoming status."""
    current_dt = datetime.now()
    current_day = current_dt.strftime("%A")
    current_date = current_dt.strftime("%Y-%m-%d")
    current_time = current_dt.strftime("%H:%M")
    rows = query_all(
        """
        SELECT
            tt.id AS timetable_id,
            tt.section,
            tt.start_time,
            tt.end_time,
            tt.room_no,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester,
            regular_teacher.name AS original_teacher_name,
            COALESCE(substitute_teacher.name, regular_teacher.name) AS teacher_name,
            CASE WHEN sa.id IS NOT NULL THEN 1 ELSE 0 END AS is_substitute
        FROM timetable tt
        JOIN subjects s ON s.id = tt.subject_id
        JOIN teachers regular_teacher ON regular_teacher.id = tt.teacher_id
        LEFT JOIN teacher_subjects ts
            ON ts.teacher_id = tt.teacher_id
            AND ts.subject_id = tt.subject_id
            AND ts.section = tt.section
        LEFT JOIN substitute_assignments sa
            ON sa.assignment_id = ts.id
            AND sa.assignment_date = ?
        LEFT JOIN teachers substitute_teacher ON substitute_teacher.id = sa.substitute_teacher_id
        WHERE tt.day_of_week = ?
        ORDER BY tt.start_time, tt.section, s.subject_name
        """,
        (current_date, current_day),
    )
    rows = [
        row for row in rows if _department_matches_branch(department, row.get("branch"))
    ]
    filtered_rows: list[dict[str, Any]] = []
    excluded_groups: set[str] = set()
    for row in rows:
        excluded_day = get_academic_calendar_day_status(
            row["course"],
            row["branch"],
            int(row["year"]),
            int(row["semester"]),
            current_date,
        )
        if excluded_day and excluded_day["day_type"] in ACADEMIC_EXCLUDED_DAY_TYPES:
            excluded_groups.add(
                (
                    f"{row['course']} | {row['branch']} | Year {row['year']} | "
                    f"Semester {row['semester']} ({excluded_day['day_type'].replace('_', ' ')})"
                )
            )
            continue
        filtered_rows.append(row)
    rows = filtered_rows
    for row in rows:
        row["status"] = (
            "Live Now"
            if row["start_time"] <= current_time <= row["end_time"]
            else ("Upcoming" if current_time < row["start_time"] else "Completed")
        )
    teacher_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        teacher_entry = teacher_map.setdefault(
            row["teacher_name"],
            {
                "teacher_name": row["teacher_name"],
                "class_count": 0,
                "subject_labels": [],
            },
        )
        teacher_entry["class_count"] += 1
        label = f"{row['subject_code']} ({row['section']})"
        if label not in teacher_entry["subject_labels"]:
            teacher_entry["subject_labels"].append(label)
    return {
        "current_day": current_day,
        "current_time": current_time,
        "rows": rows,
        "teacher_summary": sorted(
            teacher_map.values(),
            key=lambda item: (item["teacher_name"], item["class_count"]),
        ),
        "active_count": sum(1 for row in rows if row["status"] == "Live Now"),
        "upcoming_count": sum(1 for row in rows if row["status"] == "Upcoming"),
        "calendar_notice": (
            "Academic calendar exclusion active today for: "
            + ", ".join(sorted(excluded_groups))
            if excluded_groups
            else None
        ),
    }


def get_student_dashboard_context(student_erp: str) -> dict[str, Any]:
    """Assemble student analytics, alerts, and chart data."""
    student = get_student_by_erp(student_erp)
    attendance_overview = get_student_attendance_overview(student_erp)
    subject_percentages = attendance_overview["subject_percentages"]
    alerts = get_recent_alerts(student_erp=student_erp, limit=10)
    overall_attendance = float(attendance_overview["overall_attendance"])
    risk_prediction = predict_student_risk(student_erp) or {
        "risk_score": 0.0,
        "at_risk": False,
        "label": "Unknown",
        "confidence": 0.0,
        "message": "Risk score will appear after attendance data is recorded.",
        "features": {
            "attendance_percentage": overall_attendance,
            "cgpa": float(student.get("cgpa") or 0.0) if student else 0.0,
            "assignments_percentage": float(student.get("assignments_percentage") or 0.0)
            if student
            else 0.0,
        },
    }
    risk_prediction = dict(risk_prediction)
    risk_prediction["features"] = dict(risk_prediction.get("features") or {})
    risk_prediction["features"]["attendance_percentage"] = overall_attendance
    risk_prediction["message"] = (
        f"{risk_prediction.get('message', '').strip()} "
        "Attendance coverage now reflects scheduled lectures from the semester timeline."
    ).strip()
    return {
        "subject_percentages": subject_percentages,
        "alerts": alerts,
        "overall_attendance": overall_attendance,
        "risk_prediction": risk_prediction,
        "attendance_overview": attendance_overview,
    }


def find_department_risks(department: str) -> list[dict[str, Any]]:
    """Return attendance risk rows for a department."""
    students = query_all(
        """
        SELECT erp_id AS student_erp, name, section, course, branch, year, semester, phone, email
        FROM students
        ORDER BY section, name
        """
    )
    students = [
        student
        for student in students
        if _department_matches_branch(department, student.get("branch"))
    ]
    rows: list[dict[str, Any]] = []
    for student in students:
        attendance_overview = get_student_attendance_overview(student["student_erp"])
        percentage = float(attendance_overview["overall_attendance"] or 0.0)
        minimum = float(attendance_overview["min_attendance"] or 75.0)
        total_classes = int(attendance_overview["total_scheduled_classes"] or 0)
        weakest_subject = attendance_overview.get("weakest_subject") or {}
        if total_classes <= 0:
            risk_level = "No Data"
        elif percentage < minimum - 10:
            risk_level = "High"
        elif percentage < minimum:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        rows.append(
            {
                **student,
                "percentage": percentage,
                "min_attendance": minimum,
                "total_classes": total_classes,
                "present_classes": int(attendance_overview["total_present_classes"] or 0),
                "missed_classes": int(attendance_overview["total_missed_classes"] or 0),
                "unmarked_classes": int(attendance_overview["total_unmarked_classes"] or 0),
                "risk_level": risk_level,
                "weakest_subject_name": weakest_subject.get("subject_name"),
                "weakest_subject_code": weakest_subject.get("subject_code"),
                "required_classes": int(attendance_overview.get("required_classes") or 0),
            }
        )
    order = {"High": 0, "Medium": 1, "Low": 2, "No Data": 3}
    return sorted(rows, key=lambda item: (order[item["risk_level"]], item["percentage"], item["name"]))


def get_department_overview(department: str) -> dict[str, Any]:
    """Build department summary for HOD dashboard."""
    risks = find_department_risks(department)
    today_schedule = _build_department_today_schedule(department)
    section_map: dict[str, dict[str, Any]] = {}
    for student in risks:
        key = (
            f"{student['course']} | Year {student['year']} | "
            f"Semester {student['semester']} | Section {student['section']}"
        )
        section_entry = section_map.setdefault(
            key,
            {
                "label": key,
                "section": student["section"],
                "course": student["course"],
                "year": student["year"],
                "semester": student["semester"],
                "student_count": 0,
                "attendance_values": [],
                "high_risk_count": 0,
                "medium_risk_count": 0,
            },
        )
        section_entry["student_count"] += 1
        if int(student["total_classes"] or 0) > 0:
            section_entry["attendance_values"].append(float(student["percentage"] or 0.0))
        if student["risk_level"] == "High":
            section_entry["high_risk_count"] += 1
        elif student["risk_level"] == "Medium":
            section_entry["medium_risk_count"] += 1

    section_snapshots = []
    for section_entry in section_map.values():
        attendance_values = section_entry.pop("attendance_values")
        section_entry["average_attendance"] = (
            round(mean(attendance_values), 2) if attendance_values else 0.0
        )
        section_snapshots.append(section_entry)
    teacher_count = int(
        (query_one(
            """
            SELECT COUNT(*) AS teacher_count
            FROM teachers
            WHERE LOWER(department) = LOWER(?)
            """,
            (department,),
        ) or {}).get("teacher_count")
        or 0
    )
    return {
        "risks": risks,
        "high_risk_count": sum(1 for item in risks if item["risk_level"] == "High"),
        "medium_risk_count": sum(1 for item in risks if item["risk_level"] == "Medium"),
        "low_risk_count": sum(1 for item in risks if item["risk_level"] == "Low"),
        "no_data_count": sum(1 for item in risks if item["risk_level"] == "No Data"),
        "total_students": len(risks),
        "total_teachers": teacher_count,
        "section_snapshots": sorted(section_snapshots, key=lambda item: item["label"]),
        "section_count": len(section_snapshots),
        "today_schedule": today_schedule,
        "classes_today_count": len(today_schedule["rows"]),
    }


def get_department_overall_attendance(department: str) -> list[dict[str, Any]]:
    """Return overall attendance rows for all students in the department."""
    students = query_all(
        """
        SELECT erp_id, name, phone, branch
        FROM students
        """
    )
    students = [
        student
        for student in students
        if _department_matches_branch(department, student.get("branch"))
    ]

    result = []

    for student in students:
        overview = get_student_attendance_overview(student["erp_id"])
        percentage = float(overview["overall_attendance"] or 0.0)
        minimum = float(overview["min_attendance"] or 75.0)
        total_classes = int(overview["total_scheduled_classes"] or 0)
        result.append(
            {
                "erp_id": student["erp_id"],
                "name": student["name"],
                "percentage": round(percentage, 2),
                "phone": student.get("phone"),
                "min_attendance": minimum,
                "total_classes": total_classes,
                "present_classes": int(overview["total_present_classes"] or 0),
                "required_classes": int(overview.get("required_classes") or 0),
                "is_below_threshold": total_classes > 0 and percentage < minimum,
            }
        )

    return sorted(
        result,
        key=lambda item: (
            not item["is_below_threshold"],
            float(item["percentage"]),
            item["name"],
        ),
    )
