"""JWT-protected JSON API endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth_utils import token_required
from database import get_teacher_by_id, get_user_by_id
from services.analytics import get_department_overview, get_student_dashboard_context

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/me")
@token_required
def me():
    """Return the authenticated JWT identity."""
    return jsonify(
        {
            "user_id": getattr(request, "user_id", None),
            "username": getattr(request, "username", None),
            "role": getattr(request, "role", None),
        }
    )


@api_bp.route("/student/overview")
@token_required
def student_overview():
    """Return student dashboard data for JWT-authenticated student users."""
    if getattr(request, "role", None) != "student":
        return jsonify({"error": "Forbidden"}), 403
    user = get_user_by_id(int(request.user_id))
    student_erp = user.get("linked_erp") if user else None
    if not student_erp:
        return jsonify({"error": "Student record not linked"}), 400
    return jsonify(get_student_dashboard_context(student_erp))


@api_bp.route("/hod/overview")
@token_required
def hod_overview():
    """Return department-level overview for JWT-authenticated HOD users."""
    if getattr(request, "role", None) != "hod":
        return jsonify({"error": "Forbidden"}), 403
    user = get_user_by_id(int(request.user_id))
    teacher_id = user.get("linked_teacher_id") if user else None
    if not teacher_id:
        return jsonify({"error": "Teacher record not linked"}), 400
    teacher = get_teacher_by_id(int(teacher_id))
    return jsonify(get_department_overview(teacher["department"]))
