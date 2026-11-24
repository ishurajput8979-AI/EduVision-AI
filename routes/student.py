"""Student dashboard routes."""

from __future__ import annotations

from flask import Blueprint, render_template, session

from database import get_face_embeddings, get_student_by_erp
from services.analytics import get_student_dashboard_context
from services.auth import login_required, roles_required

student_bp = Blueprint("student", __name__, url_prefix="/student")


@student_bp.route("/dashboard")
@login_required
@roles_required("student")
def dashboard():
    """Render student attendance and risk dashboard."""
    student_erp = session["linked_erp"]
    student = get_student_by_erp(student_erp)
    analytics = get_student_dashboard_context(student_erp)
    embedding_count = len(get_face_embeddings([student_erp]))
    return render_template(
        "student_dashboard.html",
        student=student,
        analytics=analytics,
        embedding_count=embedding_count,
        risk_prediction=analytics["risk_prediction"],
    )
