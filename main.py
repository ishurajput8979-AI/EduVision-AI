"""Application entry point for the AI college attendance and management system."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

from flask import Flask, redirect, render_template, session, url_for

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        """Fallback no-op when python-dotenv is unavailable."""
        return False

BASE_DIR = Path(__file__).resolve().parent
for env_file in (BASE_DIR / ".env", BASE_DIR / "twilio.env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

from auth_utils import (
    PASSWORD_MIN_LENGTH,
    PASSWORD_POLICY_HINT,
    get_password_suggestion_examples,
)
from database import initialize_database, load_ml_model, reset_system_state
from routes import register_blueprints
from security_config import init_rate_limiting, configure_session_security, add_security_headers
from services.attendance import run_live_camera_attendance


def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY environment variable must be set.")
    initialize_database()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = secret_key
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    
    # Initialize security features
    init_rate_limiting(app)
    configure_session_security(app)
    add_security_headers(app)
    
    # Load ML model at startup
    load_ml_model()

    register_blueprints(app)

    @app.context_processor
    def inject_password_policy() -> dict[str, object]:
        """Expose password policy guidance to all templates."""
        return {
            "password_min_length": PASSWORD_MIN_LENGTH,
            "password_policy_hint": PASSWORD_POLICY_HINT,
            "password_examples": get_password_suggestion_examples(),
        }

    @app.route("/")
    def index():
        """Render the public home page or redirect authenticated users."""
        role = session.get("role")
        if role == "admin":
            return redirect(url_for("admin.dashboard"))
        if role == "hod":
            return redirect(url_for("hod.dashboard"))
        if role == "teacher":
            return redirect(url_for("teacher.dashboard"))
        if role == "student":
            return redirect(url_for("student.dashboard"))
        return render_template("home.html")

    @app.errorhandler(403)
    def forbidden(_error):
        """Render a friendly forbidden page."""
        return render_template("403.html"), 403

    @app.errorhandler(404)
    def not_found(_error):
        """Render a friendly not-found page."""
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def payload_too_large(_error):
        """Render a friendly file-size error page."""
        return render_template("413.html"), 413

    @app.errorhandler(500)
    def server_error(_error):
        """Render a friendly server error page."""
        return render_template("500.html"), 500

    return app


def parse_args() -> argparse.Namespace:
    """Parse optional CLI flags for live camera attendance capture."""
    parser = argparse.ArgumentParser(
        description="AI-based College Attendance and Management System"
    )
    parser.add_argument(
        "--camera-attendance",
        action="store_true",
        help="Open a live camera window and mark attendance for an assignment.",
    )
    parser.add_argument(
        "--reset-system",
        action="store_true",
        help="Delete existing database and generated media, then recreate a clean system.",
    )
    parser.add_argument("--assignment-id", type=int, help="Teacher assignment ID.")
    parser.add_argument("--teacher-id", type=int, help="Teacher ID for the assignment.")
    parser.add_argument(
        "--attendance-date",
        type=str,
        default=None,
        help="Attendance date in YYYY-MM-DD format.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask host.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.reset_system:
        reset_system_state()
        print("System reset completed.")
    if arguments.camera_attendance:
        if not arguments.assignment_id or not arguments.teacher_id:
            raise SystemExit("--assignment-id and --teacher-id are required for camera mode.")
        result = run_live_camera_attendance(
            assignment_id=arguments.assignment_id,
            teacher_id=arguments.teacher_id,
            attendance_date=arguments.attendance_date,
        )
        print(result)
    else:
        application = create_app()
        application.run(
            host=arguments.host,
            port=arguments.port,
            debug=arguments.debug,
            use_reloader=arguments.debug,
        )
