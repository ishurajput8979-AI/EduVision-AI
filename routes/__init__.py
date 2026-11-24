"""Blueprint registration helpers."""

from flask import Flask

from .admin import admin_bp
from .api import api_bp
from .auth import auth_bp
from .hod import hod_bp
from .signup import signup_bp
from .student import student_bp
from .teacher import teacher_bp


def register_blueprints(app: Flask) -> None:
    """Register all application blueprints."""
    for blueprint in (
        auth_bp,
        signup_bp,
        admin_bp,
        teacher_bp,
        student_bp,
        hod_bp,
        api_bp,
    ):
        app.register_blueprint(blueprint)
