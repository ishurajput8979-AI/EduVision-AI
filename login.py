"""Compatibility wrapper for auth routes and helpers."""

from routes.auth import auth_bp
from services.auth import (
    authenticate_user,
    generate_otp,
    login_required,
    redirect_to_dashboard,
    roles_required,
)

__all__ = [
    "auth_bp",
    "authenticate_user",
    "generate_otp",
    "login_required",
    "redirect_to_dashboard",
    "roles_required",
]
