"""Authentication services and access helpers."""

from __future__ import annotations

import os
import random
from functools import wraps
from typing import Callable

from flask import Response, flash, redirect, request, session, url_for

from auth_utils import JWT_COOKIE_NAME, generate_jwt_token, verify_jwt_token, verify_password
from database import get_user_by_id, get_user_by_username


def _sync_session_state(user: dict, token: str) -> None:
    """Keep session state aligned with the authenticated JWT user."""
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["linked_erp"] = user.get("linked_erp")
    session["linked_teacher_id"] = user.get("linked_teacher_id")
    session["access_token"] = token
    request.user_id = int(user["id"])
    request.username = user["username"]
    request.role = user["role"]


def _resolve_authenticated_user() -> dict | None:
    """Validate the current JWT and return the linked user record."""
    token = request.cookies.get(JWT_COOKIE_NAME) or session.get("access_token")
    if not token:
        return None

    payload = verify_jwt_token(token)
    if not payload:
        return None

    user = get_user_by_id(int(payload["user_id"]))
    if not user:
        return None

    _sync_session_state(user, token)
    return user


def login_required(view: Callable) -> Callable:
    """Protect a route so it can only be accessed by an authenticated user."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not _resolve_authenticated_user():
            session.clear()
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped_view


def roles_required(*allowed_roles: str) -> Callable:
    """Restrict a route to one or more user roles."""

    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapped_view(*args, **kwargs):
            if getattr(request, "role", session.get("role")) not in allowed_roles:
                flash("You do not have permission to access that page.", "danger")
                return redirect(url_for("index"))
            return view(*args, **kwargs)

        return wrapped_view

    return decorator


def authenticate_user(username: str, password: str) -> dict | None:
    """Validate login credentials and return the user when successful."""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return user


def redirect_to_dashboard(role: str | None = None) -> str:
    """Resolve the correct dashboard route name for the user role."""
    effective_role = role or session.get("role")
    if effective_role == "admin":
        return url_for("admin.dashboard")
    if effective_role == "hod":
        return url_for("hod.dashboard")
    if effective_role == "teacher":
        return url_for("teacher.dashboard")
    if effective_role == "student":
        return url_for("student.dashboard")
    return url_for("auth.login")


def issue_auth_token(user: dict) -> str:
    """Generate a JWT access token for the supplied user."""
    return generate_jwt_token(
        user_id=int(user["id"]),
        username=user["username"],
        role=user["role"],
    )


def start_user_session(user: dict) -> str:
    """Store authenticated user state in the Flask session."""
    token = issue_auth_token(user)
    session.clear()
    _sync_session_state(user, token)
    return token


def attach_auth_cookie(response: Response, token: str) -> Response:
    """Attach the JWT token to a secure HTTP-only cookie."""
    response.set_cookie(
        JWT_COOKIE_NAME,
        token,
        max_age=24 * 3600,
        httponly=True,
        samesite="Lax",
        secure=os.environ.get("ENV", "development") == "production",
    )
    return response


def clear_auth_state(response: Response) -> Response:
    """Clear session and JWT cookie."""
    session.clear()
    response.delete_cookie(JWT_COOKIE_NAME)
    return response


def generate_otp() -> str:
    """Generate a six-digit OTP for password reset."""
    return f"{random.SystemRandom().randint(0, 999999):06d}"
