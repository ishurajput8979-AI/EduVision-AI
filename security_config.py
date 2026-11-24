"""Security configuration for Flask application."""

from __future__ import annotations

import os
from datetime import timedelta

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.environ.get("LIMITER_STORAGE", "memory://"),
    strategy="fixed-window",
)


def init_rate_limiting(app: Flask) -> Limiter:
    """Initialize the shared rate-limiter instance for the Flask app."""
    limiter.init_app(app)
    return limiter


def _is_production_env() -> bool:
    """Return whether the app is running with production security settings."""
    return os.environ.get("ENV", "development").strip().lower() == "production"


def _get_int_env(name: str, default: int) -> int:
    """Safely read integer environment variables."""
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def configure_session_security(app: Flask) -> None:
    """Configure secure session handling.
    
    Args:
        app: Flask application instance
    """
    production_mode = _is_production_env()
    app.config.update(
        # Session security
        SESSION_COOKIE_SECURE=production_mode,  # HTTPS only
        SESSION_COOKIE_HTTPONLY=True,  # No JavaScript access (XSS protection)
        SESSION_COOKIE_SAMESITE='Lax',  # CSRF protection
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
        SESSION_REFRESH_EACH_REQUEST=True,
        
        # Response security headers
        SEND_FILE_MAX_AGE_DEFAULT=_get_int_env('SEND_FILE_MAX_AGE_DEFAULT', 0),
    )


def add_security_headers(app: Flask) -> None:
    """Add security headers to all responses.
    
    Args:
        app: Flask application instance
    """
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        if _is_production_env():
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' data:; "
            "connect-src 'self' https://cdn.jsdelivr.net https://api.twilio.com; "
        )
        return response


# ============ USAGE IN main.py ============
"""
from flask import Flask
from security_config import init_rate_limiting, configure_session_security, add_security_headers

def create_app():
    app = Flask(__name__)
    
    # Configure session security
    configure_session_security(app)
    
    # Initialize rate limiting
    limiter = init_rate_limiting(app)
    
    # Add security headers
    add_security_headers(app)
    
    # In login.py or auth routes:
    # @limiter.limit("5 per minute")
    # def login():
    #     ...
    
    return app
"""
