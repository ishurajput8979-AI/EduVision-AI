"""Authentication helpers, validators, and small shared utilities."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any

import bcrypt
import jwt
from flask import jsonify, request

# Configuration
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable must be set.")
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))
JWT_COOKIE_NAME = "access_token"
BCRYPT_ROUNDS = 12  # Slow hashing = secure
PASSWORD_MIN_LENGTH = int(os.environ.get("PASSWORD_MIN_LENGTH", "12"))
PASSWORD_MAX_LENGTH = 128
PASSWORD_POLICY_HINT = (
    f"Use at least {PASSWORD_MIN_LENGTH} characters with uppercase, lowercase, a number, "
    "and a special character like @, #, !, or $. Spaces are not allowed."
)
PASSWORD_SUGGESTION_EXAMPLES = (
    "ExampleOnly#2026",
    "CampusExample@29",
)
COMMON_WEAK_PASSWORDS = {
    "12345678",
    "123456789",
    "1234567890",
    "admin123",
    "admin@123",
    "password",
    "password1",
    "password123",
    "qwerty123",
    "teacher123",
    "student123",
    "welcome123",
}


def hash_password(password: str) -> str:
    """Hash any non-empty secret using bcrypt."""
    if password is None:
        raise ValueError("Password cannot be empty")
    value = str(password)
    if not value:
        raise ValueError("Password cannot be empty")
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(value.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify raw password against bcrypt hash.
    
    Args:
        password: Plain text password to verify
        password_hash: Stored bcrypt hash
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except (ValueError, TypeError):
        return False


def generate_jwt_token(user_id: int, username: str, role: str, expires_in: int = None) -> str:
    """Generate JWT token with user info.
    
    Args:
        user_id: User ID
        username: Username
        role: User role (admin, hod, teacher, student)
        expires_in: Expiry time in hours (default: 24)
        
    Returns:
        JWT token string
    """
    if expires_in is None:
        expires_in = JWT_EXPIRY_HOURS
    
    payload = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'iat': datetime.now(timezone.utc).timestamp(),
        'exp': (datetime.now(timezone.utc) + timedelta(hours=expires_in)).timestamp(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict[str, Any] | None:
    """Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload if valid, None if expired or invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def token_required(view):
    """Decorator for API endpoints requiring JWT auth.
    
    Usage:
        @app.route('/api/data')
        @token_required
        def get_data():
            user_id = request.user_id
            ...
    """
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        token = request.cookies.get(JWT_COOKIE_NAME)

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        # Add user info to request context
        request.user_id = payload['user_id']
        request.username = payload['username']
        request.role = payload['role']

        return view(*args, **kwargs)
    
    return wrapped_view


def parse_csv_list(raw_value: str) -> list[str]:
    """Convert comma-separated string into cleaned list of unique values.
    
    Args:
        raw_value: Comma-separated string
        
    Returns:
        List of unique, cleaned values
    """
    cleaned_values: list[str] = []
    for item in (raw_value or "").split(","):
        normalized = item.strip()
        if normalized and normalized not in cleaned_values:
            cleaned_values.append(normalized)
    return cleaned_values


# ============ VALIDATION UTILITIES ============

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def get_password_suggestion_examples() -> tuple[str, ...]:
    """Return reusable strong-password examples for templates."""
    return PASSWORD_SUGGESTION_EXAMPLES


def _format_password_requirements(requirements: list[str]) -> str:
    """Convert a list of requirements into a readable sentence fragment."""
    if len(requirements) == 1:
        return requirements[0]
    return ", ".join(requirements[:-1]) + f", and {requirements[-1]}"


def validate_username(username: str) -> str:
    """Validate username format.
    
    Args:
        username: Username to validate
        
    Returns:
        Validated username
        
    Raises:
        ValidationError: If username is invalid
    """
    if not username or len(username) < 3 or len(username) > 50:
        raise ValidationError("Username must be 3-50 characters")
    
    if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
        raise ValidationError("Username can only contain letters, numbers, dots, dashes, underscores")
    
    return username.strip()


def validate_password_strength(password: str) -> str:
    """Validate password strength.
    
    Requirements:
    - At least 12 characters
    - Contains uppercase letter
    - Contains lowercase letter
    - Contains digit
    - Contains special character
    - Does not contain spaces
    
    Args:
        password: Password to validate
        
    Returns:
        Validated password
        
    Raises:
        ValidationError: If password doesn't meet requirements
    """
    if password is None:
        raise ValidationError("Password is required.")

    if not password:
        raise ValidationError("Password is required.")

    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValidationError(f"Password cannot exceed {PASSWORD_MAX_LENGTH} characters.")

    if re.search(r"\s", password):
        raise ValidationError(
            "Password cannot contain spaces. "
            "Use letters, numbers, and special characters like @ or # only."
        )

    missing_requirements: list[str] = []
    if password.lower() in COMMON_WEAK_PASSWORDS:
        missing_requirements.append(
            "a unique password that is not a common pattern like admin123 or teacher123"
        )
    if len(password) < PASSWORD_MIN_LENGTH:
        missing_requirements.append(f"at least {PASSWORD_MIN_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        missing_requirements.append("one uppercase letter")
    if not re.search(r"[a-z]", password):
        missing_requirements.append("one lowercase letter")
    if not re.search(r"\d", password):
        missing_requirements.append("one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        missing_requirements.append("one special character like @ or #")

    if missing_requirements:
        raise ValidationError(
            "Password is too weak. Add "
            f"{_format_password_requirements(missing_requirements)}. "
            f"Example: {PASSWORD_SUGGESTION_EXAMPLES[0]}"
        )

    return password


def validate_email(email: str) -> str:
    """Validate email format.
    
    Args:
        email: Email to validate
        
    Returns:
        Validated email
        
    Raises:
        ValidationError: If email format is invalid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format")
    
    return email.strip().lower()


def validate_phone(phone: str) -> str:
    """Validate a phone number using a simple India-friendly format."""
    normalized = re.sub(r"\s+", "", phone or "")
    if not re.fullmatch(r"^\+?\d{10,15}$", normalized):
        raise ValidationError("Phone number must contain 10 to 15 digits.")
    return normalized


def validate_erp_id(erp_id: str) -> str:
    """Validate ERP ID format.
    
    Format: ERP followed by 3-6 digits (e.g., ERP001, ERP123456)
    
    Args:
        erp_id: ERP ID to validate
        
    Returns:
        Validated ERP ID
        
    Raises:
        ValidationError: If ERP ID format is invalid
    """
    if not erp_id or not re.match(r'^ERP\d{3,6}$', erp_id, re.IGNORECASE):
        raise ValidationError("Invalid ERP ID format (must be ERP followed by 3-6 digits)")
    
    return erp_id.upper().strip()


def normalize_username(username: str) -> str:
    """Normalize usernames for case-insensitive lookups."""
    if not username:
        raise ValidationError("Username is required.")
    return username.strip()


def sanitize_input(value: str, max_length: int = 255) -> str:
    """Sanitize user input to prevent injection attacks.
    
    Args:
        value: Input to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input
        
    Raises:
        ValidationError: If input contains dangerous patterns
    """
    value = str(value).strip()
    
    if len(value) > max_length:
        raise ValidationError(f"Input exceeds maximum length of {max_length} characters")
    
    # Check for common SQL injection patterns
    dangerous_patterns = [';', '--', '/*', '*/', 'xp_', 'sp_', 'DROP', 'DELETE', 'INSERT']
    for pattern in dangerous_patterns:
        if pattern.lower() in value.lower():
            raise ValidationError(f"Invalid input: contains '{ pattern}'")
    
    return value


def validate_file_upload(filename: str, allowed_extensions: set[str]) -> str:
    """Validate uploaded filename.
    
    Args:
        filename: Uploaded filename
        allowed_extensions: Set of allowed extensions (e.g., {'.jpg', '.png'})
        
    Returns:
        Validated filename
        
    Raises:
        ValidationError: If filename is invalid
    """
    if not filename:
        raise ValidationError("No filename provided")
    
    ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    if not ext or ext not in allowed_extensions:
        raise ValidationError(f"File type not allowed. Allowed: {allowed_extensions}")
    
    # Prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValidationError("Invalid filename: contains path characters")
    
    return filename
