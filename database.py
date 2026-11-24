"""SQLite database helpers and schema management for the college system."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Iterable

from auth_utils import (
    hash_password,
    normalize_username,
    verify_password,
    validate_email,
    validate_erp_id,
    validate_password_strength,
    validate_phone,
)
from face_embedding_security import FaceEmbeddingManager

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
VIDEOS_DIR = DATASET_DIR / "videos"
ATTENDANCE_MEDIA_DIR = DATASET_DIR / "attendance"
ACADEMIC_CALENDAR_DIR = DATASET_DIR / "academic_calendars"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = MODELS_DIR / os.environ.get(
    "ATTENDANCE_DB_FILE",
    "runtime_college_attendance.sqlite3",
)
MODEL_FILE = MODELS_DIR / os.environ.get(
    "ATTENDANCE_MODEL_FILE",
    "runtime_attendance_risk_model.pkl",
)
_embedding_manager: FaceEmbeddingManager | None = None
DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD")
if not DEFAULT_ADMIN_PASSWORD:
    raise RuntimeError("DEFAULT_ADMIN_PASSWORD environment variable must be set.")
LEGACY_ADMIN_PASSWORDS = ()
ACADEMIC_DAY_TYPES = ("holiday", "half_day", "regular", "important")
ACADEMIC_EXCLUDED_DAY_TYPES = {"holiday", "half_day"}

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'hod', 'teacher', 'student')),
    linked_erp TEXT,
    linked_teacher_id INTEGER,
    email TEXT,
    phone TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(linked_erp) REFERENCES students(erp_id) ON DELETE SET NULL,
    FOREIGN KEY(linked_teacher_id) REFERENCES teachers(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    section TEXT NOT NULL,
    UNIQUE(course, branch, year, semester, section)
);

CREATE TABLE IF NOT EXISTS students (
    erp_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    roll_no TEXT UNIQUE NOT NULL,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    section TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    cgpa REAL NOT NULL DEFAULT 7.0,
    assignments_percentage REAL NOT NULL DEFAULT 80.0,
    registered INTEGER NOT NULL DEFAULT 0 CHECK(registered IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    role TEXT NOT NULL DEFAULT 'teacher' CHECK(role IN ('teacher', 'hod')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_code TEXT UNIQUE NOT NULL,
    subject_name TEXT NOT NULL,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teacher_subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    section TEXT NOT NULL,
    UNIQUE(teacher_id, subject_id, section),
    FOREIGN KEY(teacher_id) REFERENCES teachers(id) ON DELETE CASCADE,
    FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    section TEXT NOT NULL,
    day_of_week TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    room_no TEXT,
    UNIQUE(subject_id, teacher_id, section, day_of_week, start_time),
    FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
    FOREIGN KEY(teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semester_timelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    min_attendance REAL NOT NULL,
    UNIQUE(course, branch, year, semester)
);

CREATE TABLE IF NOT EXISTS academic_calendar_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    title TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS academic_calendar_days (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    semester INTEGER NOT NULL,
    calendar_date TEXT NOT NULL,
    day_type TEXT NOT NULL CHECK(day_type IN ('holiday', 'half_day', 'regular', 'important')),
    reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(course, branch, year, semester, calendar_date)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_erp TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    sample_path TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(student_erp, sample_path),
    FOREIGN KEY(student_erp) REFERENCES students(erp_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_erp TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    attendance_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Present', 'Absent')),
    confidence REAL,
    source TEXT,
    manual_override INTEGER NOT NULL DEFAULT 0 CHECK(manual_override IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(student_erp, subject_id, attendance_date),
    FOREIGN KEY(student_erp) REFERENCES students(erp_id) ON DELETE CASCADE,
    FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
    FOREIGN KEY(teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_erp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved INTEGER NOT NULL DEFAULT 0 CHECK(resolved IN (0, 1)),
    FOREIGN KEY(student_erp) REFERENCES students(erp_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS substitute_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL,
    original_teacher_id INTEGER NOT NULL,
    substitute_teacher_id INTEGER NOT NULL,
    assignment_date TEXT NOT NULL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(assignment_id, assignment_date),
    FOREIGN KEY(assignment_id) REFERENCES teacher_subjects(id) ON DELETE CASCADE,
    FOREIGN KEY(original_teacher_id) REFERENCES teachers(id) ON DELETE CASCADE,
    FOREIGN KEY(substitute_teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS password_reset_otps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    otp_hash TEXT NOT NULL,
    contact_method TEXT NOT NULL CHECK(contact_method IN ('email', 'phone')),
    destination TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attendance_subject_date
ON attendance(subject_id, attendance_date);

CREATE INDEX IF NOT EXISTS idx_attendance_student_date
ON attendance(student_erp, attendance_date);

CREATE INDEX IF NOT EXISTS idx_users_role_username
ON users(role, username);

CREATE INDEX IF NOT EXISTS idx_students_section_lookup
ON students(course, branch, year, semester, section, roll_no);

CREATE INDEX IF NOT EXISTS idx_teacher_subjects_teacher_section
ON teacher_subjects(teacher_id, section);

CREATE INDEX IF NOT EXISTS idx_teacher_subjects_subject_section
ON teacher_subjects(subject_id, section);

CREATE INDEX IF NOT EXISTS idx_timetable_teacher_day
ON timetable(teacher_id, day_of_week, start_time);

CREATE INDEX IF NOT EXISTS idx_timetable_section_day
ON timetable(section, day_of_week, start_time);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_student
ON face_embeddings(student_erp);

CREATE INDEX IF NOT EXISTS idx_substitute_assignment_date
ON substitute_assignments(assignment_date);

CREATE INDEX IF NOT EXISTS idx_password_reset_user
ON password_reset_otps(user_id, expires_at);

CREATE INDEX IF NOT EXISTS idx_alerts_student_created
ON alerts(student_erp, resolved, created_at);

CREATE INDEX IF NOT EXISTS idx_calendar_days_lookup
ON academic_calendar_days(course, branch, year, semester, calendar_date);

CREATE INDEX IF NOT EXISTS idx_calendar_files_lookup
ON academic_calendar_files(course, branch, year, semester, uploaded_at);
"""


def dict_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    """Return SQLite rows as dictionaries for easier downstream use."""
    return {col[0]: row[index] for index, col in enumerate(cursor.description)}


def _handle_remove_readonly(func, path, exc_info) -> None:
    """Retry directory cleanup after clearing a read-only file attribute."""
    target = Path(path)
    os.chmod(target, stat.S_IWRITE)
    if target.is_dir():
        shutil.rmtree(target, onexc=_handle_remove_readonly)
        return
    os.unlink(target)


def _clear_database_contents() -> None:
    """Delete all application rows while keeping the active schema intact."""
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = OFF;")
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table_name in reversed(tables):
            connection.execute(f'DELETE FROM "{table_name}"')
        connection.execute("DELETE FROM sqlite_sequence")
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.commit()


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return the current column names for a table."""
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _table_sql(connection: sqlite3.Connection, table_name: str) -> str:
    """Return the raw CREATE TABLE SQL for one table."""
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return (row or {}).get("sql") or ""


def _apply_schema_migrations(connection: sqlite3.Connection) -> None:
    """Add forward-compatible columns and indexes for existing databases."""
    student_columns = _table_columns(connection, "students")
    if "cgpa" not in student_columns:
        connection.execute(
            "ALTER TABLE students ADD COLUMN cgpa REAL NOT NULL DEFAULT 7.0"
        )
    if "assignments_percentage" not in student_columns:
        connection.execute(
            "ALTER TABLE students ADD COLUMN assignments_percentage REAL NOT NULL DEFAULT 80.0"
        )

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS academic_calendar_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course TEXT NOT NULL,
            branch TEXT NOT NULL,
            year INTEGER NOT NULL,
            semester INTEGER NOT NULL,
            title TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS academic_calendar_days (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course TEXT NOT NULL,
            branch TEXT NOT NULL,
            year INTEGER NOT NULL,
            semester INTEGER NOT NULL,
            calendar_date TEXT NOT NULL,
            day_type TEXT NOT NULL CHECK(day_type IN ('holiday', 'half_day', 'regular', 'important')),
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(course, branch, year, semester, calendar_date)
        );

        CREATE INDEX IF NOT EXISTS idx_users_role_username
        ON users(role, username);

        CREATE INDEX IF NOT EXISTS idx_students_section_lookup
        ON students(course, branch, year, semester, section, roll_no);

        CREATE INDEX IF NOT EXISTS idx_teacher_subjects_teacher_section
        ON teacher_subjects(teacher_id, section);

        CREATE INDEX IF NOT EXISTS idx_teacher_subjects_subject_section
        ON teacher_subjects(subject_id, section);

        CREATE INDEX IF NOT EXISTS idx_timetable_teacher_day
        ON timetable(teacher_id, day_of_week, start_time);

        CREATE INDEX IF NOT EXISTS idx_timetable_section_day
        ON timetable(section, day_of_week, start_time);

        CREATE INDEX IF NOT EXISTS idx_alerts_student_created
        ON alerts(student_erp, resolved, created_at);

        CREATE INDEX IF NOT EXISTS idx_calendar_days_lookup
        ON academic_calendar_days(course, branch, year, semester, calendar_date);

        CREATE INDEX IF NOT EXISTS idx_calendar_files_lookup
        ON academic_calendar_files(course, branch, year, semester, uploaded_at);
        """
    )

    calendar_days_sql = _table_sql(connection, "academic_calendar_days").lower()
    if "regular" not in calendar_days_sql or "important" not in calendar_days_sql:
        connection.executescript(
            """
            ALTER TABLE academic_calendar_days RENAME TO academic_calendar_days_legacy;

            CREATE TABLE academic_calendar_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course TEXT NOT NULL,
                branch TEXT NOT NULL,
                year INTEGER NOT NULL,
                semester INTEGER NOT NULL,
                calendar_date TEXT NOT NULL,
                day_type TEXT NOT NULL CHECK(day_type IN ('holiday', 'half_day', 'regular', 'important')),
                reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(course, branch, year, semester, calendar_date)
            );

            INSERT INTO academic_calendar_days (
                id, course, branch, year, semester, calendar_date, day_type, reason, created_at
            )
            SELECT
                id,
                course,
                branch,
                year,
                semester,
                calendar_date,
                CASE
                    WHEN LOWER(REPLACE(day_type, ' ', '_')) IN ('holiday', 'half_day', 'regular', 'important')
                    THEN LOWER(REPLACE(day_type, ' ', '_'))
                    ELSE 'holiday'
                END,
                reason,
                created_at
            FROM academic_calendar_days_legacy;

            DROP TABLE academic_calendar_days_legacy;

            CREATE INDEX IF NOT EXISTS idx_calendar_days_lookup
            ON academic_calendar_days(course, branch, year, semester, calendar_date);
            """
        )


def ensure_directories() -> None:
    """Create the folders required by the application at startup."""
    for directory in (
        DATASET_DIR,
        IMAGES_DIR,
        VIDEOS_DIR,
        ATTENDANCE_MEDIA_DIR,
        ACADEMIC_CALENDAR_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection configured for dict-style row access."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA busy_timeout = 5000;")
    preferred_journal_mode = os.environ.get("SQLITE_JOURNAL_MODE", "MEMORY")
    try:
        connection.execute(f"PRAGMA journal_mode = {preferred_journal_mode};")
    except sqlite3.OperationalError:
        connection.execute("PRAGMA journal_mode = MEMORY;")
    connection.execute("PRAGMA synchronous = NORMAL;")
    connection.execute("PRAGMA temp_store = MEMORY;")
    connection.row_factory = dict_factory
    return connection


def execute(query: str, params: Iterable[Any] = ()) -> int:
    """Execute a write query and return the last inserted row id."""
    with get_connection() as connection:
        cursor = connection.execute(query, tuple(params))
        connection.commit()
        return cursor.lastrowid


def executemany(query: str, rows: Iterable[Iterable[Any]]) -> None:
    """Execute a batch write query."""
    with get_connection() as connection:
        connection.executemany(query, rows)
        connection.commit()


def query_one(query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    """Fetch a single row from the database."""
    with get_connection() as connection:
        return connection.execute(query, tuple(params)).fetchone()


def query_all(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    """Fetch all rows that match the supplied query."""
    with get_connection() as connection:
        return connection.execute(query, tuple(params)).fetchall()


def _get_embedding_manager() -> FaceEmbeddingManager:
    """Return a shared embedding security manager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = FaceEmbeddingManager()
    return _embedding_manager


def _decode_face_embedding(raw_payload: str) -> list[float]:
    """Decode encrypted embeddings while remaining compatible with legacy rows."""
    try:
        return _get_embedding_manager().decrypt_embedding(raw_payload)
    except ValueError:
        payload = json.loads(raw_payload)
        if not isinstance(payload, list):
            raise ValueError("Stored face embedding payload is invalid.")
        return [float(value) for value in payload]


def _hydrate_face_embedding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decrypt stored embeddings for downstream face-matching code."""
    for row in rows:
        row["embedding_json"] = json.dumps(_decode_face_embedding(row["embedding_json"]))
    return rows


def initialize_database() -> None:
    """Create all required tables and seed the predefined admin account."""
    ensure_directories()
    try:
        with get_connection() as connection:
            connection.executescript(SCHEMA_SQL)
            _apply_schema_migrations(connection)
            connection.commit()
    except sqlite3.OperationalError:
        if DB_PATH.exists():
            DB_PATH.unlink()
        with get_connection() as connection:
            connection.executescript(SCHEMA_SQL)
            _apply_schema_migrations(connection)
            connection.commit()
    seed_admin_user()


def seed_admin_user() -> None:
    """Insert the fixed admin account if it does not already exist."""
    validate_password_strength(DEFAULT_ADMIN_PASSWORD)
    admin_user = get_user_by_username("admin")
    if admin_user:
        for legacy_password in LEGACY_ADMIN_PASSWORDS:
            if verify_password(legacy_password, admin_user["password"]):
                update_user_password(
                    int(admin_user["id"]),
                    hash_password(DEFAULT_ADMIN_PASSWORD),
                )
                print("Admin password upgraded from legacy default to strong password.")
                break
        return
    create_user(
        username="admin",
        password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
        role="admin",
    )


def reset_system_state() -> None:
    """Reset database state and recreate active folders for a fresh start."""
    ensure_directories()
    if DB_PATH.exists():
        try:
            _clear_database_contents()
        except sqlite3.Error:
            DB_PATH.unlink()
    for directory in (IMAGES_DIR, VIDEOS_DIR, ATTENDANCE_MEDIA_DIR, ACADEMIC_CALENDAR_DIR):
        if directory.exists():
            try:
                shutil.rmtree(directory, onexc=_handle_remove_readonly)
            except PermissionError as exc:
                print(f"Media cleanup skipped for {directory}: {exc}")
        directory.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists():
        try:
            MODEL_FILE.unlink()
        except PermissionError as exc:
            print(f"Model cleanup skipped for {MODEL_FILE}: {exc}")
    initialize_database()


def create_section(course: str, branch: str, year: int, semester: int, section: str) -> None:
    """Create or update a logical section record for students and timetable mapping."""
    execute(
        """
        INSERT OR IGNORE INTO sections (course, branch, year, semester, section)
        VALUES (?, ?, ?, ?, ?)
        """,
        (course.strip(), branch.strip(), int(year), int(semester), section.strip().upper()),
    )


def _normalize_cgpa(raw_value: Any) -> float:
    """Validate and normalize CGPA values."""
    value = 7.0 if raw_value in (None, "") else float(raw_value)
    if not 0.0 <= value <= 10.0:
        raise ValueError("CGPA must be between 0 and 10.")
    return round(value, 2)


def _normalize_assignments_percentage(raw_value: Any) -> float:
    """Validate and normalize assignment completion/performance percentage."""
    value = 80.0 if raw_value in (None, "") else float(raw_value)
    if not 0.0 <= value <= 100.0:
        raise ValueError("Assignments percentage must be between 0 and 100.")
    return round(value, 2)


def create_student(student_data: dict[str, Any]) -> None:
    """Add a student record and auto-create the related section if required."""
    erp_id = validate_erp_id(student_data["erp_id"])
    email = validate_email(student_data["email"]) if student_data.get("email") else None
    phone = validate_phone(student_data["phone"]) if student_data.get("phone") else None
    section = student_data["section"].strip().upper()
    cgpa = _normalize_cgpa(student_data.get("cgpa"))
    assignments_percentage = _normalize_assignments_percentage(
        student_data.get("assignments_percentage")
    )

    create_section(
        student_data["course"].strip(),
        student_data["branch"].strip(),
        int(student_data["year"]),
        int(student_data["semester"]),
        section,
    )
    execute(
        """
        INSERT INTO students (
            erp_id, name, roll_no, course, branch, year, semester, section,
            email, phone, cgpa, assignments_percentage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            erp_id,
            student_data["name"].strip(),
            student_data["roll_no"].strip(),
            student_data["course"].strip(),
            student_data["branch"].strip(),
            int(student_data["year"]),
            int(student_data["semester"]),
            section,
            email,
            phone,
            cgpa,
            assignments_percentage,
        ),
    )


def create_teacher(teacher_data: dict[str, Any]) -> int:
    """Create a teacher or HOD record and the corresponding login account."""
    role = teacher_data["role"].strip().lower()
    if role not in {"teacher", "hod"}:
        raise ValueError("Role must be either teacher or hod.")
    validate_password_strength(teacher_data["password"])
    email = validate_email(teacher_data["email"]) if teacher_data.get("email") else None
    phone = validate_phone(teacher_data["phone"]) if teacher_data.get("phone") else None
    teacher_code = teacher_data["teacher_code"].strip().upper()

    teacher_id = execute(
        """
        INSERT INTO teachers (teacher_code, name, department, email, phone, role)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            teacher_code,
            teacher_data["name"].strip(),
            teacher_data["department"].strip(),
            email,
            phone,
            role,
        ),
    )
    create_user(
        username=teacher_code,
        password_hash=hash_password(teacher_data["password"]),
        role=role,
        linked_teacher_id=teacher_id,
        email=email,
        phone=phone,
    )
    return teacher_id


def create_subject(subject_data: dict[str, Any]) -> int:
    """Insert a subject that can later be assigned to teachers and timetable slots."""
    return execute(
        """
        INSERT INTO subjects (subject_code, subject_name, course, branch, year, semester)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            subject_data["subject_code"].strip().upper(),
            subject_data["subject_name"].strip(),
            subject_data["course"].strip(),
            subject_data["branch"].strip(),
            int(subject_data["year"]),
            int(subject_data["semester"]),
        ),
    )


def assign_teacher_subject(teacher_id: int, subject_id: int, section: str) -> int:
    """Assign a teacher to a subject and section combination."""
    return execute(
        """
        INSERT INTO teacher_subjects (teacher_id, subject_id, section)
        VALUES (?, ?, ?)
        """,
        (teacher_id, subject_id, section.upper()),
    )


def create_timetable_entry(timetable_data: dict[str, Any]) -> int:
    """Create a timetable row for classroom scheduling."""
    validate_timetable_slot(
        subject_id=int(timetable_data["subject_id"]),
        teacher_id=int(timetable_data["teacher_id"]),
        section=timetable_data["section"],
        day_of_week=timetable_data["day_of_week"],
        start_time=timetable_data["start_time"],
        end_time=timetable_data["end_time"],
    )
    return execute(
        """
        INSERT INTO timetable (
            subject_id, teacher_id, section, day_of_week, start_time, end_time, room_no
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(timetable_data["subject_id"]),
            int(timetable_data["teacher_id"]),
            timetable_data["section"].upper(),
            timetable_data["day_of_week"],
            timetable_data["start_time"],
            timetable_data["end_time"],
            timetable_data.get("room_no"),
        ),
    )


def get_subject_by_id(subject_id: int) -> dict[str, Any] | None:
    """Fetch a subject record by primary key."""
    return query_one("SELECT * FROM subjects WHERE id = ?", (subject_id,))


def validate_timetable_slot(
    subject_id: int,
    teacher_id: int,
    section: str,
    day_of_week: str,
    start_time: str,
    end_time: str,
    exclude_timetable_id: int | None = None,
) -> None:
    """Prevent overlapping timetable slots for a teacher or academic section."""
    if end_time <= start_time:
        raise ValueError("Timetable end time must be later than start time.")

    subject = get_subject_by_id(subject_id)
    if not subject:
        raise ValueError("Selected subject does not exist.")

    conflicts = query_all(
        """
        SELECT
            tt.id,
            tt.teacher_id,
            tt.subject_id,
            tt.section,
            tt.day_of_week,
            tt.start_time,
            tt.end_time,
            t.name AS teacher_name,
            s.subject_name,
            s.course,
            s.branch,
            s.year,
            s.semester
        FROM timetable tt
        JOIN teachers t ON t.id = tt.teacher_id
        JOIN subjects s ON s.id = tt.subject_id
        WHERE tt.day_of_week = ?
          AND (? < tt.end_time AND tt.start_time < ?)
          AND (? IS NULL OR tt.id != ?)
        """,
        (day_of_week, start_time, end_time, exclude_timetable_id, exclude_timetable_id),
    )

    for conflict in conflicts:
        same_teacher = int(conflict["teacher_id"]) == int(teacher_id)
        same_section = (
            conflict["section"].upper() == section.upper()
            and conflict["course"] == subject["course"]
            and conflict["branch"] == subject["branch"]
            and int(conflict["year"]) == int(subject["year"])
            and int(conflict["semester"]) == int(subject["semester"])
        )
        if same_teacher:
            raise ValueError(
                f"Teacher conflict: {conflict['teacher_name']} already has "
                f"{conflict['subject_name']} from {conflict['start_time']} to {conflict['end_time']}."
            )
        if same_section:
            raise ValueError(
                "Section conflict: another teacher is already assigned to this "
                f"section from {conflict['start_time']} to {conflict['end_time']}."
            )


def update_timetable_entry(timetable_id: int, timetable_data: dict[str, Any]) -> None:
    """Update an existing timetable slot after validating conflicts."""
    validate_timetable_slot(
        subject_id=int(timetable_data["subject_id"]),
        teacher_id=int(timetable_data["teacher_id"]),
        section=timetable_data["section"],
        day_of_week=timetable_data["day_of_week"],
        start_time=timetable_data["start_time"],
        end_time=timetable_data["end_time"],
        exclude_timetable_id=timetable_id,
    )
    execute(
        """
        UPDATE timetable
        SET subject_id = ?, teacher_id = ?, section = ?, day_of_week = ?,
            start_time = ?, end_time = ?, room_no = ?
        WHERE id = ?
        """,
        (
            int(timetable_data["subject_id"]),
            int(timetable_data["teacher_id"]),
            timetable_data["section"].upper(),
            timetable_data["day_of_week"],
            timetable_data["start_time"],
            timetable_data["end_time"],
            timetable_data.get("room_no"),
            int(timetable_id),
        ),
    )


def create_substitute_assignment(substitute_data: dict[str, Any]) -> int:
    """Assign a substitute teacher to handle a class on one specific date."""
    assignment = get_assignment_by_id(int(substitute_data["assignment_id"]))
    if not assignment:
        raise ValueError("Selected teacher-subject assignment does not exist.")
    substitute_teacher_id = int(substitute_data["substitute_teacher_id"])
    if substitute_teacher_id == int(assignment["teacher_id"]):
        raise ValueError("Substitute teacher must be different from the original teacher.")
    assignment_date = substitute_data["assignment_date"]
    assignment_day = datetime.strptime(assignment_date, "%Y-%m-%d").strftime("%A")
    target_slots = query_all(
        """
        SELECT day_of_week, start_time, end_time
        FROM timetable
        WHERE subject_id = ? AND section = ? AND day_of_week = ?
        """,
        (int(assignment["subject_id"]), assignment["section"], assignment_day),
    )
    for slot in target_slots:
        regular_conflict = query_one(
            """
            SELECT tt.id
            FROM timetable tt
            WHERE tt.teacher_id = ? AND tt.day_of_week = ?
              AND (? < tt.end_time AND tt.start_time < ?)
            """,
            (
                substitute_teacher_id,
                assignment_day,
                slot["start_time"],
                slot["end_time"],
            ),
        )
        if regular_conflict:
            raise ValueError(
                "Substitute teacher already has a timetable class during this time slot."
            )

        substitute_conflict = query_one(
            """
            SELECT sa.id
            FROM substitute_assignments sa
            JOIN teacher_subjects ts ON ts.id = sa.assignment_id
            JOIN timetable tt
                ON tt.subject_id = ts.subject_id
                AND tt.section = ts.section
            WHERE sa.substitute_teacher_id = ? AND sa.assignment_date = ? AND tt.day_of_week = ?
              AND (? < tt.end_time AND tt.start_time < ?)
            """,
            (
                substitute_teacher_id,
                assignment_date,
                assignment_day,
                slot["start_time"],
                slot["end_time"],
            ),
        )
        if substitute_conflict:
            raise ValueError(
                "Substitute teacher already has another substitute class in this time slot."
            )
    return execute(
        """
        INSERT INTO substitute_assignments (
            assignment_id, original_teacher_id, substitute_teacher_id, assignment_date, note
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(assignment_id, assignment_date) DO UPDATE SET
            substitute_teacher_id = excluded.substitute_teacher_id,
            note = excluded.note
        """,
        (
            int(substitute_data["assignment_id"]),
            int(assignment["teacher_id"]),
            substitute_teacher_id,
            assignment_date,
            substitute_data.get("note"),
        ),
    )


def _normalize_academic_scope(
    course: str,
    branch: str,
    year: int,
    semester: int,
) -> tuple[str, str, int, int]:
    """Normalize the shared academic group fields used across timetable and policy tables."""
    normalized_course = course.strip()
    normalized_branch = branch.strip()
    normalized_year = int(year)
    normalized_semester = int(semester)
    if not normalized_course:
        raise ValueError("Course is required.")
    if not normalized_branch:
        raise ValueError("Branch is required.")
    if normalized_year <= 0:
        raise ValueError("Year must be greater than 0.")
    if normalized_semester <= 0:
        raise ValueError("Semester must be greater than 0.")
    return (
        normalized_course,
        normalized_branch,
        normalized_year,
        normalized_semester,
    )


def _normalize_calendar_day_type(day_type: str) -> str:
    """Return a validated academic calendar day type."""
    normalized_day_type = (day_type or "").strip().lower().replace(" ", "_")
    if normalized_day_type not in ACADEMIC_DAY_TYPES:
        raise ValueError("Day type must be holiday, half_day, regular, or important.")
    return normalized_day_type


def create_semester_timeline(timeline_data: dict[str, Any]) -> int:
    """Create or replace attendance policy metadata for a semester."""
    course, branch, year, semester = _normalize_academic_scope(
        timeline_data["course"],
        timeline_data["branch"],
        int(timeline_data["year"]),
        int(timeline_data["semester"]),
    )
    min_attendance = float(timeline_data["min_attendance"])
    if not 0 <= min_attendance <= 100:
        raise ValueError("Minimum attendance must be between 0 and 100.")
    if timeline_data["end_date"] < timeline_data["start_date"]:
        raise ValueError("Semester end date must be after the start date.")

    return execute(
        """
        INSERT INTO semester_timelines (
            course, branch, year, semester, start_date, end_date, min_attendance
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(course, branch, year, semester) DO UPDATE SET
            start_date = excluded.start_date,
            end_date = excluded.end_date,
            min_attendance = excluded.min_attendance
        """,
        (
            course,
            branch,
            year,
            semester,
            timeline_data["start_date"],
            timeline_data["end_date"],
            min_attendance,
        ),
    )


def create_academic_calendar_file(calendar_data: dict[str, Any]) -> int:
    """Store metadata for an uploaded academic calendar reference file."""
    course, branch, year, semester = _normalize_academic_scope(
        calendar_data["course"],
        calendar_data["branch"],
        int(calendar_data["year"]),
        int(calendar_data["semester"]),
    )
    title = (calendar_data.get("title") or "").strip()
    original_filename = (calendar_data.get("original_filename") or "").strip()
    stored_filename = (calendar_data.get("stored_filename") or "").strip()
    if not title:
        raise ValueError("Calendar title is required.")
    if not original_filename or not stored_filename:
        raise ValueError("Calendar PDF file metadata is incomplete.")
    return execute(
        """
        INSERT INTO academic_calendar_files (
            course, branch, year, semester, title, original_filename, stored_filename
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            course,
            branch,
            year,
            semester,
            title,
            original_filename,
            stored_filename,
        ),
    )


def upsert_academic_calendar_day(day_data: dict[str, Any]) -> int:
    """Create or update a manual academic calendar day marker for an academic group."""
    course, branch, year, semester = _normalize_academic_scope(
        day_data["course"],
        day_data["branch"],
        int(day_data["year"]),
        int(day_data["semester"]),
    )
    calendar_date = (day_data.get("calendar_date") or "").strip()
    day_type = _normalize_calendar_day_type(day_data.get("day_type") or "")
    reason = (day_data.get("reason") or "").strip() or None
    if not calendar_date:
        raise ValueError("Calendar date is required.")
    try:
        datetime.strptime(calendar_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Calendar date must be in YYYY-MM-DD format.") from exc
    return execute(
        """
        INSERT INTO academic_calendar_days (
            course, branch, year, semester, calendar_date, day_type, reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(course, branch, year, semester, calendar_date) DO UPDATE SET
            day_type = excluded.day_type,
            reason = excluded.reason
        """,
        (
            course,
            branch,
            year,
            semester,
            calendar_date,
            day_type,
            reason,
        ),
    )


def get_academic_calendar_file(calendar_file_id: int) -> dict[str, Any] | None:
    """Fetch one uploaded academic calendar file metadata row."""
    return query_one(
        """
        SELECT *
        FROM academic_calendar_files
        WHERE id = ?
        """,
        (int(calendar_file_id),),
    )


def get_academic_calendar_files(
    course: str | None = None,
    branch: str | None = None,
    year: int | None = None,
    semester: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch uploaded academic calendar files, optionally filtered to one academic group."""
    filters: list[str] = []
    params: list[Any] = []
    if course is not None:
        filters.append("course = ?")
        params.append(course.strip())
    if branch is not None:
        filters.append("branch = ?")
        params.append(branch.strip())
    if year is not None:
        filters.append("year = ?")
        params.append(int(year))
    if semester is not None:
        filters.append("semester = ?")
        params.append(int(semester))
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    return query_all(
        f"""
        SELECT *
        FROM academic_calendar_files
        {where_clause}
        ORDER BY uploaded_at DESC, title
        """,
        params,
    )


def get_academic_calendar_days(
    course: str,
    branch: str,
    year: int,
    semester: int,
) -> list[dict[str, Any]]:
    """Fetch all declared academic calendar dates for one academic group."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    return query_all(
        """
        SELECT *
        FROM academic_calendar_days
        WHERE course = ? AND branch = ? AND year = ? AND semester = ?
        ORDER BY calendar_date
        """,
        (
            normalized_course,
            normalized_branch,
            normalized_year,
            normalized_semester,
        ),
    )


def get_academic_calendar_day_status(
    course: str,
    branch: str,
    year: int,
    semester: int,
    calendar_date: str,
) -> dict[str, Any] | None:
    """Return the declared academic status for one date, if it exists."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    return query_one(
        """
        SELECT *
        FROM academic_calendar_days
        WHERE course = ? AND branch = ? AND year = ? AND semester = ? AND calendar_date = ?
        """,
        (
            normalized_course,
            normalized_branch,
            normalized_year,
            normalized_semester,
            calendar_date,
        ),
    )


def get_semester_timeline(
    course: str,
    branch: str,
    year: int,
    semester: int,
) -> dict[str, Any] | None:
    """Fetch one semester timeline record for an academic group."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    return query_one(
        """
        SELECT *
        FROM semester_timelines
        WHERE course = ? AND branch = ? AND year = ? AND semester = ?
        """,
        (
            normalized_course,
            normalized_branch,
            normalized_year,
            normalized_semester,
        ),
    )


def get_all_academic_groups() -> list[dict[str, Any]]:
    """Return all known academic groups available in the system."""
    return query_all(
        """
        SELECT DISTINCT course, branch, year, semester
        FROM (
            SELECT course, branch, year, semester FROM sections
            UNION
            SELECT course, branch, year, semester FROM subjects
            UNION
            SELECT course, branch, year, semester FROM students
            UNION
            SELECT course, branch, year, semester FROM semester_timelines
        )
        ORDER BY course, branch, year, semester
        """
    )


def get_scheduled_weekdays(
    course: str,
    branch: str,
    year: int,
    semester: int,
) -> list[str]:
    """Return weekdays that have timetable slots for an academic group."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    rows = query_all(
        """
        SELECT DISTINCT tt.day_of_week
        FROM timetable tt
        JOIN subjects s ON s.id = tt.subject_id
        WHERE s.course = ? AND s.branch = ? AND s.year = ? AND s.semester = ?
        ORDER BY
            CASE tt.day_of_week
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
                ELSE 8
            END
        """,
        (
            normalized_course,
            normalized_branch,
            normalized_year,
            normalized_semester,
        ),
    )
    return [row["day_of_week"] for row in rows]


def get_attendance_daily_summary(
    course: str,
    branch: str,
    year: int,
    semester: int,
    teacher_id: int | None = None,
    student_erp: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-date attendance totals for one academic group."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    filters = [
        "s.course = ?",
        "s.branch = ?",
        "s.year = ?",
        "s.semester = ?",
    ]
    params: list[Any] = [
        normalized_course,
        normalized_branch,
        normalized_year,
        normalized_semester,
    ]
    if teacher_id is not None:
        filters.append("a.teacher_id = ?")
        params.append(int(teacher_id))
    if student_erp is not None:
        filters.append("a.student_erp = ?")
        params.append(validate_erp_id(student_erp))
    return query_all(
        f"""
        SELECT
            a.attendance_date,
            COUNT(*) AS marked_count,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS present_count,
            SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) AS absent_count,
            SUM(CASE WHEN a.manual_override = 1 THEN 1 ELSE 0 END) AS manual_count
        FROM attendance a
        JOIN subjects s ON s.id = a.subject_id
        WHERE {' AND '.join(filters)}
        GROUP BY a.attendance_date
        ORDER BY a.attendance_date
        """,
        params,
    )


def get_attendance_rows_for_date(
    course: str,
    branch: str,
    year: int,
    semester: int,
    attendance_date: str,
    teacher_id: int | None = None,
    student_erp: str | None = None,
) -> list[dict[str, Any]]:
    """Return attendance rows for one selected calendar date and academic group."""
    normalized_course, normalized_branch, normalized_year, normalized_semester = (
        _normalize_academic_scope(course, branch, year, semester)
    )
    filters = [
        "a.attendance_date = ?",
        "s.course = ?",
        "s.branch = ?",
        "s.year = ?",
        "s.semester = ?",
    ]
    params: list[Any] = [
        attendance_date,
        normalized_course,
        normalized_branch,
        normalized_year,
        normalized_semester,
    ]
    if teacher_id is not None:
        filters.append("a.teacher_id = ?")
        params.append(int(teacher_id))
    if student_erp is not None:
        filters.append("a.student_erp = ?")
        params.append(validate_erp_id(student_erp))
    return query_all(
        f"""
        SELECT
            a.attendance_date,
            a.status,
            a.source,
            a.manual_override,
            a.confidence,
            st.erp_id AS student_erp,
            st.name AS student_name,
            st.roll_no,
            st.section,
            s.subject_code,
            s.subject_name,
            t.name AS teacher_name,
            t.teacher_code
        FROM attendance a
        JOIN students st ON st.erp_id = a.student_erp
        JOIN subjects s ON s.id = a.subject_id
        JOIN teachers t ON t.id = a.teacher_id
        WHERE {' AND '.join(filters)}
        ORDER BY s.subject_code, st.section, st.roll_no, st.name
        """,
        params,
    )


def create_user(
    username: str,
    password_hash: str,
    role: str,
    linked_erp: str | None = None,
    linked_teacher_id: int | None = None,
    email: str | None = None,
    phone: str | None = None,
) -> int:
    """Create a login account tied to a student or teacher entity."""
    normalized_role = role.strip().lower()
    if normalized_role not in {"admin", "hod", "teacher", "student"}:
        raise ValueError("Invalid user role.")
    normalized_username = normalize_username(username)
    normalized_email = validate_email(email) if email else None
    normalized_phone = validate_phone(phone) if phone else None
    normalized_erp = validate_erp_id(linked_erp) if linked_erp else None
    if get_user_by_username(normalized_username):
        raise sqlite3.IntegrityError("Username already exists.")

    return execute(
        """
        INSERT INTO users (
            username, password, role, linked_erp, linked_teacher_id, email, phone
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized_username,
            password_hash,
            normalized_role,
            normalized_erp,
            linked_teacher_id,
            normalized_email,
            normalized_phone,
        ),
    )


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Fetch a login user by username."""
    return query_one(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?)",
        (normalize_username(username),),
    )


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Fetch a login user by primary key."""
    return query_one("SELECT * FROM users WHERE id = ?", (user_id,))


def get_student_by_erp(erp_id: str) -> dict[str, Any] | None:
    """Fetch a student record by ERP identifier."""
    return query_one("SELECT * FROM students WHERE erp_id = ?", (validate_erp_id(erp_id),))


def get_teacher_by_id(teacher_id: int) -> dict[str, Any] | None:
    """Fetch a teacher record by primary key."""
    return query_one("SELECT * FROM teachers WHERE id = ?", (teacher_id,))


def update_student_contact(erp_id: str, email: str | None, phone: str | None) -> None:
    """Update stored contact details for a student."""
    normalized_email = validate_email(email) if email else None
    normalized_phone = validate_phone(phone) if phone else None
    execute(
        """
        UPDATE students
        SET email = ?, phone = ?
        WHERE erp_id = ?
        """,
        (normalized_email, normalized_phone, validate_erp_id(erp_id)),
    )


def update_teacher_contact(teacher_id: int, email: str | None, phone: str | None) -> None:
    """Update stored contact details for a teacher or HOD."""
    normalized_email = validate_email(email) if email else None
    normalized_phone = validate_phone(phone) if phone else None
    execute(
        """
        UPDATE teachers
        SET email = ?, phone = ?
        WHERE id = ?
        """,
        (normalized_email, normalized_phone, teacher_id),
    )


def update_user_contact(user_id: int, email: str | None, phone: str | None) -> None:
    """Update a user's contact details and sync linked role records."""
    normalized_email = validate_email(email) if email else None
    normalized_phone = validate_phone(phone) if phone else None
    execute(
        """
        UPDATE users
        SET email = ?, phone = ?
        WHERE id = ?
        """,
        (normalized_email, normalized_phone, user_id),
    )
    user = get_user_by_id(user_id)
    if not user:
        return
    if user.get("linked_erp"):
        update_student_contact(user["linked_erp"], normalized_email, normalized_phone)
    if user.get("linked_teacher_id"):
        update_teacher_contact(int(user["linked_teacher_id"]), normalized_email, normalized_phone)


def update_user_password(user_id: int, password_hash: str) -> None:
    """Replace a user's password hash."""
    execute(
        """
        UPDATE users
        SET password = ?
        WHERE id = ?
        """,
        (password_hash, user_id),
    )


def mark_student_registered(erp_id: str) -> None:
    """Mark a student as having completed portal signup."""
    execute("UPDATE students SET registered = 1 WHERE erp_id = ?", (erp_id,))


def create_password_reset_otp(
    user_id: int,
    otp_hash: str,
    contact_method: str,
    destination: str,
    expires_at: str,
) -> None:
    """Store a single active OTP for password reset and invalidate older ones."""
    execute(
        """
        UPDATE password_reset_otps
        SET used_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND used_at IS NULL
        """,
        (user_id,),
    )
    execute(
        """
        INSERT INTO password_reset_otps (
            user_id, otp_hash, contact_method, destination, expires_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, otp_hash, contact_method, destination, expires_at),
    )


def get_active_password_reset(user_id: int) -> dict[str, Any] | None:
    """Return the most recent active password reset OTP for a user."""
    return query_one(
        """
        SELECT *
        FROM password_reset_otps
        WHERE user_id = ? AND used_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    )


def consume_password_reset_otp(reset_id: int) -> None:
    """Mark a password-reset OTP as used."""
    execute(
        """
        UPDATE password_reset_otps
        SET used_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (reset_id,),
    )


def clear_face_embeddings(student_erp: str) -> None:
    """Delete old embeddings so a student can re-enroll with fresh samples."""
    execute("DELETE FROM face_embeddings WHERE student_erp = ?", (validate_erp_id(student_erp),))


def save_face_embedding(student_erp: str, embedding: list[float], sample_path: str) -> None:
    """Persist a single 128D face embedding sample for a student."""
    if len(embedding) != 128:
        raise ValueError("Face embedding must contain exactly 128 values.")
    if any(value is None for value in embedding):
        raise ValueError("Face embedding contains invalid values.")
    encrypted_embedding = _get_embedding_manager().encrypt_embedding(embedding)

    execute(
        """
        INSERT OR REPLACE INTO face_embeddings (student_erp, embedding_json, sample_path)
        VALUES (?, ?, ?)
        """,
        (validate_erp_id(student_erp), encrypted_embedding, str(sample_path)),
    )


def get_face_embeddings(student_erps: list[str] | None = None) -> list[dict[str, Any]]:
    """Return stored face embeddings, optionally filtered to a roster."""
    if not student_erps:
        return _hydrate_face_embedding_rows(
            query_all(
                """
                SELECT f.*, s.name
                FROM face_embeddings f
                JOIN students s ON s.erp_id = f.student_erp
                """
            )
        )
    placeholders = ", ".join("?" for _ in student_erps)
    return _hydrate_face_embedding_rows(
        query_all(
            f"""
            SELECT f.*, s.name
            FROM face_embeddings f
            JOIN students s ON s.erp_id = f.student_erp
            WHERE f.student_erp IN ({placeholders})
            """,
            student_erps,
        )
    )


def get_section_roster(
    course: str,
    branch: str,
    year: int,
    semester: int,
    section: str,
) -> list[dict[str, Any]]:
    """Fetch all students belonging to a given academic section."""
    return query_all(
        """
        SELECT *
        FROM students
        WHERE course = ? AND branch = ? AND year = ? AND semester = ? AND section = ?
        ORDER BY roll_no
        """,
        (course, branch, int(year), int(semester), section.upper()),
    )


# ===== ML Model Management =====

_ml_model = None


def load_ml_model():
    """Load ML model from disk at startup."""
    global _ml_model
    try:
        from ml.risk_model import AttendanceRiskPredictor

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        predictor = AttendanceRiskPredictor()
        model_loaded = False
        if MODEL_FILE.exists():
            try:
                predictor.load_model(MODEL_FILE)
                model_loaded = True
            except Exception as exc:
                print(f"Existing ML model reload failed: {exc}")
        if not model_loaded:
            predictor.train_default()
            predictor.save_model(MODEL_FILE)
        _ml_model = predictor
    except Exception as exc:
        print(f"Failed to load ML model: {exc}")
        _ml_model = None
    return _ml_model


def get_ml_model():
    """Get the loaded ML model instance."""
    global _ml_model
    if _ml_model is None:
        load_ml_model()
    return _ml_model


def predict_student_risk(student_erp: str) -> dict[str, Any] | None:
    """Predict if a student is at risk of failing based on their metrics."""
    student = get_student_by_erp(student_erp)
    if not student:
        return None

    attendance_stats = query_one(
        """
        SELECT
            COUNT(*) AS total_classes,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) AS attended_classes
        FROM attendance
        WHERE student_erp = ?
        """,
        (student["erp_id"],),
    )

    total_classes = int((attendance_stats or {}).get("total_classes") or 0)
    attended_classes = int((attendance_stats or {}).get("attended_classes") or 0)
    if total_classes == 0:
        return {
            "at_risk": False,
            "risk_score": 0.0,
            "confidence": 0.0,
            "label": "Unknown",
            "message": "Risk score will appear after attendance data is recorded.",
            "features": {
                "attendance_percentage": 0.0,
                "cgpa": float(student.get("cgpa") or 0.0),
                "assignments_percentage": float(student.get("assignments_percentage") or 0.0),
            },
        }

    attendance_percentage = round((attended_classes / total_classes) * 100, 2)
    feature_data = {
        "attendance_percentage": attendance_percentage,
        "cgpa": float(student.get("cgpa") or 7.0),
        "assignments_percentage": float(student.get("assignments_percentage") or 80.0),
    }

    model = get_ml_model()
    if model is not None:
        try:
            prediction = model.predict_risk(feature_data)
            return prediction
        except Exception as exc:
            print(f"Prediction error for {student_erp}: {exc}")
            model = load_ml_model()
            if model is not None:
                try:
                    return model.predict_risk(feature_data)
                except Exception as retry_exc:
                    print(f"Prediction retry error for {student_erp}: {retry_exc}")

    return {
        "at_risk": False,
        "risk_score": 0.0,
        "confidence": 0.0,
        "label": "Unknown",
        "message": "ML risk prediction is temporarily unavailable.",
        "features": feature_data,
    }


def get_assignment_by_id(assignment_id: int) -> dict[str, Any] | None:
    """Fetch a teacher-subject assignment with joined subject metadata."""
    return query_one(
        """
        SELECT
            ts.id,
            ts.section,
            ts.teacher_id,
            ts.subject_id,
            t.name AS teacher_name,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester
        FROM teacher_subjects ts
        JOIN teachers t ON t.id = ts.teacher_id
        JOIN subjects s ON s.id = ts.subject_id
        WHERE ts.id = ?
        """,
        (assignment_id,),
    )


def get_teacher_assignments(teacher_id: int) -> list[dict[str, Any]]:
    """Return all subject assignments currently mapped to a teacher."""
    return query_all(
        """
        SELECT
            ts.id,
            ts.section,
            s.id AS subject_id,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester,
            'regular' AS assignment_source,
            NULL AS assignment_date,
            NULL AS original_teacher_name
        FROM teacher_subjects ts
        JOIN subjects s ON s.id = ts.subject_id
        WHERE ts.teacher_id = ?
        ORDER BY s.subject_name, ts.section
        """,
        (teacher_id,),
    )


def get_teacher_substitute_assignments(
    teacher_id: int,
    assignment_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return one-day substitute classes assigned to a teacher."""
    params: list[Any] = [teacher_id]
    date_filter = ""
    if assignment_date:
        date_filter = "AND sa.assignment_date = ?"
        params.append(assignment_date)
    return query_all(
        f"""
        SELECT
            ts.id,
            ts.section,
            s.id AS subject_id,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester,
            'substitute' AS assignment_source,
            sa.assignment_date,
            original_teacher.name AS original_teacher_name
        FROM substitute_assignments sa
        JOIN teacher_subjects ts ON ts.id = sa.assignment_id
        JOIN subjects s ON s.id = ts.subject_id
        JOIN teachers original_teacher ON original_teacher.id = sa.original_teacher_id
        WHERE sa.substitute_teacher_id = ?
        {date_filter}
        ORDER BY sa.assignment_date DESC, s.subject_name
        """,
        params,
    )


def get_teacher_effective_assignments(
    teacher_id: int,
    assignment_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return direct assignments plus any substitute assignments for the given date."""
    assignments = get_teacher_assignments(teacher_id)
    substitute_assignments = get_teacher_substitute_assignments(
        teacher_id, assignment_date=assignment_date
    )
    existing_ids = {assignment["id"] for assignment in assignments}
    for assignment in substitute_assignments:
        if assignment["id"] in existing_ids and not assignment_date:
            continue
        assignments.append(assignment)
    return sorted(
        assignments,
        key=lambda item: (
            item.get("assignment_date") or "",
            item["subject_name"],
            item["section"],
        ),
    )


def teacher_can_take_attendance(
    teacher_id: int,
    assignment_id: int,
    attendance_date: str,
) -> bool:
    """Return whether a teacher can mark attendance for an assignment on a date."""
    assignment = get_assignment_by_id(assignment_id)
    if not assignment:
        return False
    substitute = query_one(
        """
        SELECT substitute_teacher_id
        FROM substitute_assignments
        WHERE assignment_id = ? AND assignment_date = ?
        """,
        (assignment_id, attendance_date),
    )
    if substitute:
        return int(substitute["substitute_teacher_id"]) == int(teacher_id)
    return int(assignment["teacher_id"]) == int(teacher_id)


def get_teacher_timetable(teacher_id: int) -> list[dict[str, Any]]:
    """Return timetable rows for a teacher with subject and assignment context."""
    regular_slots = query_all(
        """
        SELECT
            tt.id AS timetable_id,
            tt.teacher_id,
            tt.subject_id,
            tt.section,
            tt.day_of_week,
            tt.start_time,
            tt.end_time,
            tt.room_no,
            ts.id AS assignment_id,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester
        FROM timetable tt
        JOIN subjects s ON s.id = tt.subject_id
        LEFT JOIN teacher_subjects ts
            ON ts.teacher_id = tt.teacher_id
            AND ts.subject_id = tt.subject_id
            AND ts.section = tt.section
        WHERE tt.teacher_id = ?
        ORDER BY
            CASE tt.day_of_week
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
                ELSE 8
            END,
            tt.start_time
        """,
        (teacher_id,),
    )
    substitute_slots = query_all(
        """
        SELECT
            tt.id AS timetable_id,
            sa.substitute_teacher_id AS teacher_id,
            tt.subject_id,
            tt.section,
            tt.day_of_week,
            tt.start_time,
            tt.end_time,
            tt.room_no,
            ts.id AS assignment_id,
            s.subject_name,
            s.subject_code,
            s.course,
            s.branch,
            s.year,
            s.semester,
            sa.assignment_date,
            original_teacher.name AS original_teacher_name,
            1 AS is_substitute
        FROM substitute_assignments sa
        JOIN teacher_subjects ts ON ts.id = sa.assignment_id
        JOIN subjects s ON s.id = ts.subject_id
        JOIN timetable tt
            ON tt.subject_id = ts.subject_id
            AND tt.section = ts.section
        JOIN teachers original_teacher ON original_teacher.id = sa.original_teacher_id
        WHERE sa.substitute_teacher_id = ?
        """,
        (teacher_id,),
    )
    for slot in regular_slots:
        slot["assignment_date"] = None
        slot["original_teacher_name"] = None
        slot["is_substitute"] = 0
    return regular_slots + substitute_slots


def get_teacher_live_schedule(teacher_id: int, current_dt: datetime | None = None) -> dict[str, Any]:
    """Return today's timetable plus active and upcoming slots for the teacher."""
    current_dt = current_dt or datetime.now()
    current_day = current_dt.strftime("%A")
    current_date = current_dt.strftime("%Y-%m-%d")
    current_time = current_dt.strftime("%H:%M")
    overridden_assignment_ids = {
        row["assignment_id"]
        for row in query_all(
            """
            SELECT assignment_id
            FROM substitute_assignments
            WHERE original_teacher_id = ? AND assignment_date = ?
            """,
            (teacher_id, current_date),
        )
    }
    today_slots = [
        slot
        for slot in get_teacher_timetable(teacher_id)
        if slot["day_of_week"] == current_day
        and (not slot.get("assignment_date") or slot["assignment_date"] == current_date)
        and not (
            not slot.get("is_substitute")
            and slot.get("assignment_id") in overridden_assignment_ids
        )
    ]
    filtered_today_slots: list[dict[str, Any]] = []
    excluded_groups: set[str] = set()
    for slot in today_slots:
        excluded_day = get_academic_calendar_day_status(
            slot["course"],
            slot["branch"],
            int(slot["year"]),
            int(slot["semester"]),
            current_date,
        )
        if excluded_day and excluded_day["day_type"] in ACADEMIC_EXCLUDED_DAY_TYPES:
            excluded_groups.add(
                (
                    f"{slot['course']} | {slot['branch']} | Year {slot['year']} | "
                    f"Semester {slot['semester']} ({excluded_day['day_type'].replace('_', ' ')})"
                )
            )
            continue
        filtered_today_slots.append(slot)
    today_slots = filtered_today_slots

    active_slots: list[dict[str, Any]] = []
    upcoming_slots: list[dict[str, Any]] = []
    for slot in today_slots:
        slot["is_active"] = slot["start_time"] <= current_time <= slot["end_time"]
        slot["is_upcoming"] = current_time < slot["start_time"]
        if slot["is_active"]:
            active_slots.append(slot)
        elif slot["is_upcoming"]:
            upcoming_slots.append(slot)

    return {
        "current_day": current_day,
        "current_time": current_time,
        "today_slots": today_slots,
        "active_slots": active_slots,
        "upcoming_slots": upcoming_slots,
        "calendar_notice": (
            "Academic calendar exclusion active today for: "
            + ", ".join(sorted(excluded_groups))
            if excluded_groups
            else None
        ),
    }


def upsert_attendance(
    student_erp: str,
    subject_id: int,
    teacher_id: int,
    attendance_date: str,
    status: str,
    confidence: float | None,
    source: str,
    manual_override: int = 0,
) -> None:
    """Insert or update a student attendance row for a specific class date."""
    execute(
        """
        INSERT INTO attendance (
            student_erp, subject_id, teacher_id, attendance_date, status,
            confidence, source, manual_override
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_erp, subject_id, attendance_date) DO UPDATE SET
            teacher_id = excluded.teacher_id,
            status = excluded.status,
            confidence = excluded.confidence,
            source = excluded.source,
            manual_override = excluded.manual_override
        """,
        (
            student_erp,
            int(subject_id),
            int(teacher_id),
            attendance_date,
            status,
            confidence,
            source,
            int(manual_override),
        ),
    )


def create_alert(student_erp: str, alert_type: str, message: str) -> bool:
    """Create a dated alert while suppressing identical duplicates on the same day."""
    duplicate = query_one(
        """
        SELECT id
        FROM alerts
        WHERE student_erp = ? AND alert_type = ? AND message = ?
          AND DATE(created_at) = DATE('now', 'localtime')
        """,
        (student_erp, alert_type, message),
    )
    if duplicate:
        return False
    execute(
        """
        INSERT INTO alerts (student_erp, alert_type, message)
        VALUES (?, ?, ?)
        """,
        (student_erp, alert_type, message),
    )
    return True


def get_recent_alerts(student_erp: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent alerts globally or for a single student."""
    if student_erp:
        return query_all(
            """
            SELECT *
            FROM alerts
            WHERE student_erp = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (student_erp, limit),
        )
    return query_all(
        """
        SELECT *
        FROM alerts
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def get_dashboard_snapshot() -> dict[str, list[dict[str, Any]]]:
    """Return the datasets required to render the admin dashboard."""
    return {
        "students": query_all("SELECT * FROM students ORDER BY created_at DESC"),
        "teachers": query_all("SELECT * FROM teachers ORDER BY created_at DESC"),
        "subjects": query_all("SELECT * FROM subjects ORDER BY created_at DESC"),
        "sections": query_all(
            "SELECT * FROM sections ORDER BY course, branch, year, semester, section"
        ),
        "assignments": query_all(
            """
            SELECT
                ts.id,
                t.name AS teacher_name,
                t.teacher_code,
                s.subject_name,
                s.subject_code,
                ts.section
            FROM teacher_subjects ts
            JOIN teachers t ON t.id = ts.teacher_id
            JOIN subjects s ON s.id = ts.subject_id
            ORDER BY s.subject_name, ts.section
            """
        ),
        "timetable": query_all(
            """
            SELECT
                tt.id,
                tt.subject_id,
                tt.teacher_id,
                s.subject_name,
                s.subject_code,
                t.name AS teacher_name,
                tt.section,
                tt.day_of_week,
                tt.start_time,
                tt.end_time,
                tt.room_no
            FROM timetable tt
            JOIN subjects s ON s.id = tt.subject_id
            JOIN teachers t ON t.id = tt.teacher_id
            ORDER BY tt.day_of_week, tt.start_time
            """
        ),
        "timelines": query_all(
            """
            SELECT *
            FROM semester_timelines
            ORDER BY course, branch, year, semester
            """
        ),
        "academic_calendars": query_all(
            """
            SELECT *
            FROM academic_calendar_files
            ORDER BY uploaded_at DESC, title
            """
        ),
        "academic_calendar_days": query_all(
            """
            SELECT *
            FROM academic_calendar_days
            ORDER BY calendar_date DESC, course, branch, year, semester
            """
        ),
        "substitutes": query_all(
            """
            SELECT
                sa.id,
                sa.assignment_date,
                sa.note,
                s.subject_name,
                s.subject_code,
                ts.section,
                original_teacher.name AS original_teacher_name,
                substitute_teacher.name AS substitute_teacher_name
            FROM substitute_assignments sa
            JOIN teacher_subjects ts ON ts.id = sa.assignment_id
            JOIN subjects s ON s.id = ts.subject_id
            JOIN teachers original_teacher ON original_teacher.id = sa.original_teacher_id
            JOIN teachers substitute_teacher ON substitute_teacher.id = sa.substitute_teacher_id
            ORDER BY sa.assignment_date DESC, s.subject_name
            """
        ),
    }
