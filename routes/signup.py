"""Student signup routes."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, flash, redirect, render_template, request, url_for

from auth_utils import ValidationError, hash_password, validate_erp_id, validate_password_strength
from cache_manager import cache_manager
from database import (
    clear_face_embeddings,
    create_user,
    get_student_by_erp,
    mark_student_registered,
    save_face_embedding,
    update_student_contact,
)
from security_config import limiter
from services.face import (
    build_embeddings_from_samples,
    evaluate_saved_frame_paths,
    extract_frames_from_video,
    save_photo_samples,
    save_video_sample,
)

signup_bp = Blueprint("signup", __name__)


@signup_bp.route("/student/signup", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def student_signup():
    """Register a student account after ERP validation and face enrollment."""
    if request.method == "POST":
        erp_id = request.form.get("erp_id", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip() or None
        phone = request.form.get("phone", "").strip() or None
        photos = [file for file in request.files.getlist("photos") if file and file.filename]
        video = request.files.get("video")

        try:
            erp_id = validate_erp_id(erp_id)
            validate_password_strength(password)
            student = get_student_by_erp(erp_id)
            if not student:
                flash("ERP ID not found. Ask admin to add your record first.", "danger")
                return render_template("signup.html", erp_id=erp_id, email=email, phone=phone)
            if student["registered"]:
                flash("This ERP ID is already registered. Please log in.", "warning")
                return redirect(url_for("auth.login"))
            if len(photos) < 10:
                flash("Upload at least 10 photos for better recognition accuracy.", "danger")
                return render_template("signup.html", erp_id=erp_id, email=email, phone=phone)
            if not video or not video.filename:
                flash("Please upload one enrollment video.", "danger")
                return render_template("signup.html", erp_id=erp_id, email=email, phone=phone)

            image_paths = save_photo_samples(erp_id, photos)
            video_path = save_video_sample(erp_id, video)
            frame_paths = extract_frames_from_video(video_path, video_path.parent / "extracted_frames")
            liveness_result = evaluate_saved_frame_paths(frame_paths, threshold=0.35)
            if not liveness_result.get("is_likely_live"):
                flash("Enrollment video failed liveness. Blink or move slightly and try again.", "danger")
                return render_template("signup.html", erp_id=erp_id, email=email, phone=phone)

            embedding_records = build_embeddings_from_samples(image_paths + frame_paths)
            if len(embedding_records) < 5:
                flash(
                    "Not enough valid face samples were extracted. Use clearer images and video.",
                    "danger",
                )
                return render_template("signup.html", erp_id=erp_id, email=email, phone=phone)

            clear_face_embeddings(erp_id)
            for record in embedding_records:
                save_face_embedding(erp_id, record["embedding"], record["sample_path"])
            update_student_contact(erp_id, email, phone)
            create_user(
                username=erp_id,
                password_hash=hash_password(password),
                role="student",
                linked_erp=erp_id,
                email=email,
                phone=phone,
            )
            mark_student_registered(erp_id)
            cache_manager.invalidate("embedding_index_*")
            flash(
                f"Signup completed with {len(embedding_records)} face embeddings. Please log in.",
                "success",
            )
            return redirect(url_for("auth.login"))
        except (RuntimeError, sqlite3.IntegrityError, ValidationError, ValueError) as exc:
            flash(f"Signup failed: {exc}", "danger")

    return render_template("signup.html", erp_id="", email="", phone="")
