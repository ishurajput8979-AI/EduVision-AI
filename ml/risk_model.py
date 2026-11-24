"""RandomForest-based student risk model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class AttendanceRiskPredictor:
    """Predict at-risk students from attendance, CGPA, and assignments."""

    def __init__(self) -> None:
        self.feature_names = [
            "attendance_percentage",
            "cgpa",
            "assignments_percentage",
        ]
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "2.0"

    def prepare_features(self, student_data: dict[str, Any]) -> np.ndarray:
        """Shape feature data for model prediction."""
        features = np.array(
            [
                float(student_data.get("attendance_percentage", 0.0) or 0.0),
                float(student_data.get("cgpa", 0.0) or 0.0),
                float(student_data.get("assignments_percentage", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )
        return features.reshape(1, -1)

    def train_default(self, sample_count: int = 1200) -> None:
        """Train the model on a synthetic but policy-aligned dataset."""
        rng = np.random.default_rng(42)
        attendance = rng.uniform(35, 100, sample_count)
        cgpa = rng.uniform(4.0, 10.0, sample_count)
        assignments = rng.uniform(30, 100, sample_count)
        features = np.column_stack([attendance, cgpa, assignments])

        labels: list[int] = []
        for attendance_value, cgpa_value, assignment_value in features:
            attendance_gap = max(0.0, 75.0 - float(attendance_value)) / 25.0
            cgpa_gap = max(0.0, 6.5 - float(cgpa_value)) / 2.5
            assignment_gap = max(0.0, 75.0 - float(assignment_value)) / 25.0
            weighted_risk = (
                0.55 * attendance_gap
                + 0.25 * cgpa_gap
                + 0.20 * assignment_gap
                + float(rng.normal(0.0, 0.05))
            )
            is_at_risk = (
                weighted_risk >= 0.45
                or attendance_value < 60
                or (attendance_value < 70 and cgpa_value < 6.0)
                or (cgpa_value < 5.5 and assignment_value < 70)
            )
            labels.append(int(is_at_risk))

        matrix = np.array(features, dtype=np.float32)
        targets = np.array(labels, dtype=np.int32)
        scaled = self.scaler.fit_transform(matrix)
        self.model.fit(scaled, targets)
        self.is_trained = True

    def predict_risk(self, student_data: dict[str, Any]) -> dict[str, Any]:
        """Return risk score, flag, and presentation metadata."""
        if not self.is_trained:
            self.train_default()

        features = self.prepare_features(student_data)
        scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(scaled)[0]
        risk_score = float(probabilities[1])
        at_risk = risk_score >= 0.55

        if risk_score >= 0.75:
            label = "High"
        elif risk_score >= 0.45:
            label = "Medium"
        else:
            label = "Low"

        return {
            "risk_score": round(risk_score, 3),
            "at_risk": at_risk,
            "label": label,
            "confidence": round(float(max(probabilities)), 3),
            "features": {
                "attendance_percentage": float(student_data.get("attendance_percentage", 0.0) or 0.0),
                "cgpa": float(student_data.get("cgpa", 0.0) or 0.0),
                "assignments_percentage": float(student_data.get("assignments_percentage", 0.0) or 0.0),
            },
            "message": (
                "Immediate intervention recommended."
                if label == "High"
                else "Student should be monitored closely."
                if label == "Medium"
                else "Student is currently stable."
            ),
        }

    def save_model(self, filepath: Path) -> Path:
        """Persist model and scaler to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as file_handle:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "is_trained": self.is_trained,
                    "model_version": self.model_version,
                },
                file_handle,
            )
        return filepath

    def load_model(self, filepath: Path) -> None:
        """Load model and scaler from disk."""
        with filepath.open("rb") as file_handle:
            payload = pickle.load(file_handle)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.is_trained = payload["is_trained"]
        self.model_version = payload.get("model_version", self.model_version)
