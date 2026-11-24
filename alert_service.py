"""Alert services for notifications (Email, SMS)."""

from __future__ import annotations

from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from pathlib import Path
import re
import smtplib
from string import Formatter
from typing import Any

from dotenv import load_dotenv

try:
    from twilio.rest import Client

    TWILIO_AVAILABLE = True
except ImportError:
    Client = None
    TWILIO_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
for env_file in (BASE_DIR / ".env", BASE_DIR / "twilio.env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

LOW_ATTENDANCE_TEMPLATE = (
    "Dear {student_name}, your attendance in {subject_name} ({subject_code}) is "
    "{attendance_percentage:.1f}% which is below the required {threshold:.1f}%. "
    "Attend at least {required_classes} more classes to improve. "
    "- {teacher_name}"
)
LOW_ATTENDANCE_PLACEHOLDERS = {
    "student_name",
    "student_erp",
    "attendance_percentage",
    "threshold",
    "required_classes",
    "teacher_name",
    "subject_name",
    "subject_code",
    "attendance_date",
    "total_classes",
    "present_classes",
}


class _SafeTemplateDict(dict):
    """Provide empty text for any optional placeholders not supplied."""

    def __missing__(self, key: str) -> str:
        return ""


class AlertService:
    """Send attendance and risk alerts over SMS or email."""

    def __init__(self) -> None:
        """Initialize configured outbound channels."""
        self.twilio_available = False
        self.email_available = False
        self.alerts_sent: list[dict[str, Any]] = []
        self.twilio_client = None
        self.twilio_phone: str | None = None

        twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")

        if TWILIO_AVAILABLE and twilio_sid and twilio_token and self.twilio_phone:
            try:
                self.twilio_client = Client(twilio_sid, twilio_token)
                self.twilio_available = True
                print("Twilio SMS service initialized.")
            except Exception as exc:
                print(f"Twilio initialization failed: {exc}")
        elif not TWILIO_AVAILABLE:
            print("Twilio not installed; SMS alerts are disabled.")

        self.email_sender = os.environ.get("EMAIL_SENDER")
        self.email_password = os.environ.get("EMAIL_PASSWORD")
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))

        if self.email_sender and self.email_password:
            self.email_available = True
            print("Email service initialized.")

    def _record_alert(self, alert_type: str, recipient: str, **extra: Any) -> None:
        """Store an in-memory audit trail of sent alerts."""
        payload = {
            "type": alert_type,
            "recipient": recipient,
            "timestamp": datetime.now(),
        }
        payload.update(extra)
        self.alerts_sent.append(payload)

    def get_sms_status(self) -> dict[str, Any]:
        """Expose a safe snapshot of current SMS configuration for the UI."""
        sender = None
        if self.twilio_phone:
            sender = (
                f"{self.twilio_phone[:5]}"
                f"{'*' * max(len(self.twilio_phone) - 7, 3)}"
                f"{self.twilio_phone[-2:]}"
            )
        return {
            "configured": bool(self.twilio_available and self.twilio_client),
            "sender": sender,
            "trial_notice": "Twilio trial account par sirf verified numbers par SMS jayega.",
        }

    def _normalize_template(self, template: str) -> str:
        """Validate the custom SMS template and keep it compact for trial usage."""
        normalized = re.sub(r"\s+", " ", (template or "").strip())
        if not normalized:
            raise ValueError("Custom SMS message blank nahi ho sakta.")
        if len(normalized) > 320:
            raise ValueError("Custom SMS message 320 characters se chhota rakhein.")

        used_placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(normalized)
            if field_name
        }
        unsupported = sorted(used_placeholders - LOW_ATTENDANCE_PLACEHOLDERS)
        if unsupported:
            placeholders = ", ".join(f"{{{name}}}" for name in unsupported)
            raise ValueError(
                f"Unsupported SMS placeholders: {placeholders}. "
                "Allowed placeholders ko hi use karein."
            )
        return normalized

    def build_low_attendance_message(
        self,
        student_name: str,
        attendance_percentage: float,
        threshold: float = 75,
        required_classes: int = 0,
        custom_message_template: str | None = None,
        message_context: dict[str, Any] | None = None,
    ) -> str:
        """Create a default or teacher-customized low-attendance SMS body."""
        context: dict[str, Any] = {
            "student_name": student_name,
            "student_erp": "",
            "attendance_percentage": float(attendance_percentage),
            "threshold": float(threshold),
            "required_classes": int(required_classes or 0),
            "teacher_name": "Teacher",
            "subject_name": "your subject",
            "subject_code": "N/A",
            "attendance_date": datetime.now().strftime("%Y-%m-%d"),
            "total_classes": 0,
            "present_classes": 0,
        }
        if message_context:
            context.update(
                {
                    key: value
                    for key, value in message_context.items()
                    if key in LOW_ATTENDANCE_PLACEHOLDERS
                }
            )

        template = LOW_ATTENDANCE_TEMPLATE
        if custom_message_template:
            template = self._normalize_template(custom_message_template)

        message_text = re.sub(r"\s+", " ", template.format_map(_SafeTemplateDict(context))).strip()
        if len(message_text) > 320:
            raise ValueError(
                "Rendered SMS message 320 characters se lamba ho gaya. "
                "Custom text thoda chhota karein."
            )
        return message_text

    def send_low_attendance_alert(
        self,
        phone_number: str,
        student_name: str,
        attendance_percentage: float,
        threshold: float = 75,
        required_classes: int = 0,
        custom_message_template: str | None = None,
        message_context: dict[str, Any] | None = None,
    ) -> bool:
        """Send SMS alert for low attendance."""
        if not self.twilio_available or not self.twilio_client:
            print("Twilio SMS is not configured.")
            return False

        message_text = self.build_low_attendance_message(
            student_name=student_name,
            attendance_percentage=attendance_percentage,
            threshold=threshold,
            required_classes=required_classes,
            custom_message_template=custom_message_template,
            message_context=message_context,
        )

        try:
            msg = self.twilio_client.messages.create(
                body=message_text,
                from_=self.twilio_phone,
                to=phone_number,
            )
            self._record_alert(
                "SMS_ATTENDANCE",
                phone_number,
                message_id=msg.sid,
                message=message_text,
            )
            print(f"SMS sent to {phone_number}")
            return True
        except Exception as exc:
            print(f"SMS send error: {exc}")
            return False

    def send_at_risk_alert(
        self,
        phone_number: str,
        student_name: str,
        risk_factors: list[str],
        risk_score: float,
    ) -> bool:
        """Send SMS alert for at-risk status."""
        if not self.twilio_available or not self.twilio_client:
            print("Twilio SMS is not configured.")
            return False

        factors_text = ", ".join(risk_factors[:2]) or "No specific factors available"
        message_text = (
            f"Alert: {student_name} is flagged as at risk "
            f"(score: {risk_score * 100:.0f}%). "
            f"Factors: {factors_text}. "
            "Action: Contact your instructor or visit student services."
        )

        try:
            msg = self.twilio_client.messages.create(
                body=message_text,
                from_=self.twilio_phone,
                to=phone_number,
            )
            self._record_alert(
                "SMS_AT_RISK",
                phone_number,
                message_id=msg.sid,
            )
            print(f"Risk alert SMS sent to {phone_number}")
            return True
        except Exception as exc:
            print(f"SMS send error: {exc}")
            return False

    def send_email_alert(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        plain_text: str | None = None,
    ) -> bool:
        """Send an email alert."""
        if not self.email_available:
            print("Email service is not configured.")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_sender or ""
            msg["To"] = recipient_email

            if plain_text:
                msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)

            self._record_alert("EMAIL", recipient_email, subject=subject)
            print(f"Email sent to {recipient_email}")
            return True
        except Exception as exc:
            print(f"Email send error: {exc}")
            return False

    def send_attendance_report_email(
        self,
        recipient_email: str,
        student_name: str,
        attendance_percentage: float,
        status: str,
        recommendations: list[str],
    ) -> bool:
        """Send a detailed attendance report email."""
        status_color = "#28a745" if status == "safe" else "#dc3545"
        status_text = "SAFE" if status == "safe" else "AT RISK"

        html_body = f"""
        <html>
            <body>
                <h2>Attendance Report for {student_name}</h2>
                <div style="background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <h3>Status: {status_text}</h3>
                    <p>Current Attendance: {attendance_percentage:.1f}%</p>
                </div>
                <h4>Recommendations:</h4>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
                <p>For more details, visit: https://attendance.college.edu</p>
                <p>Contact: support@college.edu</p>
            </body>
        </html>
        """

        return self.send_email_alert(
            recipient_email,
            f"Attendance Report for {student_name}",
            html_body,
            plain_text=(
                f"Attendance Report: {student_name} - "
                f"{status_text} ({attendance_percentage:.1f}%)"
            ),
        )

    def send_bulk_alerts(self, alerts: list[dict[str, Any]]) -> dict[str, int]:
        """Send multiple alerts and return a summary."""
        results = {
            "sms_success": 0,
            "sms_failed": 0,
            "email_success": 0,
            "email_failed": 0,
        }

        for alert in alerts:
            alert_type = alert.get("type")
            data = alert.get("data", {})

            if alert_type == "sms_attendance":
                if self.send_low_attendance_alert(**data):
                    results["sms_success"] += 1
                else:
                    results["sms_failed"] += 1
            elif alert_type == "sms_at_risk":
                if self.send_at_risk_alert(**data):
                    results["sms_success"] += 1
                else:
                    results["sms_failed"] += 1
            elif alert_type == "email":
                if self.send_email_alert(**data):
                    results["email_success"] += 1
                else:
                    results["email_failed"] += 1

        return results

    def get_alert_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the latest in-memory alert history."""
        return self.alerts_sent[-limit:]


alert_service = AlertService()


"""
# Example usage
results = alert_service.send_bulk_alerts(
    [
        {"type": "sms_attendance", "data": {}},
        {"type": "email", "data": {}},
    ]
)
print(results)
"""
