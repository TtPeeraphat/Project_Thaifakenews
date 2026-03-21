"""
validators.py
Production-ready input validation for Fake News Detection system

Fixes:
- Issue 5.1: No input validation
- Issue 6.2: Log injection risk
"""

import re
import html
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st

logger = logging.getLogger(__name__)

TZ_BKK = ZoneInfo("Asia/Bangkok")

RATE_LIMIT_PER_MINUTE = 5
RATE_LIMIT_PER_HOUR   = 30
COOLDOWN_SECONDS      = 3
# =========================================================
# VALIDATION RESULT
# =========================================================

@dataclass
class ValidationResult:
    """Result of validation with detailed error information."""
    is_valid: bool
    error_message: str = ""
    warning_message: str = ""

    def __bool__(self):
        return self.is_valid


# =========================================================
# INPUT VALIDATOR
# =========================================================

class InputValidator:
    """
    Comprehensive input validation for fake news detection
    """

    # -------------------------
    # CONFIG
    # -------------------------

    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 2000     # suitable for BERT
    MAX_WORDS = 600
    MAX_URL_LENGTH = 2048

    THAI_CHAR_MIN = "\u0E00"
    THAI_CHAR_MAX = "\u0E7F"

    # Suspicious domains
    SUSPICIOUS_TLDS = {
        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".xyz",
        ".work",
        ".webcam",
        ".download",
        ".review"
    }

    # -------------------------
    # REGEX (Compiled for speed)
    # -------------------------

    SQL_INJECTION_PATTERNS = [
        re.compile(r"(union|select|insert|update|delete|drop|create)\s+(select|from|into|values|table|database)", re.I),
        re.compile(r"(;|--|\/\*|\*\/)", re.I),
        re.compile(r"xp_", re.I),
        re.compile(r"exec\s*\(", re.I),
    ]

    SUSPICIOUS_PATTERNS = [
        (re.compile(r'(https?://[^\s]+){5,}'), "Too many URLs"),
        (re.compile(r'(.{1,10})\1{10,}'), "High repetition pattern"),
        (re.compile(r'[^\w\s\u0E00-\u0E7F]{30,}'), "Too many special characters"),
        (re.compile(r'(!!!+|\?\?\?+|\.\.\.)\s*(!!!+|\?\?\?+|\.\.\.)+'), "Excessive punctuation"),
    ]

    URL_REGEX = re.compile(r'https?://[^\s]+')

    # =========================================================
    # TEXT VALIDATION
    # =========================================================

    @staticmethod
    def validate_text(
        text: str,
        require_thai: bool = True
    ) -> ValidationResult:

        if not text or not isinstance(text, str):
            return ValidationResult(False, "Invalid text input")

        text = text.strip()

        # Length check
        if len(text) < InputValidator.MIN_TEXT_LENGTH:
            return ValidationResult(
                False,
                f"Text too short ({len(text)} chars)"
            )

        if len(text) > InputValidator.MAX_TEXT_LENGTH:
            return ValidationResult(
                False,
                f"Text too long ({len(text)} chars)"
            )

        # Word count
        word_count = len(text.split())

        if word_count < 3:
            return ValidationResult(
                False,
                "Text must contain at least 3 words"
            )

        if word_count > InputValidator.MAX_WORDS:
            return ValidationResult(
                False,
                f"Too many words ({word_count})"
            )

        warning = ""

        # -------------------------
        # Thai language check
        # -------------------------

        if require_thai:

            thai_chars = sum(
                1 for c in text
                if InputValidator.THAI_CHAR_MIN <= c <= InputValidator.THAI_CHAR_MAX
            )

            thai_ratio = thai_chars / len(text)

            if thai_ratio < 0.2:
                return ValidationResult(
                    False,
                    "Text must contain Thai language"
                )

            if thai_ratio < 0.4:
                warning += "Text contains many non-Thai characters. "

        # -------------------------
        # Character repetition
        # -------------------------

        for char in set(text):

            count = text.count(char)
            ratio = count / len(text)

            if ratio > 0.5 and char not in " \n\t-:,.":
                return ValidationResult(
                    False,
                    f"Excessive repeated character '{char}'"
                )

        # -------------------------
        # Suspicious patterns
        # -------------------------

        for pattern, reason in InputValidator.SUSPICIOUS_PATTERNS:
            if pattern.search(text):
                return ValidationResult(False, f"Suspicious pattern: {reason}")

        # -------------------------
        # SQL Injection
        # -------------------------

        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return ValidationResult(
                    False,
                    "SQL-like pattern detected"
                )

        # -------------------------
        # URL validation
        # -------------------------

        urls = InputValidator.URL_REGEX.findall(text)

        if len(urls) > 3:
            warning += f"Contains {len(urls)} URLs. "

        for url in urls[:5]:

            url_result = InputValidator.validate_url(url)

            if not url_result.is_valid:
                warning += f"Suspicious URL ({url}). "

        return ValidationResult(True, warning_message=warning.strip())

    # =========================================================
    # URL VALIDATION
    # =========================================================

    @staticmethod
    def validate_url(url: str) -> ValidationResult:

        if not url or not isinstance(url, str):
            return ValidationResult(False, "Invalid URL")

        url = url.strip()

        if not url.startswith(("http://", "https://")):
            return ValidationResult(
                False,
                "URL must start with http/https"
            )

        if len(url) > InputValidator.MAX_URL_LENGTH:
            return ValidationResult(False, "URL too long")

        try:
            parsed = urlparse(url)
        except Exception:
            return ValidationResult(False, "Invalid URL format")

        domain = parsed.netloc.lower()

        if not domain:
            return ValidationResult(False, "URL has no domain")

        # suspicious tld
        for tld in InputValidator.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return ValidationResult(
                    False,
                    f"Suspicious domain {tld}"
                )

        # IP address
        if re.match(r"^(\d+\.){3}\d+$", domain):
            return ValidationResult(
                False,
                "IP based URL not allowed"
            )

        phishing_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'0\.0\.0\.0',
            r'xn--'
        ]

        for pattern in phishing_patterns:
            if re.search(pattern, domain, re.I):
                return ValidationResult(False, "Phishing pattern detected")

        return ValidationResult(True)

    # =========================================================
    # LOG SANITIZATION
    # =========================================================

    @staticmethod
    def sanitize_for_logging(value: Any, max_length: int = 200) -> str:

        text = str(value)

        text = html.escape(text, quote=False)

        text = text.replace("\n", " ").replace("\r", " ")

        text = text.replace(";", "-").replace("--", "-").replace("/*", "*")

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    # =========================================================
    # USERNAME
    # =========================================================

    @staticmethod
    def validate_username(username: str) -> ValidationResult:

        if not username or len(username) < 3:
            return ValidationResult(False, "Username too short")

        if len(username) > 32:
            return ValidationResult(False, "Username too long")

        if not re.match(r"^[A-Za-z0-9_]+$", username):
            return ValidationResult(
                False,
                "Only letters numbers underscore allowed"
            )

        return ValidationResult(True)

    # =========================================================
    # EMAIL
    # =========================================================

    @staticmethod
    def validate_email(email: str) -> ValidationResult:

        if not email:
            return ValidationResult(False, "Invalid email")

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(pattern, email):
            return ValidationResult(False, "Invalid email format")

        return ValidationResult(True)

    # =========================================================
    # PASSWORD
    # =========================================================

    @staticmethod
    def validate_password(password: str) -> ValidationResult:

        if not password or len(password) < 8:
            return ValidationResult(False, "Password too short")

        if len(password) > 128:
            return ValidationResult(False, "Password too long")

        common = {"password", "password123", "12345678"}

        if password.lower() in common:
            return ValidationResult(
                False,
                "Password too common"
            )

        warning = ""

        if not re.search(r"[a-z]", password):
            warning += "Missing lowercase. "

        if not re.search(r"[A-Z]", password):
            warning += "Missing uppercase. "

        if not re.search(r"[0-9]", password):
            warning += "Missing number. "

        if not re.search(r"[^a-zA-Z0-9]", password):
            warning += "Missing special char. "

        return ValidationResult(True, warning_message=warning.strip())

def check_rate_limit() -> tuple[bool, str]:
    now = datetime.now(tz=TZ_BKK)
    if "rate_timestamps" not in st.session_state:
        st.session_state["rate_timestamps"] = []
    if "last_predict_time" not in st.session_state:
        st.session_state["last_predict_time"] = None

    last = st.session_state["last_predict_time"]
    if last is not None:
        elapsed = (now - last).total_seconds()
        if elapsed < COOLDOWN_SECONDS:
            wait = COOLDOWN_SECONDS - int(elapsed)
            return False, f"⏳ กรุณารอ {wait} วินาทีก่อนวิเคราะห์ใหม่"

    cutoff_hour   = now - timedelta(hours=1)
    cutoff_minute = now - timedelta(minutes=1)
    timestamps = [t for t in st.session_state["rate_timestamps"] if t > cutoff_hour]
    st.session_state["rate_timestamps"] = timestamps

    if len(timestamps) >= RATE_LIMIT_PER_HOUR:
        return False, f"⚠️ คุณวิเคราะห์ครบ {RATE_LIMIT_PER_HOUR} ครั้งต่อชั่วโมงแล้ว — กรุณารอสักครู่"

    recent = [t for t in timestamps if t > cutoff_minute]
    if len(recent) >= RATE_LIMIT_PER_MINUTE:
        return False, f"⚠️ คุณวิเคราะห์เร็วเกินไป — กรุณารอสักครู่"

    return True, ""


def record_prediction_timestamp():
    now = datetime.now(tz=TZ_BKK)
    if "rate_timestamps" not in st.session_state:
        st.session_state["rate_timestamps"] = []
    st.session_state["rate_timestamps"].append(now)
    st.session_state["last_predict_time"] = now
# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    text = "ข่าวรัฐบาลประกาศนโยบายใหม่เพื่อพัฒนาเศรษฐกิจ"

    result = InputValidator.validate_text(text)

    print("Valid:", result.is_valid)
    print("Error:", result.error_message)
    print("Warning:", result.warning_message)

    url = "https://example.tk/malware"

    result = InputValidator.validate_url(url)

    print("URL Valid:", result.is_valid)

    dangerous = "'; DROP TABLE users; --"

    print(
        "Sanitized:",
        InputValidator.sanitize_for_logging(dangerous)
    )
