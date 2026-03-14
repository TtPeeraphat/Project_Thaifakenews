# ✅ REFACTORED: Input Validation
# Location: validators.py
# This file fixes Issue 5.1 (No Input Validation)

import re
from dataclasses import dataclass
from typing import Tuple, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation with detailed error information."""
    is_valid: bool
    error_message: str = ""
    warning_message: str = ""
    
    def __bool__(self):
        return self.is_valid


class InputValidator:
    """
    Comprehensive input validation for fake news detection.
    
    ✅ FIXES:
    - Issue 5.1: No input validation
    - Prevents DoS attacks, injection, and invalid input
    """
    
    # Configuration
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 5000
    MAX_WORDS = 800
    MAX_URL_LENGTH = 2048
    
    # Thai Unicode range
    THAI_CHAR_MIN = '\u0E00'
    THAI_CHAR_MAX = '\u0E7F'
    
    # Suspicious domains (high spam rate)
    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf',  # Free hosting, high spam
        '.xyz', '.work', '.webcam',  # High abuse rate
        '.download', '.review'
    }
    
    # SQL injection patterns (for logged content)
    SQL_INJECTION_PATTERNS = [
        r\"(union|select|insert|update|delete|drop|create)\s+(select|from|into|values|table|database)\",
        r\"(;|--|\\/\\*|\\*\\/)\",  # SQL comment syntax
        r\"xp_\",  # SQL Server extended procedures
        r\"exec\\s*\\(\",  # EXEC
    ]
    
    @staticmethod
    def validate_text(
        text: str,
        min_length: int = MIN_TEXT_LENGTH,
        max_length: int = MAX_TEXT_LENGTH,
        max_words: int = MAX_WORDS,
        require_thai: bool = True,
        check_spam: bool = True
    ) -> ValidationResult:
        \"\"\"
        Comprehensive text validation.
        
        ✅ FIXES Issue 5.1
        
        Args:
            text: Input text to validate
            min_length: Minimum character length
            max_length: Maximum character length
            max_words: Maximum word count
            require_thai: Must be mostly Thai
            check_spam: Check for spam patterns
        
        Returns:
            ValidationResult with detailed error/warning info
        \"\"\"
        
        # 1. Type and null check
        if not text or not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                error_message=\"Invalid text input (empty or not string)\"
            )
        
        text = text.strip()
        
        # 2. Length checks
        if len(text) < min_length:
            return ValidationResult(
                is_valid=False,
                error_message=f\"Text too short ({len(text)} chars < {min_length} minimum)\"
            )
        
        if len(text) > max_length:
            return ValidationResult(
                is_valid=False,
                error_message=f\"Text too long ({len(text)} chars > {max_length} maximum)\"
            )
        
        # 3. Word count validation
        word_count = len(text.split())
        if word_count < 3:
            return ValidationResult(
                is_valid=False,
                error_message=\"Text must contain at least 3 words\"
            )
        
        if word_count > max_words:
            return ValidationResult(
                is_valid=False,
                error_message=f\"Too many words ({word_count} > {max_words})\"
            )
        
        warning = \"\"
        
        # 4. Language detection (Thai content)
        if require_thai:
            thai_chars = sum(
                1 for c in text
                if InputValidator.THAI_CHAR_MIN <= c <= InputValidator.THAI_CHAR_MAX
            )
            thai_ratio = thai_chars / len(text) if text else 0
            
            if thai_ratio < 0.2:
                return ValidationResult(
                    is_valid=False,
                    error_message=\"Text must be mostly Thai language (at least 20% Thai characters)\"
                )
            
            if thai_ratio < 0.4:
                warning += \"Warning: Text contains many non-Thai characters. \\n\"
        
        # 5. Character repetition check (spam indicator)
        for char in set(text):
            char_count = text.count(char)
            repetition_ratio = char_count / len(text)
            
            if repetition_ratio > 0.5 and len(char) == 1 and char not in \" \\n\\t\"-:,.
                return ValidationResult(
                    is_valid=False,
                    error_message=f\"Text contains excessive repeated character: '{char}' ({int(repetition_ratio*100)}%)\"
                )
        
        # 6. Suspicious pattern detection
        suspicious_patterns = [
            (r'(https?://[^\\s]+){5,}', \"Too many URLs (more than 5)\"),
            (r'(.{1,10})\\1{10,}', \"Very high character/pattern repetition\"),
            (r'[^\\w\\s\\u0E00-\\u0E7F]{30,}', \"Too many consecutive special characters\"),
            (r'(!!!+|\\?\\?\\?+|\\.\\.\\.)\\s*(!!!+|\\?\\?\\?+|\\.\\.\\.)+', \"Excessive punctuation\"),
        ]
        
        for pattern, reason in suspicious_patterns:
            if re.search(pattern, text):
                return ValidationResult(
                    is_valid=False,
                    error_message=f\"Suspicious pattern detected: {reason}\"
                )
        
        # 7. SQL injection patterns (for safety even though using parameterized queries)
        for sql_pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(sql_pattern, text, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_message=\"Text contains suspicious SQL-like patterns\"
                )
        
        # 8. URL extraction and validation
        urls = re.findall(r'https?://[^\\s]+', text)
        if len(urls) > 3:
            warning += f\"Text contains {len(urls)} URLs. \"
        
        for url in urls[:5]:  # Check first 5 URLs
            url_valid = InputValidator.validate_url(url)
            if not url_valid.is_valid:
                warning += f\"Found suspicious URL: {url_valid.error_message}. \"
        
        return ValidationResult(
            is_valid=True,
            warning_message=warning.strip() if warning else \"\"
        )
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        \"\"\"
        Comprehensive URL validation.
        
        Args:
            url: URL to validate
        
        Returns:
            ValidationResult
        \"\"\"
        
        # 1. Format check
        if not url or not isinstance(url, str):
            return ValidationResult(
                is_valid=False,
                error_message=\"Invalid URL (empty or not string)\"
            )
        
        url = url.strip()
        
        # 2. Scheme check
        if not url.startswith(('http://', 'https://')):
            return ValidationResult(
                is_valid=False,
                error_message=\"URL must start with http:// or https://\"
            )
        
        # 3. Length check
        if len(url) > InputValidator.MAX_URL_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message=f\"URL too long ({len(url)} > {InputValidator.MAX_URL_LENGTH} max)\"
            )
        
        # 4. Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f\"Invalid URL format: {str(e)[:50]}\"
            )
        
        # 5. Check domain
        domain = parsed.netloc.lower()
        if not domain:
            return ValidationResult(
                is_valid=False,
                error_message=\"URL has no domain\"
            )
        
        # 6. Check for suspicious TLDs
        for tld in InputValidator.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return ValidationResult(
                    is_valid=False,
                    error_message=f\"URL from suspicious domain ({tld})\"
                )
        
        # 7. IP address check (could be internal/private)
        ip_pattern = r'^(\\d+\\.){3}\\d+$'
        if re.match(ip_pattern, domain):
            return ValidationResult(
                is_valid=False,
                error_message=\"URL with IP address is not allowed\"
            )
        
        # 8. Check for common phishing patterns
        phishing_patterns = [
            r'amazon\\..*facebook',  # Typosquatting
            r'localhost|127\\.0\\.0\\.1|0\\.0\\.0\\.0',  # Local addresses
            r'xn--',  # IDN homograph attack
        ]
        
        for pattern in phishing_patterns:
            if re.search(pattern, domain, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_message=\"URL matches phishing pattern\"
                )
        
        return ValidationResult(is_valid=True)
    
    @staticmethod
    def sanitize_for_logging(value: any, max_length: int = 200) -> str:
        \"\"\"
        Sanitize value for safe logging (prevents injection in logs).
        
        ✅ FIXES Issue 6.2 (SQL injection risk in logs)
        
        Args:
            value: Value to sanitize
            max_length: Maximum length
        
        Returns:
            Sanitized string safe for logging
        \"\"\"
        import html
        
        str_val = str(value)
        
        # 1. Escape HTML/XML
        str_val = html.escape(str_val, quote=False)
        
        # 2. Remove newlines (prevent log injection)
        str_val = str_val.replace('\\n', ' ').replace('\\r', ' ')
        
        # 3. Remove common injection characters
        str_val = str_val.replace(';', '-').replace('--', '-').replace('/*', '*')
        
        # 4. Truncate
        if len(str_val) > max_length:
            str_val = str_val[:max_length] + \"...\"
        
        return str_val
    
    @staticmethod
    def validate_username(username: str) -> ValidationResult:
        \"\"\"Validate username format.\"\"\"
        
        if not username or len(username) < 3:
            return ValidationResult(
                is_valid=False,
                error_message=\"Username too short (minimum 3 characters)\"
            )
        
        if len(username) > 32:
            return ValidationResult(
                is_valid=False,
                error_message=\"Username too long (maximum 32 characters)\"
            )
        
        # Only alphanumeric and underscore
        if not re.match(r'^[\\w]+$', username):
            return ValidationResult(
                is_valid=False,
                error_message=\"Username can only contain letters, numbers, and underscore\"
            )
        
        return ValidationResult(is_valid=True)
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        \"\"\"Validate email format.\"\"\"
        
        if not email or len(email) < 5:
            return ValidationResult(
                is_valid=False,
                error_message=\"Invalid email address\"
            )
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return ValidationResult(
                is_valid=False,
                error_message=\"Invalid email format\"
            )
        
        return ValidationResult(is_valid=True)
    
    @staticmethod
    def validate_password(password: str) -> ValidationResult:
        \"\"\"Validate password strength.\"\"\"
        
        if not password or len(password) < 8:
            return ValidationResult(
                is_valid=False,
                error_message=\"Password too short (minimum 8 characters)\"
            )
        
        if len(password) > 128:
            return ValidationResult(
                is_valid=False,
                error_message=\"Password too long (maximum 128 characters)\"
            )
        
        # Check for common patterns
        if password.lower() in ['password', 'password123', '12345678']:
            return ValidationResult(
                is_valid=False,
                error_message=\"Password too common, please choose a stronger password\"
            )
        
        warning = \"\"
        
        # Check password strength
        if not re.search(r'[a-z]', password):
            warning += \"Missing lowercase letters. \"
        if not re.search(r'[A-Z]', password):
            warning += \"Missing uppercase letters. \"
        if not re.search(r'[0-9]', password):
            warning += \"Missing numbers. \"
        if not re.search(r'[^a-zA-Z0-9]', password):
            warning += \"Missing special characters. \"
        
        if warning:
            return ValidationResult(
                is_valid=True,
                warning_message=f\"Weak password: {warning.strip()}\"
            )
        
        return ValidationResult(is_valid=True)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == \"__main__\":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Valid Thai text
    result = InputValidator.validate_text(\"เนื้อหาข่าวที่ดีและเชื่อถือได้สำหรับการวิเคราะห์\")
    print(f\"Thai text: Valid={result.is_valid}, Message={result.error_message}\")
    
    # Example 2: Too short
    result = InputValidator.validate_text(\"Hi\")
    print(f\"Short text: Valid={result.is_valid}, Message={result.error_message}\")
    
    # Example 3: Spam pattern
    result = InputValidator.validate_text(\"!!!!!!!! ???????? กกกกกก\" * 100)
    print(f\"Spam text: Valid={result.is_valid}, Message={result.error_message}\")
    
    # Example 4: URL validation
    result = InputValidator.validate_url(\"https://example.tk/malware\")
    print(f\"Suspicious URL: Valid={result.is_valid}, Message={result.error_message}\")
    
    # Example 5: Sanitize for logging
    dangerous = \"'; DROP TABLE users; --\"; 
    safe = InputValidator.sanitize_for_logging(dangerous)
    print(f\"Sanitized: {safe}\")
