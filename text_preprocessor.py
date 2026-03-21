import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ✅ constant อยู่นอก class (ถูกต้อง)
THAI_TOKEN_RATIO: float = 1.8
BERT_MAX_TOKENS:  int   = 256


class TextPreprocessor:
    """Comprehensive text preprocessing for Thai fake news detection."""

    THAI_UNICODE_START = '\u0E00'
    THAI_UNICODE_END   = '\u0E7F'

    @staticmethod
    def html_to_text(html_text: str) -> str:
        html_text = re.sub(r'<[^>]+>', '', html_text)
        html_text = html_text.replace('&nbsp;', ' ')
        html_text = html_text.replace('&amp;', '&')
        html_text = html_text.replace('&lt;', '<')
        html_text = html_text.replace('&gt;', '>')
        html_text = html_text.replace('&quot;', '"')
        html_text = html_text.replace('&#39;', "'")
        return html_text

    @staticmethod
    def remove_urls(text: str) -> str:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '[URL]', text)

    @staticmethod
    def remove_email(text: str) -> str:
        return re.sub(r'\S+@\S+', '[EMAIL]', text)

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        return text.strip()

    @staticmethod
    def remove_emoji(text: str) -> str:
        emoji_pattern = re.compile(
            '['
            '\U0001F600-\U0001F64F'
            '\U0001F300-\U0001F5FF'
            '\U0001F680-\U0001F6FF'
            '\U0001F1E0-\U0001F1FF'
            '\U00002702-\U000027B0'
            '\U000024C2-\U0001F251'
            ']+'
        )
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_newlines(text: str) -> str:
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        return text

    @staticmethod
    def remove_special_chars(text: str) -> str:
        text = re.sub(r'[^\w\s\u0E00-\u0E7F\.,\!\?\;\'\"•\-]', '', text)
        return text

    @staticmethod
    def deduplicate_consecutive_chars(text: str) -> str:
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    @staticmethod
    def remove_repeated_phrases(text: str, max_repetition: int = 3) -> str:
        words = text.split()
        if len(words) < 5:
            return text

        phrase_count: dict[str, int] = {}
        for i in range(len(words) - 4):
            phrase = ' '.join(words[i:i+5])
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1

        if not any(v > max_repetition for v in phrase_count.values()):
            return text

        filtered: list[str] = []
        phrase_seen: dict[str, int] = {}

        for i, word in enumerate(words):
            if i <= len(words) - 5:
                phrase = ' '.join(words[i:i+5])
                count = phrase_count.get(phrase, 1)
                seen  = phrase_seen.get(phrase, 0)
                if count > max_repetition and seen >= max_repetition:
                    continue
                if count > max_repetition:
                    phrase_seen[phrase] = seen + 1
            filtered.append(word)

        return ' '.join(filtered)

    @staticmethod
    def is_mostly_thai(text: str, threshold: float = 0.3) -> bool:
        if not text:
            return False
        thai_chars = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')
        return (thai_chars / len(text)) >= threshold

    @staticmethod
    def detect_spam_patterns(text: str) -> Tuple[bool, str]:
        if re.search(r'(.)\1{10,}', text):
            return True, "Excessive character repetition"
        url_count = len(re.findall(r'https?://', text))
        if url_count > 5:
            return True, f"Too many URLs ({url_count})"
        if len(re.findall(r'[!@#$%^&*]{5,}', text)) > 3:
            return True, "Excessive special characters"
        if len(text) > 20 and text.isupper():
            return True, "All uppercase text"
        return False, ""

    @staticmethod
    def truncate_text(
        text: str,
        max_length: int = 256,
        by_words: bool = True
    ) -> str:
        if by_words:
            words = text.split()
            if len(words) > max_length:
                return ' '.join(words[:max_length])
        else:
            if len(text) > max_length:
                return text[:max_length] + '...'
        return text

    # ✅ อยู่ใน class — indent ถูกต้อง
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """ประมาณจำนวน BERT token จากภาษาไทย"""
        word_count = len(text.split())
        return int(word_count * THAI_TOKEN_RATIO)

    # ✅ อยู่ใน class — indent ถูกต้อง
    @staticmethod
    def preprocess(
        text: str,
        max_length: int = BERT_MAX_TOKENS,
        min_length: int = 10,
        check_thai: bool = True,
        check_spam: bool = True
    ) -> Tuple[str, bool, str]:
        """Full preprocessing pipeline."""

        if not text or not isinstance(text, str):
            return "", False, "Invalid text input"

        # 1. Basic cleanup
        text = TextPreprocessor.html_to_text(text)
        text = TextPreprocessor.remove_newlines(text)
        text = TextPreprocessor.remove_urls(text)
        text = TextPreprocessor.remove_email(text)
        text = TextPreprocessor.remove_emoji(text)
        text = TextPreprocessor.remove_special_chars(text)
        text = TextPreprocessor.remove_extra_spaces(text)
        text = TextPreprocessor.deduplicate_consecutive_chars(text)
        text = TextPreprocessor.remove_repeated_phrases(text)

        # 2. Validation
        if len(text) < min_length:
            return "", False, f"ข้อความสั้นเกินไป ({len(text)} < {min_length})"

        word_count = len(text.split())
        if word_count < 3:
            return "", False, "ข้อความต้องมีอย่างน้อย 3 คำ"

        # ✅ ใช้ estimate_token_count จาก class เดียวกัน
        estimated_tokens = TextPreprocessor.estimate_token_count(text)
        if estimated_tokens > max_length:
            safe_word_limit = int(max_length / THAI_TOKEN_RATIO)
            text = TextPreprocessor.truncate_text(
                text, safe_word_limit, by_words=True
            )

        # 3. Language check
        if check_thai and not TextPreprocessor.is_mostly_thai(text):
            return "", False, "ข้อความต้องเป็นภาษาไทยเป็นส่วนใหญ่"

        # 4. Spam detection
        if check_spam:
            is_spam, reason = TextPreprocessor.detect_spam_patterns(text)
            if is_spam:
                return "", False, f"ตรวจพบ spam: {reason}"

        logger.debug("Preprocessed: %d chars, %d words", len(text), word_count)
        return text, True, ""


# ============================================================
# ทดสอบ (รันโดยตรง)
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        "สวัสดี!!! ข่าว https://example.com มา !!!!!",
        "<p>ข่าวจาก &nbsp; ไทย</p>",
        "!!!!!!!! AAAAAAA ลืมลืมลืมลืม",
    ]
    for raw in test_cases:
        cleaned, valid, msg = TextPreprocessor.preprocess(raw)
        print(f"Input : {raw[:40]}")
        print(f"Output: {cleaned[:40]}")
        print(f"Valid : {valid}  |  Msg: {msg}\n")
