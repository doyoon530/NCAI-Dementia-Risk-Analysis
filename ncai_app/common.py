import re


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[ㅋㅎㅠㅜ]{4,}", "", normalized)
    return normalized.strip()


def validate_user_text(text: str) -> bool:
    return bool(text and len(text.strip()) >= 2)


def clamp_score(score: int) -> int:
    return max(0, min(score, 100))


def clamp_subscore(score: int, max_value: int) -> int:
    return max(0, min(score, max_value))
