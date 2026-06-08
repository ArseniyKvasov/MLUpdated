from __future__ import annotations

from typing import Any


def ensure_text(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is not None:
        try:
            items = str(value).strip()
            if items:
                return items
        except Exception:
            pass
    return fallback


def coerce_score(value: Any) -> int:
    try:
        return 1 if int(value) else 0
    except (TypeError, ValueError):
        return 0


def normalize_question_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    normalized = stripped.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "риторический": "rhetorical",
        "check": "checking_understanding",
        "check_understanding": "checking_understanding",
        "checkingunderstanding": "checking_understanding",
        "checking_understand": "checking_understanding",
        "checking_understanding": "checking_understanding",
        "checking_questions": "checking_understanding",
        "проверка_понимания": "checking_understanding",
        "проверкапонимания": "checking_understanding",
        "проверка_класса": "checking_understanding",
        "вопрос_на_понимание": "checking_understanding",
        "quiz": "quiz",
        "test": "quiz",
        "test_question": "quiz",
        "test_questions": "quiz",
        "quiz_question": "quiz",
        "тест": "quiz",
        "викторина": "quiz",
        "проверочный_вопрос": "quiz",
        "rhetorical_question": "rhetorical",
        "риторический_вопрос": "rhetorical",
        "openended": "open_ended",
        "open_question": "open_ended",
        "открытый": "open_ended",
        "открытый_вопрос": "open_ended",
        "открытый_ответ": "open_ended",
        "fact": "factual",
        "factual_question": "factual",
        "факт": "factual",
        "фактический": "factual",
        "фактический_вопрос": "factual",
        "clarify": "clarifying",
        "уточняющий": "clarifying",
        "уточняющий_вопрос": "clarifying",
        "другое": "other",
        "другой": "other",
    }
    return aliases.get(normalized, stripped)


def normalize_example_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "пример": "example",
        "аналогия": "analogy",
        "метафора": "metaphor",
        "история": "storytelling",
        "рассказ": "storytelling",
        "story": "storytelling",
        "storytelling": "storytelling",
        "повествование": "storytelling",
        "illustration": "example",
        "example": "example",
        "analogy": "analogy",
        "metaphor": "metaphor",
    }
    return aliases.get(normalized, normalized)
