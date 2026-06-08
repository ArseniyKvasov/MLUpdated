from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json(content: str) -> Any:
    text = content.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    candidates = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    candidates.extend(balanced_json_candidates(stripped))

    for candidate in candidates:
        parsed = loads_relaxed_json(candidate)
        if parsed is not None:
            return parsed

    return {}


def balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for open_char, close_char in (("{", "}"), ("[", "]")):
        idx = 0
        while idx < len(text):
            start = text.find(open_char, idx)
            if start == -1:
                break
            end = find_matching_bracket(text, start, open_char, close_char)
            if end != -1:
                candidates.append(text[start : end + 1])
                idx = end + 1
            else:
                idx = start + 1
    return candidates


def find_matching_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def loads_relaxed_json(text: str) -> Optional[Any]:
    def escape_newlines_inside_strings(raw: str) -> str:
        result: list[str] = []
        in_string = False
        escaped = False
        for char in raw:
            if in_string:
                if escaped:
                    result.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    result.append(char)
                    escaped = True
                    continue
                if char == '"':
                    result.append(char)
                    in_string = False
                    continue
                if char == "\n":
                    result.append("\\n")
                    continue
                if char == "\r":
                    continue
            else:
                if char == '"':
                    in_string = True
            result.append(char)
        return "".join(result)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        sanitized = escape_newlines_inside_strings(text)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        sanitized = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)
        sanitized = escape_newlines_inside_strings(sanitized)
        sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            return None
