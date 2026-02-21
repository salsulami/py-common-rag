"""Utilities for parsing model output into JSON."""

from __future__ import annotations

import json
from typing import Any

from rag_common.exceptions import ResponseFormatError
from rag_common.types import JSONDict


def parse_json_dict(text: str) -> JSONDict:
    """Parse a JSON object from text that may contain extra wrappers."""
    normalized = _strip_code_fences(text.strip())
    parsed = _try_load_json(normalized)
    if isinstance(parsed, dict):
        return parsed

    candidate = _extract_first_json_object(normalized)
    if candidate:
        parsed = _try_load_json(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ResponseFormatError("Model output did not contain a valid JSON object.")


def _strip_code_fences(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    if text.startswith("```json") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _try_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaping = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None
