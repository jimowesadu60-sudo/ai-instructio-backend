"""JSON parsing and key validation utilities (string/dict only, no I/O)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class JSONParseError(Exception):
    """Raised when text is not valid JSON object data or schema checks fail."""


_FENCE_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\r?\n?(.*?)```",
    re.DOTALL,
)


def _unwrap_markdown_fence(text: str) -> str:
    """If ``text`` contains a fenced code block, return the first inner body."""
    m = _FENCE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _strip_leading_json_label(text: str) -> str:
    """Remove a lone ``json`` header line sometimes emitted before raw JSON."""
    t = text.strip()
    while True:
        lines = t.splitlines()
        if not lines:
            break
        if lines[0].strip().lower() == "json":
            t = "\n".join(lines[1:]).strip()
            continue
        break
    return t


def _extract_first_json_object(text: str) -> str:
    """Return substring of the first top-level ``{ ... }`` using brace depth (strings aware)."""
    start = text.find("{")
    if start == -1:
        raise JSONParseError("No '{' found; expected a JSON object")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise JSONParseError("Unclosed '{' while scanning for JSON object")


def safe_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON object from ``text``. Raises :class:`JSONParseError` on failure."""
    if text is None:
        raise JSONParseError("Input is None")
    if not isinstance(text, str):
        raise JSONParseError(f"Expected str, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise JSONParseError("Empty string cannot be parsed as JSON")

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Invalid JSON: {e}") from e

    if not isinstance(value, dict):
        raise JSONParseError(
            f"JSON root must be an object (dict), got {type(value).__name__}"
        )
    return value


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    """Parse a JSON object from typical LLM output (prose, markdown fences, extra lines).

    Strips BOM, unwraps the first ``` / ```json fenced block if present, drops a lone
    leading ``json`` line, then either parses the whole string (if already strict JSON)
    or extracts the first top-level ``{...}`` object and parses it.

    For input that is already a bare JSON object string, use :func:`safe_parse_json`.
    """
    if text is None:
        raise JSONParseError("Input is None")
    if not isinstance(text, str):
        raise JSONParseError(f"Expected str, got {type(text).__name__}")

    t = text.strip().replace("\ufeff", "")
    if not t:
        raise JSONParseError("Empty string cannot be parsed as JSON")

    t = _unwrap_markdown_fence(t)
    t = _strip_leading_json_label(t)
    if not t:
        raise JSONParseError("Empty string after stripping LLM wrappers")

    try:
        return safe_parse_json(t)
    except JSONParseError:
        pass

    try:
        fragment = _extract_first_json_object(t)
        value = json.loads(fragment)
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Invalid JSON in extracted object: {e}") from e

    if not isinstance(value, dict):
        raise JSONParseError(
            f"JSON root must be an object (dict), got {type(value).__name__}"
        )
    return value


def validate_json(data: Dict[str, Any], required_keys: List[str]) -> None:
    """Ensure ``data`` contains every key in ``required_keys``."""
    if not isinstance(data, dict):
        raise JSONParseError(f"Expected dict, got {type(data).__name__}")
    if required_keys is None or not isinstance(required_keys, list):
        raise JSONParseError("required_keys must be a non-None list")

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise JSONParseError(f"Missing required keys: {missing}")
