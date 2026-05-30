"""Load prompt files from the project ``prompts/`` tree and combine text parts."""

from __future__ import annotations

from pathlib import Path
from typing import List


def get_prompts_dir() -> Path:
    """Return ``<project_root>/prompts`` (``utils`` and ``prompts`` are siblings)."""
    return Path(__file__).resolve().parent.parent / "prompts"


def _validate_segment(name: str, label: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"{label} must be str, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise ValueError(f"{label} must be non-empty")
    if ".." in stripped or stripped.startswith(("/", "\\")):
        raise ValueError(f"{label} contains invalid path segments: {name!r}")
    if any(sep in stripped for sep in ("/", "\\")):
        raise ValueError(f"{label} must be a single path segment, got {name!r}")
    return stripped


def load_prompt(folder: str, filename: str) -> str:
    """Read UTF-8 text from ``prompts/{folder}/{filename}`` under the project root."""
    folder_s = _validate_segment(folder, "folder")
    filename_s = _validate_segment(filename, "filename")

    path = get_prompts_dir() / folder_s / filename_s
    if not path.is_file():
        raise ValueError(f"Prompt file not found: {path}")

    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ValueError(f"Failed to read prompt file {path}: {e}") from e

    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode prompt file as UTF-8 or GBK: {path}")

    return text.strip()


def combine_prompts(parts: List[str]) -> str:
    """Join prompt fragments with blank lines; strips each part and the result."""
    if parts is None or not isinstance(parts, list):
        raise ValueError("parts must be a non-empty list of strings")
    if not parts:
        raise ValueError("parts must not be empty")

    cleaned: List[str] = []
    for i, p in enumerate(parts):
        if not isinstance(p, str):
            raise ValueError(f"parts[{i}] must be str, got {type(p).__name__}")
        s = p.strip()
        if not s:
            raise ValueError(f"parts[{i}] is empty or whitespace-only after strip")
        cleaned.append(s)

    return "\n\n".join(cleaned).strip()
