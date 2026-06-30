"""Shared text normalization helpers."""

from collections.abc import Iterable, Iterator
import re
from typing import TypeVar
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+")
T = TypeVar("T")


def clean_text(value: object, default: str = "") -> str:
    """Return a compact single-line string."""

    if value is None:
        value = default
    return " ".join(str(value).split())


def normalize_ascii_words(value: object) -> str:
    """Normalize text to lowercase ASCII words separated by spaces."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(TOKEN_RE.findall(ascii_text.lower()))


def unique_clean_texts(values: object, max_length: int = 120) -> list[str]:
    """Coerce a value or list of values into unique cleaned strings."""

    if values is None:
        return []
    raw_items = values if isinstance(values, list) else [values]

    strings: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = clean_text(item)[:max_length]
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        strings.append(text)
    return strings


def merge_unique_texts(
    groups: Iterable[Iterable[str]],
    max_length: int = 120,
) -> list[str]:
    """Merge several string groups while preserving order and uniqueness."""

    items: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = clean_text(item)[:max_length]
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            items.append(text)
    return items


def chunked(items: list[T], size: int) -> Iterator[list[T]]:
    """Yield fixed-size chunks from a list."""

    for index in range(0, len(items), size):
        yield items[index : index + size]


def find_constraint_hits(text: object, constraints: list[object]) -> list[str]:
    """Find negative/constraint terms inside normalized text."""

    normalized_text = " ".join(TOKEN_RE.findall(str(text).lower()))
    hits: list[str] = []
    seen: set[str] = set()

    for constraint in constraints:
        label = clean_text(constraint)
        normalized_constraint = " ".join(TOKEN_RE.findall(label.lower()))
        if not normalized_constraint or normalized_constraint in seen:
            continue

        if " " in normalized_constraint:
            matched = normalized_constraint in normalized_text
        else:
            matched = re.search(rf"\b{re.escape(normalized_constraint)}\b", normalized_text)

        if matched:
            seen.add(normalized_constraint)
            hits.append(label)
    return hits
