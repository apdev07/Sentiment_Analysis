from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

TEXT_COLUMN_ALIASES = (
    "Tweet Text",
    "review",
    "reviews",
    "text",
    "comment",
    "comments",
    "content",
    "message",
    "body",
)
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 50
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


@dataclass(frozen=True)
class CsvTextBatch:
    column: str
    texts: tuple[str, ...]
    truncated: bool


def normalize_header(header: object) -> str:
    return " ".join(
        str(header or "").replace("_", " ").replace("-", " ").split()
    ).lower()


def find_text_column(headers: Iterable[object]) -> str | None:
    """Return the first available column in TEXT_COLUMN_ALIASES priority order."""
    normalized_headers: dict[str, str] = {}
    for header in headers:
        normalized_headers.setdefault(normalize_header(header), str(header))
    for alias in TEXT_COLUMN_ALIASES:
        if column := normalized_headers.get(normalize_header(alias)):
            return column
    return None


def read_texts_from_csv(
    csv_input: bytes | str,
    limit: int = MAX_CSV_ROWS,
) -> CsvTextBatch:
    if limit < 1:
        raise ValueError("CSV row limit must be at least 1.")
    if isinstance(csv_input, bytes) and len(csv_input) > MAX_CSV_BYTES:
        raise ValueError("CSV file must be 5 MiB or smaller.")

    csv_text = (
        csv_input.decode("utf-8-sig") if isinstance(csv_input, bytes) else csv_input
    )
    if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError("CSV file must be 5 MiB or smaller.")
    reader = csv.DictReader(io.StringIO(csv_text))

    if reader.fieldnames is None:
        raise ValueError("CSV file must include a header row.")

    column = find_text_column(reader.fieldnames)
    if column is None:
        aliases = ", ".join(TEXT_COLUMN_ALIASES)
        raise ValueError(f"CSV must include one text column such as: {aliases}.")

    texts: list[str] = []
    truncated = False
    for row in reader:
        value = row.get(column)
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            continue
        if len(texts) >= limit:
            truncated = True
            break
        texts.append(cleaned)

    if not texts:
        raise ValueError("CSV has no non-empty text values.")

    return CsvTextBatch(column=column, texts=tuple(texts), truncated=truncated)


def _spreadsheet_safe(value: object) -> str:
    text = str(value)
    return f"'{text}" if text.startswith(FORMULA_PREFIXES) else text


def render_batch_results_csv(rows: Iterable[Mapping[str, object]]) -> str:
    buffer = io.StringIO()
    fieldnames = ("review", "sentiment", "confidence")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {field: _spreadsheet_safe(row.get(field, "")) for field in fieldnames}
        for row in rows
    )
    return buffer.getvalue()
