"""Text segmentation helpers.

Every function returns character spans ``(start, end)`` into the original text, so a chunk can
always be traced back to the exact characters it covers.
"""

from __future__ import annotations

import re
from itertools import pairwise

Span = tuple[int, int]

_WORD = re.compile(r"\S+")
_TOKEN = re.compile(r"[a-z0-9]+")
_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n\s*")
_ATX_HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]|$)")
_SETEXT_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
_CODE_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_NON_PARAGRAPH = re.compile(r" {4}| {0,3}(?:>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)|\t)")
# Sentence-ending punctuation, optional closing quotes or brackets, whitespace, and then
# something that looks like the start of a new sentence.
_SENTENCE_END = re.compile(r"""[.!?]+["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e",
        "fig", "inc", "ltd", "gen", "col", "lt", "sgt", "capt", "rev", "u.s", "u.k",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, used for retrieval."""
    return _TOKEN.findall(text.lower())


def word_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Spans of whitespace-separated words inside ``text[start:end]``."""
    stop = len(text) if end is None else end
    return [match.span() for match in _WORD.finditer(text, start, stop)]


def count_words(text: str, start: int = 0, end: int | None = None) -> int:
    stop = len(text) if end is None else end
    return sum(1 for _ in _WORD.finditer(text, start, stop))


def paragraph_spans(text: str) -> list[Span]:
    """Spans of paragraphs, which are separated by one or more blank lines."""
    spans: list[Span] = []
    position = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, len(text)))
    return [span for span in (_trim(text, start, end) for start, end in spans) if span]


def markdown_section_spans(text: str) -> list[Span]:
    """Split at top-level ATX and Setext headings outside fenced code blocks."""
    boundaries = [0]
    position = 0
    paragraph_start = None
    fence = ""
    for raw_line in text.splitlines(keepends=True):
        line_start = position
        position += len(raw_line)
        line = raw_line.rstrip("\r\n")
        marker = _CODE_FENCE.match(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker and (marker[1][0] == "~" or "`" not in marker[2]):
            fence = marker[1]
            paragraph_start = None
        elif _ATX_HEADING.match(line):
            boundaries.append(line_start)
            paragraph_start = None
        elif _SETEXT_UNDERLINE.fullmatch(line):
            if paragraph_start is not None:
                boundaries.append(paragraph_start)
            paragraph_start = None
        elif not line.strip() or _NON_PARAGRAPH.match(line):
            paragraph_start = None
        elif paragraph_start is None:
            paragraph_start = line_start
    boundaries.append(len(text))
    return [
        span for start, end in pairwise(boundaries) if (span := _trim(text, start, end)) is not None
    ]


def sentence_spans(text: str) -> list[Span]:
    """Spans of sentences. A sentence never crosses a paragraph break.

    The splitter is rule based: it breaks after ``.``, ``!`` or ``?`` when the next word starts
    with a capital letter or a digit, and it skips common abbreviations and single initials.
    """
    spans: list[Span] = []
    for paragraph_start, paragraph_end in paragraph_spans(text):
        spans.extend(sentences_in(text, paragraph_start, paragraph_end))
    return spans


def sentences_in(text: str, start: int, end: int) -> list[Span]:
    """Spans of the sentences inside one paragraph, ``text[start:end]``."""
    spans: list[Span] = []
    for match in _SENTENCE_END.finditer(text, start, end):
        if _ends_with_abbreviation(text[start : match.start()]):
            continue
        spans.append((start, match.start() + len(match.group().rstrip())))
        start = match.end()
    if start < end:
        spans.append((start, end))
    return spans


def _ends_with_abbreviation(fragment: str) -> bool:
    words = fragment.split()
    if not words:
        return False
    last = words[-1].lstrip("\"'([").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None
