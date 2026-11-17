"""Chunking strategies.

A chunker turns a document into a list of character spans. Sizes are counted in
whitespace-separated words, which is close enough to tokens for comparing strategies and keeps
the package free of tokenizer dependencies.

Strategies are written as short specs so they can be passed on the command line:

    fixed:200        200 words per chunk
    fixed:200:50     200 words per chunk, 50 words shared with the previous chunk
    sentence:5       5 sentences per chunk
    sentence:5:1     5 sentences per chunk, 1 sentence shared with the previous chunk
    paragraph:200    whole paragraphs, packed together up to 200 words
    recursive:200    paragraphs, then sentences, then words, packed up to 200 words
    heading:300      Markdown sections, split further if they exceed 300 words
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .text import (
    Span,
    count_words,
    markdown_section_spans,
    paragraph_spans,
    sentence_spans,
    sentences_in,
    word_spans,
)


class Chunker(Protocol):
    """Anything with a ``name`` and a ``split`` method can be audited."""

    @property
    def name(self) -> str: ...

    def split(self, text: str) -> list[Span]: ...


def _windows(units: list[Span], size: int, overlap: int) -> list[Span]:
    """Slide a window of ``size`` units over ``units``, stepping ``size - overlap`` each time."""
    spans: list[Span] = []
    step = size - overlap
    for first in range(0, len(units), step):
        window = units[first : first + size]
        spans.append((window[0][0], window[-1][1]))
        if first + size >= len(units):
            break
    return spans


def _check_window(size: int, overlap: int) -> None:
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")


def _pack(text: str, units: list[Span], max_words: int) -> list[Span]:
    """Join neighbouring units while the joined chunk stays within ``max_words``."""
    spans: list[Span] = []
    current: Span | None = None
    current_words = 0
    for start, end in units:
        words = count_words(text, start, end)
        if current is not None and current_words + words <= max_words:
            current = (current[0], end)
            current_words += words
        else:
            if current is not None:
                spans.append(current)
            current, current_words = (start, end), words
    if current is not None:
        spans.append(current)
    return spans


@dataclass(frozen=True)
class FixedChunker:
    """Fixed number of words per chunk, cut wherever the count runs out."""

    size: int
    overlap: int = 0

    def __post_init__(self) -> None:
        _check_window(self.size, self.overlap)

    @property
    def name(self) -> str:
        return f"fixed:{self.size}" + (f":{self.overlap}" if self.overlap else "")

    def split(self, text: str) -> list[Span]:
        return _windows(word_spans(text), self.size, self.overlap)


@dataclass(frozen=True)
class SentenceChunker:
    """Fixed number of sentences per chunk. Never cuts inside a sentence."""

    sentences: int
    overlap: int = 0

    def __post_init__(self) -> None:
        _check_window(self.sentences, self.overlap)

    @property
    def name(self) -> str:
        return f"sentence:{self.sentences}" + (f":{self.overlap}" if self.overlap else "")

    def split(self, text: str) -> list[Span]:
        return _windows(sentence_spans(text), self.sentences, self.overlap)


@dataclass(frozen=True)
class ParagraphChunker:
    """Whole paragraphs, packed together up to ``max_words``.

    A paragraph longer than ``max_words`` is kept whole, so chunks can exceed the limit.
    """

    max_words: int

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be at least 1")

    @property
    def name(self) -> str:
        return f"paragraph:{self.max_words}"

    def split(self, text: str) -> list[Span]:
        return _pack(text, paragraph_spans(text), self.max_words)


@dataclass(frozen=True)
class RecursiveChunker:
    """Split on the largest boundary that fits: paragraph, then sentence, then word.

    Pieces are then packed together up to ``max_words``, so no chunk exceeds the limit.
    """

    max_words: int

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be at least 1")

    @property
    def name(self) -> str:
        return f"recursive:{self.max_words}"

    def split(self, text: str) -> list[Span]:
        units: list[Span] = []
        for paragraph in paragraph_spans(text):
            if count_words(text, *paragraph) <= self.max_words:
                units.append(paragraph)
                continue
            for sentence in sentences_in(text, *paragraph):
                if count_words(text, *sentence) <= self.max_words:
                    units.append(sentence)
                else:
                    units.extend(_windows(word_spans(text, *sentence), self.max_words, 0))
        return _pack(text, units, self.max_words)


@dataclass(frozen=True)
class HeadingChunker:
    """Keep Markdown sections separate, splitting long sections by paragraph first."""

    max_words: int

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be at least 1")

    @property
    def name(self) -> str:
        return f"heading:{self.max_words}"

    def split(self, text: str) -> list[Span]:
        chunker = RecursiveChunker(self.max_words)
        return [
            (start + inner_start, start + inner_end)
            for start, end in markdown_section_spans(text)
            for inner_start, inner_end in chunker.split(text[start:end])
        ]


_KINDS = {
    "fixed": (FixedChunker, 1, 2),
    "sentence": (SentenceChunker, 1, 2),
    "paragraph": (ParagraphChunker, 1, 1),
    "recursive": (RecursiveChunker, 1, 1),
    "heading": (HeadingChunker, 1, 1),
}

DEFAULT_STRATEGIES = (
    "fixed:100",
    "fixed:200",
    "fixed:200:50",
    "fixed:400",
    "sentence:3",
    "sentence:5:1",
    "sentence:10:2",
    "paragraph:200",
    "recursive:200",
)


def parse_strategy(spec: str) -> Chunker:
    """Build a chunker from a spec such as ``fixed:200:50``."""
    kind, _, rest = spec.strip().partition(":")
    if kind not in _KINDS:
        raise ValueError(f"unknown strategy {spec!r}: expected one of {', '.join(sorted(_KINDS))}")
    factory, fewest, most = _KINDS[kind]
    parts = rest.split(":") if rest else []
    if not fewest <= len(parts) <= most:
        raise ValueError(f"strategy {spec!r} takes {fewest} to {most} numbers after {kind!r}")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        raise ValueError(f"strategy {spec!r}: sizes must be whole numbers") from None
    try:
        chunker: Chunker = factory(*numbers)
    except ValueError as error:
        raise ValueError(f"strategy {spec!r}: {error}") from None
    return chunker
