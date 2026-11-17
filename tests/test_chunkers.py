import pytest

from chunkaudit import (
    DEFAULT_STRATEGIES,
    FixedChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    load_dataset,
    parse_strategy,
    sample_path,
)
from chunkaudit.text import count_words

TEXT = (
    "One two three four five. Six seven eight nine ten.\n\n"
    "Eleven twelve thirteen. Fourteen fifteen sixteen seventeen eighteen nineteen twenty."
)


def pieces(text, spans):
    return [text[start:end] for start, end in spans]


def test_fixed_chunker_cuts_every_n_words():
    assert pieces(TEXT, FixedChunker(8).split(TEXT)) == [
        "One two three four five. Six seven eight",
        "nine ten.\n\nEleven twelve thirteen. Fourteen fifteen sixteen",
        "seventeen eighteen nineteen twenty.",
    ]


def test_fixed_chunker_overlap_repeats_words():
    chunks = pieces(TEXT, FixedChunker(8, overlap=4).split(TEXT))
    assert chunks[0].split()[4:] == chunks[1].split()[:4]
    assert chunks[-1].endswith("twenty.")
    assert all(len(chunk.split()) <= 8 for chunk in chunks)


def test_fixed_chunker_does_not_emit_a_tail_already_covered():
    text = "a b c d e f"
    assert pieces(text, FixedChunker(4, overlap=2).split(text)) == ["a b c d", "c d e f"]


def test_sentence_chunker_keeps_sentences_whole():
    assert pieces(TEXT, SentenceChunker(2).split(TEXT)) == [
        "One two three four five. Six seven eight nine ten.",
        "Eleven twelve thirteen. Fourteen fifteen sixteen seventeen eighteen nineteen twenty.",
    ]
    overlapped = pieces(TEXT, SentenceChunker(2, overlap=1).split(TEXT))
    assert len(overlapped) == 3
    assert overlapped[1].startswith("Six seven") and overlapped[1].endswith("thirteen.")


def test_paragraph_chunker_packs_up_to_the_limit():
    assert len(ParagraphChunker(100).split(TEXT)) == 1
    assert pieces(TEXT, ParagraphChunker(10).split(TEXT)) == [
        "One two three four five. Six seven eight nine ten.",
        "Eleven twelve thirteen. Fourteen fifteen sixteen seventeen eighteen nineteen twenty.",
    ]
    # a paragraph longer than the limit is kept whole
    assert len(ParagraphChunker(3).split(TEXT)) == 2


def test_recursive_chunker_falls_back_to_sentences_then_words():
    assert pieces(TEXT, RecursiveChunker(5).split(TEXT)) == [
        "One two three four five.",
        "Six seven eight nine ten.",
        "Eleven twelve thirteen.",
        "Fourteen fifteen sixteen seventeen eighteen",
        "nineteen twenty.",
    ]
    assert pieces(TEXT, RecursiveChunker(10).split(TEXT)) == [
        "One two three four five. Six seven eight nine ten.",
        "Eleven twelve thirteen. Fourteen fifteen sixteen seventeen eighteen nineteen twenty.",
    ]


def test_heading_chunker_splits_sections_and_keeps_the_preamble():
    text = "Preamble.\r\n\r\n# One\r\nFirst section.\r\n\r\nTwo\r\n---\r\nSecond section."
    assert pieces(text, parse_strategy("heading:20").split(text)) == [
        "Preamble.",
        "# One\r\nFirst section.",
        "Two\r\n---\r\nSecond section.",
    ]


def test_heading_chunker_ignores_headings_inside_fences():
    text = "# Setup\n````md\n# Example\n```\nTitle\n---\n````\n## Next\nDone."
    assert pieces(text, parse_strategy("heading:40").split(text)) == [
        "# Setup\n````md\n# Example\n```\nTitle\n---\n````",
        "## Next\nDone.",
    ]


def test_heading_chunker_splits_oversized_sections_at_paragraph_boundaries():
    text = "# One\n\nOne two three.\n\nFour five six.\n\n# Two\nSeven."
    assert pieces(text, parse_strategy("heading:5").split(text)) == [
        "# One\n\nOne two three.",
        "Four five six.",
        "# Two\nSeven.",
    ]


@pytest.mark.parametrize(
    "spec", [*DEFAULT_STRATEGIES, "fixed:7", "sentence:1", "recursive:12", "heading:12"]
)
def test_every_strategy_covers_every_word_of_the_sample_documents(spec):
    chunker = parse_strategy(spec)
    assert chunker.name == spec
    for document in load_dataset(sample_path()).documents:
        spans = chunker.split(document.text)
        assert spans
        covered = set()
        for start, end in spans:
            assert 0 <= start < end <= len(document.text)
            assert document.text[start:end] == document.text[start:end].strip()
            covered.update(range(start, end))
        missing = [
            character for position, character in enumerate(document.text)
            if position not in covered and not character.isspace()
        ]  # fmt: skip
        assert not missing


@pytest.mark.parametrize("spec", ["fixed:50", "fixed:50:10", "recursive:50", "heading:50"])
def test_size_limits_are_respected(spec):
    chunker = parse_strategy(spec)
    for document in load_dataset(sample_path()).documents:
        for start, end in chunker.split(document.text):
            assert count_words(document.text, start, end) <= 50


def test_chunkers_handle_empty_text():
    for spec in ("fixed:5", "sentence:2", "paragraph:5", "recursive:5", "heading:5"):
        assert parse_strategy(spec).split("") == []


@pytest.mark.parametrize(
    "spec",
    [
        "semantic:200",
        "fixed",
        "fixed:a",
        "fixed:10:10",
        "fixed:0",
        "paragraph:10:2",
        "sentence:2:5",
        "heading:0",
        "heading:10:2",
    ],
)
def test_invalid_specs_are_rejected_with_the_spec_in_the_message(spec):
    with pytest.raises(ValueError, match=spec.split(":")[0]):
        parse_strategy(spec)
