"""chunkaudit: audit RAG chunking strategies without an API key."""

__version__ = "0.4.0"

from .bm25 import BM25Index
from .chunkers import (
    DEFAULT_STRATEGIES,
    Chunker,
    FixedChunker,
    HeadingChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    parse_strategy,
)
from .dataset import Dataset, DatasetError, Document, Question, load_dataset, sample_path
from .embeddings import EmbeddingIndex, EmbeddingRetriever
from .evaluate import SplitExample, StrategyResult, compare, evaluate, split_examples
from .report import render_csv, render_json, render_markdown, render_table, sort_results

__all__ = [
    "DEFAULT_STRATEGIES",
    "BM25Index",
    "Chunker",
    "Dataset",
    "DatasetError",
    "Document",
    "EmbeddingIndex",
    "EmbeddingRetriever",
    "FixedChunker",
    "HeadingChunker",
    "ParagraphChunker",
    "Question",
    "RecursiveChunker",
    "SentenceChunker",
    "SplitExample",
    "StrategyResult",
    "__version__",
    "compare",
    "evaluate",
    "load_dataset",
    "parse_strategy",
    "render_csv",
    "render_json",
    "render_markdown",
    "render_table",
    "sample_path",
    "sort_results",
    "split_examples",
]
