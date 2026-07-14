"""Retrieval-augmented generation package."""

from app.rag.chain import AnswerResult, answer_question, semantic_search
from app.rag.chunking import Chunk, chunk_text
from app.rag.ingest import extract_text, index_document
from app.rag.retriever import RetrievedChunk, retrieve

__all__ = [
    "AnswerResult",
    "Chunk",
    "RetrievedChunk",
    "answer_question",
    "chunk_text",
    "extract_text",
    "index_document",
    "retrieve",
    "semantic_search",
]
