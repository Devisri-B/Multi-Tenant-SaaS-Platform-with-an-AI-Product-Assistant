"""Retrieval-augmented generation package."""

from app.rag.chain import AnswerResult, answer_question, semantic_search
from app.rag.chunking import Chunk, chunk_text
from app.rag.graph import assistant_graph, create_assistant_graph
from app.rag.ingest import extract_text, index_document
from app.rag.retriever import RetrievedChunk, retrieve
from app.rag.web_search import WebSearchResult, get_web_search_provider

__all__ = [
    "AnswerResult",
    "Chunk",
    "RetrievedChunk",
    "WebSearchResult",
    "answer_question",
    "assistant_graph",
    "chunk_text",
    "create_assistant_graph",
    "extract_text",
    "get_web_search_provider",
    "index_document",
    "retrieve",
    "semantic_search",
]
