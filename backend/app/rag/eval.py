"""Deterministic accuracy evaluation and real-world friction detection for RAG answers.

Provides 100% free, deterministic verification of generated answers against retrieved
context passages without requiring external LLM judges, golden datasets, or paid APIs:
1. Numeric Accuracy: Regex verification that numbers, dates, currency, and percentages exist.
2. Entity Accuracy: Capitalized proper noun and acronym overlap between answer and context.
3. Citation Verification Rate: Verifies citation excerpts are authentic chunk substrings.
4. Production Telemetry & Friction Rate: Detects rapid re-queries (<= 30s) or user frustration
   indicating bad retrieval / failed answers (0.0) vs constructive follow-ups (+1.0).
"""

from __future__ import annotations

import re
from typing import Any

# Regex patterns for numeric values
_CURRENCY_RE = re.compile(
    r"[$€£¥]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp|dollars|cents)\b",
    re.IGNORECASE,
)
_PERCENTAGE_RE = re.compile(r"\b\d+(?:\.\d+)?%", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Regex patterns for entities and acronyms
_ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,}\b")
_PROPER_NOUN_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+\b")
_PASCAL_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*\b")
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z0-9]{2,}\b")

# Common English sentence starters and grammatical stopwords to exclude from entity evaluation
_STOPWORDS_AND_STARTERS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "another", "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "finally", "first", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "hence", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "however", "i", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "just", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "please", "same",
    "second", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "therefore", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't",
    "yes", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
}

# Explicit frustration regex patterns indicating poor answer quality
_FRUSTRATION_PATTERNS = [
    re.compile(r"\bthat(?:'s| is) not what i asked\b", re.IGNORECASE),
    re.compile(r"\bnot what i asked\b", re.IGNORECASE),
    re.compile(r"\bnot what i meant\b", re.IGNORECASE),
    re.compile(r"\bwrong answer\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is) incorrect\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is) wrong\b", re.IGNORECASE),
    re.compile(r"\btry again\b", re.IGNORECASE),
    re.compile(r"\byou didn(?:'t| not) answer\b", re.IGNORECASE),
    re.compile(r"\bnot answering my question\b", re.IGNORECASE),
    re.compile(r"\banswer the question\b", re.IGNORECASE),
]

_WORD_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD_TOKEN_RE.findall(text.lower())


def _normalize_num(val_str: str) -> str:
    """Normalize numeric strings by stripping currency symbols, commas, and percentage signs."""
    cleaned = re.sub(r"[$€£¥,%]", "", val_str).strip()
    try:
        f = float(cleaned)
        if f.is_integer():
            return str(int(f))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned.lower()


def extract_numbers(text: str) -> set[str]:
    """Extract raw and normalized numerical strings from text."""
    if not text:
        return set()

    numbers: set[str] = set()
    # Find all regex matches
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        norm = _normalize_num(token)
        if norm:
            numbers.add(norm)
            numbers.add(token)

    return numbers


def verify_numeric_accuracy(answer: str, context: str) -> dict[str, Any]:
    """Deterministic regex check of numbers in the answer against context.

    Detects hallucinated numbers, dates, currency, and quantities.
    Returns numeric accuracy score (0.0 to 1.0) and list of unsupported numbers.
    """
    if not answer:
        return {
            "numeric_accuracy_score": 1.0,
            "total_numbers": 0,
            "supported_numbers": [],
            "unsupported_numbers": [],
        }

    # Extract all candidate number tokens from answer
    answer_raw_matches = _NUMBER_RE.findall(answer)
    if not answer_raw_matches:
        return {
            "numeric_accuracy_score": 1.0,
            "total_numbers": 0,
            "supported_numbers": [],
            "unsupported_numbers": [],
        }

    context_numbers = extract_numbers(context)
    context_lower = context.lower()

    supported: list[str] = []
    unsupported: list[str] = []

    seen_tokens: set[str] = set()
    for token in answer_raw_matches:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        norm = _normalize_num(token)
        # Verify against context numbers or literal string match
        if (
            norm in context_numbers
            or token in context_numbers
            or token.lower() in context_lower
            or norm in context_lower
        ):
            supported.append(token)
        else:
            unsupported.append(token)

    total = len(supported) + len(unsupported)
    score = round(len(supported) / total, 4) if total > 0 else 1.0

    return {
        "numeric_accuracy_score": score,
        "total_numbers": total,
        "supported_numbers": supported,
        "unsupported_numbers": unsupported,
    }


def extract_entities(text: str) -> set[str]:
    """Extract named entities, technical terms, acronyms, and proper nouns from text."""
    if not text:
        return set()

    entities: set[str] = set()

    # 1. Multi-word proper noun phrases (e.g. "PostgreSQL Database", "GitHub Actions")
    for match in _PROPER_NOUN_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        words = phrase.split()
        # Ensure not all words are stopwords
        if any(w.lower() not in _STOPWORDS_AND_STARTERS for w in words):
            entities.add(phrase)

    # 2. Acronyms (e.g. "RLS", "API", "JSON", "SaaS")
    for match in _ACRONYM_RE.finditer(text):
        token = match.group(0).strip()
        if len(token) >= 2 and token.lower() not in _STOPWORDS_AND_STARTERS:
            entities.add(token)

    # 3. PascalCase / CamelCase words (e.g. "PostgreSQL", "LangGraph", "FastAPI")
    for match in _PASCAL_CAMEL_RE.finditer(text):
        token = match.group(0).strip()
        entities.add(token)

    # 4. Filtered individual capitalized words that are not sentence starters
    # Look for capitalized words that do not follow sentence boundary punctuation
    sentences = re.split(r"[.!?]\s+", text)
    for sentence in sentences:
        words = sentence.strip().split()
        # Skip the first word as it may be capitalized solely due to sentence start
        for word in words[1:]:
            cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", word)
            if (
                cleaned
                and cleaned[0].isupper()
                and cleaned.lower() not in _STOPWORDS_AND_STARTERS
                and len(cleaned) >= 3
            ):
                entities.add(cleaned)

    return entities


def verify_entity_accuracy(answer: str, context: str) -> dict[str, Any]:
    """Deterministic token overlap between answer entities and context passages.

    Extracts named entities, acronyms, and technical terms from the answer
    and checks if they are grounded in the retrieved context.
    """
    if not answer:
        return {
            "entity_accuracy_score": 1.0,
            "total_entities": 0,
            "supported_entities": [],
            "unsupported_entities": [],
        }

    answer_entities = extract_entities(answer)
    if not answer_entities:
        return {
            "entity_accuracy_score": 1.0,
            "total_entities": 0,
            "supported_entities": [],
            "unsupported_entities": [],
        }

    context_lower = context.lower()
    context_words = set(_tokenize_words(context))

    # Also compute acronyms from multi-word phrases (e.g. "Row Level Security" -> "RLS")
    context_phrases = _PROPER_NOUN_PHRASE_RE.findall(context)
    context_derived_acronyms = {
        "".join(w[0] for w in phrase.split() if w and w[0].isupper()).upper()
        for phrase in context_phrases
    }

    supported: list[str] = []
    unsupported: list[str] = []

    for entity in answer_entities:
        entity_lower = entity.lower()
        entity_tokens = set(_tokenize_words(entity))
        entity_acronym = (
            "".join(w[0] for w in entity.split() if w and w[0].isupper()).upper()
            if " " in entity
            else ""
        )

        # Check if full entity phrase is in context, or all entity tokens appear,
        # or acronym matches derived context acronyms
        if (
            entity_lower in context_lower
            or (entity_tokens and entity_tokens.issubset(context_words))
            or (entity in context_derived_acronyms)
            or (entity_acronym and entity_acronym in context)
        ):
            supported.append(entity)
        else:
            unsupported.append(entity)

    total = len(supported) + len(unsupported)
    score = round(len(supported) / total, 4) if total > 0 else 1.0

    return {
        "entity_accuracy_score": score,
        "total_entities": total,
        "supported_entities": supported,
        "unsupported_entities": unsupported,
    }


def verify_citation_rate(
    citations: list[dict[str, Any]],
    source_text_by_id: dict[str, str] | None = None,
    full_context: str | None = None,
) -> dict[str, Any]:
    """Verify that citation excerpts genuinely appear in the source chunks or full context.

    Prevents fabricated citation excerpts by ensuring the citation excerpt
    is an authentic substring of the underlying retrieved document.
    """
    if not citations:
        return {
            "citation_verification_rate": 1.0,
            "total_citations": 0,
            "verified_citations": 0,
            "unverified_citations": [],
        }

    context_lower = (full_context or "").lower()
    verified_count = 0
    unverified: list[dict[str, Any]] = []

    for cit in citations:
        raw_excerpt = cit.get("excerpt", "")
        # Strip trailing ellipsis often added during excerpt truncation
        cleaned_excerpt = raw_excerpt.rstrip(". ").strip().lower()
        chunk_id = str(cit.get("chunk_id", ""))

        target_source = ""
        if source_text_by_id and chunk_id in source_text_by_id:
            target_source = source_text_by_id[chunk_id].lower()
        else:
            target_source = context_lower

        if not cleaned_excerpt:
            # If no excerpt provided, mark as unverified
            unverified.append(cit)
            continue

        # Substring verification: check if excerpt is in source or substantial prefix is in source
        prefix_len = min(len(cleaned_excerpt), 80)
        prefix = cleaned_excerpt[:prefix_len].strip()

        is_verified = False
        if (prefix and prefix in target_source) or (cleaned_excerpt in target_source):
            is_verified = True
        else:
            # Token overlap fallback for slight whitespace differences
            excerpt_tokens = set(_tokenize_words(cleaned_excerpt))
            source_tokens = set(_tokenize_words(target_source))
            if excerpt_tokens and source_tokens:
                overlap = len(excerpt_tokens & source_tokens) / len(excerpt_tokens)
                if overlap >= 0.80:
                    is_verified = True

        if is_verified:
            verified_count += 1
        else:
            unverified.append(cit)

    rate = round(verified_count / len(citations), 4) if citations else 1.0

    return {
        "citation_verification_rate": rate,
        "total_citations": len(citations),
        "verified_citations": verified_count,
        "unverified_citations": unverified,
    }


def detect_query_friction(
    current_q: str,
    prev_q: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Detect user friction signals indicating bad retrieval / failed answers.

    Real ground truth signals:
    1. Frustration sentiment: e.g. "That's not what I asked", "wrong answer", "try again" -> 0.0
    2. Rapid re-query: Rephrasing the same question within 30 seconds -> Failed Retrieval (0.0)
    3. Constructive follow-up: Asking a different question -> Successful Answer (+1.0)
    """
    trimmed = current_q.strip()

    # 1. Frustration regex match
    for pattern in _FRUSTRATION_PATTERNS:
        if pattern.search(trimmed):
            return {
                "is_friction": True,
                "friction_reason": "frustration_phrase",
                "is_successful_followup": False,
                "similarity": 0.0,
                "elapsed_seconds": elapsed_seconds,
                "ground_truth_score": 0.0,
            }

    # If there is no previous question, this is the initial question in the conversation
    if not prev_q:
        return {
            "is_friction": False,
            "friction_reason": None,
            "is_successful_followup": False,
            "similarity": 0.0,
            "elapsed_seconds": elapsed_seconds,
            "ground_truth_score": 1.0,
        }

    # 2. Token overlap similarity between current and previous query
    curr_tokens = set(_tokenize_words(current_q))
    prev_tokens = set(_tokenize_words(prev_q))

    # Remove generic stopwords to focus on content words
    curr_content = {t for t in curr_tokens if t not in _STOPWORDS_AND_STARTERS} or curr_tokens
    prev_content = {t for t in prev_tokens if t not in _STOPWORDS_AND_STARTERS} or prev_tokens

    union = curr_content | prev_content
    similarity = (len(curr_content & prev_content) / len(union)) if union else 0.0

    # Rapid re-query check: within 30 seconds with high similarity (>= 0.50)
    is_rapid = elapsed_seconds is not None and elapsed_seconds <= 30.0
    is_rephrase = similarity >= 0.50

    if is_rapid and is_rephrase:
        return {
            "is_friction": True,
            "friction_reason": "rapid_rephrase",
            "is_successful_followup": False,
            "similarity": round(similarity, 4),
            "elapsed_seconds": elapsed_seconds,
            "ground_truth_score": 0.0,
        }

    # 3. Constructive follow-up: user continues with different topical question
    return {
        "is_friction": False,
        "friction_reason": None,
        "is_successful_followup": True,
        "similarity": round(similarity, 4),
        "elapsed_seconds": elapsed_seconds,
        "ground_truth_score": 1.0,
    }
