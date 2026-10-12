"""Small text helpers shared by the retriever, the mock LLM and the metrics."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "at", "for",
    "and", "or", "by", "with", "as", "from", "that", "this", "it", "its", "their", "they", "them",
    "do", "does", "did", "how", "what", "which", "who", "when", "where", "why", "many", "much",
    "can", "may", "must", "will", "would", "should", "per", "any", "all", "if", "then", "than",
    "s", "there", "have", "has", "had", "after", "before", "about", "up", "out", "not",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = strip_accents(text.lower())
    t = t.replace("’", "'").replace("'s", "")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,200 -> 1200
    return t


def stem(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", normalize(text))


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", normalize(text)))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip(" -*•\t") for p in parts if p and p.strip(" -*•\t")]


def strip_citations(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", " ", text)
