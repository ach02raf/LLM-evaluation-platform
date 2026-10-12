"""Document loading + BM25 retrieval (pure Python, no embeddings needed)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .text_utils import content_tokens


@dataclass
class Chunk:
    source: str   # file name, e.g. hr_vacation.md
    section: str  # markdown "## " heading
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} > {self.section}"


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """One chunk per '## ' section of every markdown file."""
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        for block in re.split(r"(?m)^## ", raw)[1:]:
            heading, _, body = block.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(path.name, heading.strip(), body))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [content_tokens(f"{c.section} {c.text}") for c in chunks]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = content_tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            score = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scored.append((self.chunks[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s > 0]
