"""RAG pipeline: retrieve -> build prompt -> generate."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMResponse
from .prompts import PromptTemplate
from .retriever import BM25Retriever, Chunk


def format_context(scored_chunks: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.label}]\n{c.text}" for c, _ in scored_chunks)


@dataclass
class RAGResult:
    answer: str
    retrieved: list[tuple[Chunk, float]]
    response: LLMResponse
    context: str


class RAGPipeline:
    def __init__(self, retriever: BM25Retriever, client, prompt: PromptTemplate, top_k: int = 3):
        self.retriever, self.client, self.prompt, self.top_k = retriever, client, prompt, top_k

    def generate(self, question: str, retrieved: list[tuple[Chunk, float]]) -> RAGResult:
        context = format_context(retrieved)
        system, user = self.prompt.render(context, question)
        resp = self.client.generate(system, user)
        return RAGResult(resp.text, retrieved, resp, context)

    def answer(self, question: str) -> RAGResult:
        return self.generate(question, self.retriever.search(question, self.top_k))
