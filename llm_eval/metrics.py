"""Evaluation metrics.

Heuristic metrics (fast, free, deterministic) are used by default.
An optional LLM-as-judge (see judge.py) can override accuracy/groundedness.
"""
from __future__ import annotations

from .text_utils import content_tokens, numbers, split_sentences, strip_citations, normalize

NOT_IN_DOCS = "NOT_IN_DOCS"

_ABSTAIN = (
    "i don't know", "i do not know", "don't know based", "not in the", "no information",
    "cannot find", "can't find", "not mentioned", "not specified", "not provided",
    "unable to find", "does not contain", "doesn't contain", "not available in",
    "insufficient information", "no mention", "not stated", "does not specify",
    "does not mention", "do not have",
)


def is_abstention(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return any(p in t for p in _ABSTAIN)


def accuracy(expected: str, answer: str) -> float:
    """1.0 if the answer contains the expected facts, else 0.0.

    - unanswerable questions (expected == NOT_IN_DOCS): correct only if the model abstains
    - otherwise every number in `expected` must appear in `answer` and
      >= 60% of the expected keywords must be present.
    """
    answer = strip_citations(answer or "")
    if expected == NOT_IN_DOCS:
        return 1.0 if is_abstention(answer) else 0.0
    if not answer.strip() or is_abstention(answer):
        return 0.0
    exp_nums = numbers(expected)
    if exp_nums and not exp_nums <= numbers(answer):
        return 0.0
    exp_tokens = set(content_tokens(expected))
    if not exp_tokens:
        return 1.0 if normalize(expected) in normalize(answer) else 0.0
    recall = len(exp_tokens & set(content_tokens(answer))) / len(exp_tokens)
    return 1.0 if recall >= 0.6 else 0.0


def groundedness(answer: str, context: str, min_coverage: float = 0.7) -> float:
    """Share of answer sentences supported by the retrieved context (0..1).

    A sentence is supported when >= 70% of its content words appear in the
    context AND every number it contains appears in the context.
    An abstention claims nothing, so it is fully grounded.
    """
    answer = strip_citations(answer or "")
    if not answer.strip():
        return 0.0
    if is_abstention(answer):
        return 1.0
    ctx_tokens = set(content_tokens(context))
    ctx_nums = numbers(context)
    supported = counted = 0
    for sent in split_sentences(answer):
        toks = content_tokens(sent)
        if not toks:
            continue
        counted += 1
        coverage = sum(t in ctx_tokens for t in toks) / len(toks)
        if coverage >= min_coverage and numbers(sent) <= ctx_nums:
            supported += 1
    return supported / counted if counted else 0.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def compute_cost(input_tokens: int, output_tokens: int, price_in_per_1m: float, price_out_per_1m: float) -> float:
    return (input_tokens * price_in_per_1m + output_tokens * price_out_per_1m) / 1_000_000
