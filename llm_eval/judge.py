"""Optional LLM-as-judge. Falls back to heuristics if the judge output can't be parsed."""
from __future__ import annotations

import json
import re

JUDGE_SYSTEM = "You are a strict evaluator of question-answering systems. Reply with JSON only."
JUDGE_USER = """Question: {question}
Expected answer: {expected}
Retrieved context:
{context}

Candidate answer: {answer}

Return JSON: {{"accuracy": 0 or 1, "groundedness": number between 0 and 1}}
- accuracy = 1 if the candidate conveys the expected answer (if expected is NOT_IN_DOCS, 1 only if the candidate says it does not know).
- groundedness = fraction of the candidate's claims that are supported by the retrieved context (1 if it only says it does not know)."""


class LLMJudge:
    def __init__(self, client):
        self.client = client

    def evaluate(self, question: str, expected: str, answer: str, context: str) -> dict | None:
        try:
            resp = self.client.generate(
                JUDGE_SYSTEM,
                JUDGE_USER.format(question=question, expected=expected, context=context, answer=answer),
                max_tokens=120,
            )
            m = re.search(r"\{.*?\}", resp.text, re.S)
            data = json.loads(m.group(0))
            return {
                "accuracy": float(1.0 if float(data["accuracy"]) >= 0.5 else 0.0),
                "groundedness": min(1.0, max(0.0, float(data["groundedness"]))),
            }
        except Exception:
            return None
