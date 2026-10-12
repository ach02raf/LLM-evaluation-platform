"""LLM clients.

- OpenAICompatClient: works with Ollama, Groq, Gemini, OpenRouter... (all free options)
- MockClient: deterministic extractive "model" so the whole pipeline runs offline
  (for tests/CI/demo only: its numbers say nothing about real LLM quality).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests

from .config import PROVIDERS
from .metrics import estimate_tokens
from .text_utils import content_tokens, split_sentences


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 120, max_retries: int = 5):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.max_retries = timeout, max_retries

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"network error: {e}"
                time.sleep(2 ** attempt)
                continue
            latency = time.perf_counter() - t0
            if r.status_code == 200:
                data = r.json()
                text = (data["choices"][0]["message"].get("content") or "").strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # reasoning models
                usage = data.get("usage") or {}
                return LLMResponse(
                    text,
                    usage.get("prompt_tokens") or estimate_tokens(system + user),
                    usage.get("completion_tokens") or estimate_tokens(text),
                    latency,
                )
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503, 504):  # free tiers rate-limit: back off
                wait = float(r.headers.get("retry-after", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            break
        raise LLMError(last_err)


class MockClient:
    """Offline extractive baseline. Picks the context sentence(s) closest to the question.

    Behaviour depends on the prompt so that V1/V2 differ like a real model would:
      - strict prompts (containing "I don't know") abstain when evidence is weak and
        answer with a single sentence + citation
      - loose prompts always answer with the 2 best sentences (and can hallucinate)
    """
    model = "extractive-mock"

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> LLMResponse:
        t0 = time.perf_counter()
        m = re.search(r"Context:\s*(.*?)\n\s*Question:\s*(.*?)(?:\nAnswer:.*)?$", user, re.S)
        context, question = (m.group(1), m.group(2)) if m else (user, user)
        strict = "i don't know" in system.lower()

        q_toks = set(content_tokens(question))
        scored: list[tuple[float, int, str, str]] = []
        source = ""
        for line in context.splitlines():
            hm = re.match(r"\[(.+?)(?: > .*)?\]\s*$", line.strip())
            if hm:
                source = hm.group(1)
                continue
            for sent in split_sentences(line):
                s_toks = set(content_tokens(sent))
                overlap = len(q_toks & s_toks)
                scored.append((overlap / max(1, len(q_toks)), overlap, sent, source))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or (strict and (scored[0][1] < 2 or scored[0][0] < 0.4)):
            text = "I don't know based on the provided documents."
        elif strict:
            text = f"{scored[0][2].rstrip('.')}. [{scored[0][3]}]"
        else:
            text = " ".join(s[2] for s in scored[:2])
        return LLMResponse(text, estimate_tokens(system + user), estimate_tokens(text), time.perf_counter() - t0)


def get_client(provider: str, model: str | None = None):
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    if provider == "mock":
        return MockClient()
    model = model or (os.getenv(cfg["model_env"]) if cfg["model_env"] else None) or cfg["default_model"]
    key = os.getenv(cfg["key_env"]) if cfg["key_env"] else None
    if cfg["key_env"] and not key:
        raise LLMError(f"Missing {cfg['key_env']}. Copy .env.example to .env and add your free key.")
    return OpenAICompatClient(cfg["base_url"], model, key)


def model_name(client) -> str:
    return getattr(client, "model", "unknown")
