"""Central configuration: paths, provider presets, pricing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"
TEST_SET_PATH = ROOT / "data" / "test_set.json"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

# All real providers expose an OpenAI-compatible /chat/completions endpoint,
# so a single client handles all of them. Model names change over time:
# override them with the *_MODEL variables in .env if one is retired.
PROVIDERS: dict[str, dict] = {
    "mock": {
        "label": "Mock (offline, no AI, for testing the pipeline)",
        "base_url": None,
        "key_env": None,
        "model_env": None,
        "default_model": "extractive-mock",
    },
    "ollama": {
        "label": "Ollama (local, 100% free)",
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model_env": "OLLAMA_MODEL",
        "default_model": "llama3.2:3b",
    },
    "groq": {
        "label": "Groq (free tier, very fast)",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "label": "Google Gemini (free tier)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
    },
    "openrouter": {
        "label": "OpenRouter (free models available)",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}


@dataclass
class Settings:
    # Reference price used to compute an *equivalent* cost even on free providers
    # (defaults = gpt-4o-mini list price, USD per 1M tokens). Edit in .env.
    price_input_per_1m: float = 0.15
    price_output_per_1m: float = 0.60
    request_delay: float = 0.0  # seconds between calls (useful for free-tier rate limits)


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv is optional
        pass
    return Settings(
        price_input_per_1m=float(os.getenv("PRICE_INPUT_PER_1M", 0.15)),
        price_output_per_1m=float(os.getenv("PRICE_OUTPUT_PER_1M", 0.60)),
        request_delay=float(os.getenv("REQUEST_DELAY", 0.0)),
    )
