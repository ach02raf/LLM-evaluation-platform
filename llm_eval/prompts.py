"""Prompt versions are plain JSON files in /prompts. Add v3.json to test a new one."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PROMPTS_DIR


@dataclass
class PromptTemplate:
    name: str
    description: str
    system: str
    user: str  # must contain {context} and {question}

    def render(self, context: str, question: str) -> tuple[str, str]:
        return self.system, self.user.format(context=context, question=question)


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    data = json.loads((Path(prompts_dir) / f"{name}.json").read_text(encoding="utf-8"))
    return PromptTemplate(data["name"], data.get("description", ""), data["system"], data["user"])


def list_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(prompts_dir).glob("*.json"))
