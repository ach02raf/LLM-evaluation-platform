"""Run the test set against one or several prompt versions and compare them."""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import metrics
from .config import DOCS_DIR, RESULTS_DIR, TEST_SET_PATH, Settings, load_settings
from .judge import LLMJudge
from .llm import LLMError, LLMResponse, get_client, model_name
from .prompts import load_prompt
from .rag import RAGPipeline
from .retriever import BM25Retriever, load_chunks

# key, label, kind, higher_is_better
METRICS = [
    ("accuracy", "Accuracy", "pct", True),
    ("groundedness", "Groundedness", "pct", True),
    ("retrieval_hit", "Retrieval hit@k", "pct", True),
    ("avg_latency_s", "Avg latency", "sec", False),
    ("p95_latency_s", "P95 latency", "sec", False),
    ("avg_cost_usd", "Cost / question", "usd", False),
    ("total_cost_usd", "Total cost", "usd", False),
]


def fmt(kind: str, v: float | None) -> str:
    if v is None:
        return "n/a"
    return {"pct": f"{v * 100:.0f}%", "sec": f"{v:.2f}s", "usd": f"${v:.5f}"}[kind]


def load_test_set(path: Path = TEST_SET_PATH, limit: int | None = None) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def evaluate_question(q: dict, pipeline: RAGPipeline, settings: Settings, judge: LLMJudge | None) -> dict:
    t0 = time.perf_counter()
    retrieved = pipeline.retriever.search(q["question"], pipeline.top_k)
    retrieval_s = time.perf_counter() - t0

    error = None
    try:
        res = pipeline.generate(q["question"], retrieved)
        answer, resp, context = res.answer, res.response, res.context
    except LLMError as e:
        error, answer, context = str(e), "", ""
        resp = LLMResponse("", 0, 0, 0.0)

    acc = metrics.accuracy(q["expected"], answer)
    grd = metrics.groundedness(answer, "\n".join(c.text for c, _ in retrieved))
    judged = False
    if judge and not error:
        j = judge.evaluate(q["question"], q["expected"], answer, context)
        if j:
            acc, grd, judged = j["accuracy"], j["groundedness"], True

    src = q.get("expected_source")
    return {
        "id": q["id"],
        "category": q.get("category", ""),
        "question": q["question"],
        "expected": q["expected"],
        "answer": answer,
        "retrieved": [{"source": c.source, "section": c.section, "text": c.text, "score": round(s, 3)}
                      for c, s in retrieved],
        "retrieval_hit": None if not src else any(c.source == src for c, _ in retrieved),
        "accuracy": acc,
        "groundedness": grd,
        "judged": judged,
        "latency_s": round(retrieval_s + resp.latency_s, 4),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": metrics.compute_cost(resp.input_tokens, resp.output_tokens,
                                         settings.price_input_per_1m, settings.price_output_per_1m),
        "error": error,
    }


def summarize(items: list[dict]) -> dict:
    n = len(items)
    lat = sorted(i["latency_s"] for i in items)
    hits = [i["retrieval_hit"] for i in items if i["retrieval_hit"] is not None]
    return {
        "n": n,
        "accuracy": statistics.fmean(i["accuracy"] for i in items) if n else 0.0,
        "groundedness": statistics.fmean(i["groundedness"] for i in items) if n else 0.0,
        "retrieval_hit": statistics.fmean(hits) if hits else None,
        "avg_latency_s": statistics.fmean(lat) if n else 0.0,
        "p95_latency_s": lat[min(n - 1, int(0.95 * n))] if n else 0.0,
        "avg_cost_usd": statistics.fmean(i["cost_usd"] for i in items) if n else 0.0,
        "total_cost_usd": sum(i["cost_usd"] for i in items),
        "avg_input_tokens": statistics.fmean(i["input_tokens"] for i in items) if n else 0.0,
        "avg_output_tokens": statistics.fmean(i["output_tokens"] for i in items) if n else 0.0,
        "errors": sum(1 for i in items if i["error"]),
    }


def run_comparison(
    prompt_names: list[str],
    provider: str = "mock",
    model: str | None = None,
    top_k: int = 3,
    limit: int | None = None,
    use_judge: bool = False,
    settings: Settings | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    test_set_path: Path = TEST_SET_PATH,
    docs_dir: Path = DOCS_DIR,
) -> dict:
    settings = settings or load_settings()
    client = get_client(provider, model)
    retriever = BM25Retriever(load_chunks(docs_dir))
    test_set = load_test_set(test_set_path, limit)
    judge = LLMJudge(client) if use_judge else None

    runs = {}
    for name in prompt_names:
        pipeline = RAGPipeline(retriever, client, load_prompt(name), top_k)
        items = []
        for i, q in enumerate(test_set, 1):
            items.append(evaluate_question(q, pipeline, settings, judge))
            if progress:
                progress(name, i, len(test_set))
            if settings.request_delay and provider != "mock":
                time.sleep(settings.request_delay)
        runs[name] = {"summary": summarize(items), "items": items}

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": provider,
            "model": model_name(client),
            "top_k": top_k,
            "n_questions": len(test_set),
            "judge": use_judge,
            "price_input_per_1m": settings.price_input_per_1m,
            "price_output_per_1m": settings.price_output_per_1m,
        },
        "runs": runs,
    }


def comparison_rows(result: dict) -> list[dict]:
    """Rows like: {'metric': 'Accuracy', 'v1': '78%', 'v2': '91%', 'delta': '+13 pts', 'winner': 'v2'}."""
    names = list(result["runs"])
    rows = []
    for key, label, kind, higher in METRICS:
        vals = {n: result["runs"][n]["summary"].get(key) for n in names}
        row = {"metric": label, **{n: fmt(kind, v) for n, v in vals.items()}}
        valid = {n: v for n, v in vals.items() if v is not None}
        if len(names) >= 2 and len(valid) == len(names):
            d = vals[names[-1]] - vals[names[0]]
            row["delta"] = (f"{d * 100:+.0f} pts" if kind == "pct" else
                            f"{d:+.2f}s" if kind == "sec" else f"{d:+.5f}$")
            best = max(valid, key=valid.get) if higher else min(valid, key=valid.get)
            row["winner"] = "tie" if len(set(round(v, 6) for v in valid.values())) == 1 else best
        rows.append(row)
    return rows


def to_markdown(result: dict) -> str:
    m, names = result["meta"], list(result["runs"])
    lines = [
        f"# LLM Evaluation Report ({m['timestamp']})",
        f"Provider: `{m['provider']}` · Model: `{m['model']}` · top_k={m['top_k']} · "
        f"{m['n_questions']} questions · judge={m['judge']}",
        "",
        "| Metric | " + " | ".join(names) + " | Delta | Winner |",
        "|---|" + "---|" * (len(names) + 2),
    ]
    for r in comparison_rows(result):
        lines.append(f"| {r['metric']} | " + " | ".join(r[n] for n in names)
                     + f" | {r.get('delta', '')} | {r.get('winner', '')} |")
    lines += ["", "Cost is an *equivalent* cost using reference pricing "
              f"(${m['price_input_per_1m']}/1M input, ${m['price_output_per_1m']}/1M output tokens)."]
    return "\n".join(lines) + "\n"


def save_results(result: dict, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["meta"]["timestamp"].replace(":", "-")
    path = out_dir / f"run_{stamp}_{result['meta']['provider']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "latest_report.md").write_text(to_markdown(result), encoding="utf-8")
    return path


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
