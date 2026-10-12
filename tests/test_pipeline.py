from llm_eval.config import DOCS_DIR
from llm_eval.evaluator import comparison_rows, load_test_set, run_comparison, save_results, to_markdown
from llm_eval.retriever import BM25Retriever, load_chunks


def test_chunks_loaded():
    chunks = load_chunks(DOCS_DIR)
    assert len(chunks) >= 10
    assert all(c.text for c in chunks)


def test_retriever_finds_right_document():
    r = BM25Retriever(load_chunks(DOCS_DIR))
    top = r.search("How many vacation days can be carried over?", k=3)
    assert top[0][0].source == "hr_vacation.md"


def test_retrieval_hit_rate_on_test_set():
    r = BM25Retriever(load_chunks(DOCS_DIR))
    qs = [q for q in load_test_set() if q["expected_source"]]
    hits = sum(any(c.source == q["expected_source"] for c, _ in r.search(q["question"], 3)) for q in qs)
    assert hits / len(qs) >= 0.9


def test_test_set_has_20_questions():
    assert len(load_test_set()) == 20


def test_full_run_with_mock(tmp_path):
    result = run_comparison(["v1", "v2"], provider="mock")
    assert set(result["runs"]) == {"v1", "v2"}
    for run in result["runs"].values():
        assert run["summary"]["n"] == 20
        assert run["summary"]["errors"] == 0
    # strict prompt must not be worse on unanswerable questions
    assert result["runs"]["v2"]["summary"]["accuracy"] >= result["runs"]["v1"]["summary"]["accuracy"]
    rows = comparison_rows(result)
    assert rows[0]["metric"] == "Accuracy" and "winner" in rows[0]
    path = save_results(result, tmp_path)
    assert path.exists() and "Accuracy" in to_markdown(result)
