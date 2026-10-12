"""Streamlit UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from llm_eval import evaluator
from llm_eval.config import PROVIDERS, RESULTS_DIR, TEST_SET_PATH, load_settings
from llm_eval.llm import LLMError
from llm_eval.prompts import list_prompts, load_prompt

st.set_page_config(page_title="LLM Evaluation Platform", page_icon="🧪", layout="wide")
st.title("🧪 LLM Evaluation Platform")
st.caption("RAG evaluation suite: accuracy · groundedness · latency · cost — compare prompt versions side by side.")

settings = load_settings()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("AI provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k]["label"])
    model = st.text_input("Model (optional)", placeholder=PROVIDERS[provider]["default_model"],
                          disabled=provider == "mock") or None
    prompts = st.multiselect("Prompt versions to compare", list_prompts(), default=["v1", "v2"])
    top_k = st.slider("Retrieved chunks (top_k)", 1, 6, 3)
    n_total = len(evaluator.load_test_set())
    limit = st.slider("Number of test questions", 1, n_total, n_total)
    use_judge = st.checkbox("LLM-as-judge (extra calls)", value=False,
                            help="Uses the same model to grade accuracy/groundedness. Falls back to heuristics on parse errors.")
    run = st.button("▶ Run evaluation", type="primary", disabled=not prompts)
    if provider == "mock":
        st.info("Mock = offline extractive baseline. It validates the pipeline; its scores say nothing about real LLMs.")

if run:
    bar = st.progress(0.0, text="Starting…")
    total_steps = len(prompts) * limit
    state = {"done": 0}

    def progress(name, i, n):
        state["done"] += 1
        bar.progress(state["done"] / total_steps, text=f"Prompt {name}: question {i}/{n}")

    try:
        st.session_state["result"] = evaluator.run_comparison(
            prompts, provider, model, top_k, limit, use_judge, settings, progress)
        path = evaluator.save_results(st.session_state["result"])
        st.success(f"Done. Saved to {path.name}")
    except (LLMError, FileNotFoundError) as e:
        st.error(str(e))
    bar.empty()

tab_cmp, tab_q, tab_hist, tab_data = st.tabs(["📊 Compare prompts", "🔍 Question explorer", "🕘 History", "📚 Test set & prompts"])
result = st.session_state.get("result")

# ---------------------------------------------------------------- compare
with tab_cmp:
    if not result:
        st.info("Pick a provider in the sidebar and click **Run evaluation** (or load a past run from History).")
    else:
        m, names = result["meta"], list(result["runs"])
        st.write(f"**{m['provider']}** · `{m['model']}` · top_k={m['top_k']} · {m['n_questions']} questions · judge={m['judge']}")
        rows = evaluator.comparison_rows(result)
        st.dataframe(pd.DataFrame(rows).rename(columns={"metric": "Metric"}), hide_index=True)

        c1, c2, c3 = st.columns(3)
        quality = pd.DataFrame({n: {"Accuracy": result["runs"][n]["summary"]["accuracy"] * 100,
                                    "Groundedness": result["runs"][n]["summary"]["groundedness"] * 100}
                                for n in names})
        c1.subheader("Quality (%)"); c1.bar_chart(quality)
        c2.subheader("Avg latency (s)")
        c2.bar_chart(pd.Series({n: result["runs"][n]["summary"]["avg_latency_s"] for n in names}))
        c3.subheader("Cost / question ($)")
        c3.bar_chart(pd.Series({n: result["runs"][n]["summary"]["avg_cost_usd"] for n in names}))
        st.caption(f"Cost = equivalent cost at ${m['price_input_per_1m']}/1M input and ${m['price_output_per_1m']}/1M output tokens "
                   "(free providers cost $0 in reality).")
        st.download_button("⬇ Download Markdown report", evaluator.to_markdown(result), "report.md")

# ---------------------------------------------------------------- explorer
with tab_q:
    if not result:
        st.info("Run an evaluation first.")
    else:
        names = list(result["runs"])
        view = pd.DataFrame([
            {"prompt": n, "id": i["id"], "category": i["category"], "question": i["question"],
             "expected": i["expected"], "accuracy": i["accuracy"], "groundedness": round(i["groundedness"], 2),
             "latency_s": round(i["latency_s"], 2), "cost_usd": round(i["cost_usd"], 6)}
            for n in names for i in result["runs"][n]["items"]])
        only_fail = st.checkbox("Only failures (accuracy 0 or groundedness < 1)")
        if only_fail:
            view = view[(view.accuracy < 1) | (view.groundedness < 1)]
        st.dataframe(view, hide_index=True)

        st.divider()
        ids = [i["id"] for i in result["runs"][names[0]]["items"]]
        qid = st.selectbox("Inspect a question", ids,
                           format_func=lambda x: f"{x} — " + next(i["question"] for i in result["runs"][names[0]]["items"] if i["id"] == x))
        cols = st.columns(len(names))
        for col, n in zip(cols, names):
            it = next(i for i in result["runs"][n]["items"] if i["id"] == qid)
            with col:
                st.markdown(f"### Prompt `{n}`")
                st.markdown(f"**Question:** {it['question']}")
                st.markdown(f"**Expected:** {it['expected']}")
                st.markdown(f"**AI answer:** {it['answer'] or '_(empty)_'}")
                if it["error"]:
                    st.error(it["error"])
                a, b = st.columns(2)
                a.metric("Accuracy", f"{it['accuracy'] * 100:.0f}%")
                b.metric("Groundedness", f"{it['groundedness'] * 100:.0f}%")
                c, d = st.columns(2)
                c.metric("Latency", f"{it['latency_s']:.2f}s")
                d.metric("Cost", f"${it['cost_usd']:.5f}")
                with st.expander(f"Retrieved documents ({len(it['retrieved'])})"):
                    for r in it["retrieved"]:
                        st.markdown(f"**{r['source']} › {r['section']}** (BM25 {r['score']})")
                        st.write(r["text"])

# ---------------------------------------------------------------- history
with tab_hist:
    files = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    if not files:
        st.info("No saved runs yet.")
    else:
        pick = st.selectbox("Saved runs", files, format_func=lambda p: p.name)
        if st.button("Load this run"):
            st.session_state["result"] = evaluator.load_results(pick)
            st.rerun()

# ---------------------------------------------------------------- data
with tab_data:
    st.subheader("Test set")
    st.dataframe(pd.DataFrame(evaluator.load_test_set()), hide_index=True)
    st.caption(f"Edit `{TEST_SET_PATH.relative_to(TEST_SET_PATH.parents[1])}` to add questions. "
               "Use `NOT_IN_DOCS` as expected answer for questions the documents can't answer (hallucination test).")
    st.subheader("Prompt versions")
    for name in list_prompts():
        p = load_prompt(name)
        with st.expander(f"{p.name} — {p.description}"):
            st.code(f"SYSTEM:\n{p.system}\n\nUSER:\n{p.user}")
