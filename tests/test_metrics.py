from llm_eval.metrics import NOT_IN_DOCS, accuracy, groundedness, is_abstention

CTX = "Full-time employees receive 25 paid vacation days per calendar year. The learning budget is €1,200 per year."


def test_accuracy_correct():
    assert accuracy("25 days", "Employees receive 25 annual leave days.") == 1.0


def test_accuracy_wrong_number():
    assert accuracy("25 days", "Employees receive 30 annual leave days.") == 0.0


def test_accuracy_thousands_separator():
    assert accuracy("€1,200 per year", "The budget is 1200 euros per year.") == 1.0


def test_accuracy_ignores_citation_numbers():
    assert accuracy("25 days", "You get 25 days. [policy_v3.md]") == 1.0


def test_unanswerable_requires_abstention():
    assert accuracy(NOT_IN_DOCS, "I don't know based on the provided documents.") == 1.0
    assert accuracy(NOT_IN_DOCS, "The CEO earns 200k.") == 0.0


def test_abstention_is_never_correct_for_answerable():
    assert accuracy("25 days", "I don't know based on the provided documents.") == 0.0


def test_groundedness_supported():
    assert groundedness("Employees receive 25 vacation days.", CTX) == 1.0


def test_groundedness_hallucinated_number():
    assert groundedness("Employees receive 40 vacation days.", CTX) == 0.0


def test_groundedness_partial():
    ans = "Employees receive 25 vacation days. They also get a free company car and a yacht."
    assert groundedness(ans, CTX) == 0.5


def test_groundedness_abstention_and_empty():
    assert groundedness("I don't know based on the provided documents.", CTX) == 1.0
    assert groundedness("", CTX) == 0.0


def test_is_abstention():
    assert is_abstention("Sorry, that is not mentioned in the context.")
    assert not is_abstention("25 days")
