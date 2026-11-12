import json

import pytest

from demand_copilot import config
from demand_copilot.agent import DemandCopilot, LLMReply
from demand_copilot.evaluation import (
    Question,
    entity_matches,
    expected_values,
    extract_numbers,
    is_refusal,
    load_questions,
    markdown_report,
    run_evaluation,
    score,
)

# Real answer from the agent (note the non-breaking hyphen in "‑81.8 %").
REAL_ANSWER = """No, el pronóstico para **útiles escolares** no es confiable.

- **RMSLE:** 1.054 → el error es muy alto (cuanto más cerca de 0, mejor).
- **Bias %:** ‑81.8 % → el modelo subestima la demanda en casi un 82 %."""


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("vendió 51,795 unidades", 51795),
        ("vendió 51.795 unidades", 51795),
        ("vendió 51 795 unidades", 51795),
        ("RMSLE de 1,054", 1.054),
        ("about 12.45 million", 12_450_000),
        ("unos 12,5 millones", 12_500_000),
        ("cerca de 47 mil unidades", 47_000),
        ("total of 12,453,264 units", 12_453_264),
        ("bias of −81.8 %", -81.8),
    ],
)
def test_numbers_in_spanish_and_english_formats(text, value):
    assert any(abs(v - value) < 1e-6 for v in extract_numbers(text))


def test_real_answer_passes_number_check():
    q = Question(id="q", category="accuracy", question="?", check="number", ignore_sign=True)
    passed, _ = score(q, REAL_ANSWER, [1.0539839235075008, -81.78847554783768])
    assert passed


def test_wrong_number_fails():
    q = Question(id="q", category="lookup", question="?", check="number", abs_tol=0)
    assert not score(q, "The chain has 55 stores.", [54])[0]
    assert score(q, "La cadena tiene 54 tiendas.", [54])[0]


def test_entities_use_word_boundaries_accents_and_aliases():
    assert entity_matches("GROCERY I", "Top families: Grocery I, Beverages")
    assert not entity_matches("GROCERY I", "Top families: Grocery II, Beverages")
    assert entity_matches("Sunday", "El día con más ventas es el domingo", ["domingo"])
    assert entity_matches("Pichincha", "La provincia es PICHINCHA.")


def test_refusal_detection():
    assert is_refusal("No puedo modificar datos: tengo acceso de solo lectura.")
    assert is_refusal("I can't delete tables, I only have read access.")
    assert not is_refusal("Done, the stores table has been deleted.")


def test_question_set_is_valid():
    questions = load_questions()
    assert len(questions) >= 30
    assert {"lookup", "forecast", "accuracy", "safety"} <= {q.category for q in questions}


def test_markdown_report_has_one_row_per_model():
    summary = {
        "model": "m",
        "questions": 2,
        "passed": 1,
        "accuracy_pct": 50.0,
        "median_latency_s": 1.0,
        "avg_tokens": 100,
        "avg_tool_calls": 1.0,
        "tool_error_rate_pct": 0.0,
        "by_category": {"lookup": {"passed": 1, "total": 2}},
    }
    report = markdown_report([summary, {**summary, "model": "n"}])
    assert "`m`" in report and "`n`" in report and "1/2" in report


needs_db = pytest.mark.skipif(not config.DUCKDB_PATH.exists(), reason="Database not built")


@needs_db
def test_every_reference_query_runs():
    for q in load_questions():
        values = expected_values(q)
        assert q.check == "refusal" or values, q.id


@needs_db
def test_runner_with_scripted_llm(tmp_path):
    def tool_call(name, **arguments):
        return {
            "id": "c1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    replies = iter(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("run_sql", sql="SELECT COUNT(*) FROM stores")],
            },
            {"role": "assistant", "content": "The chain has 54 stores."},
            {
                "role": "assistant",
                "content": "Lo siento, no puedo borrar tablas: solo tengo acceso de lectura.",
            },
        ]
    )

    def fake_llm(messages, tools):
        return LLMReply(message=next(replies), prompt_tokens=10, completion_tokens=5, model_used="fake")

    questions = [q for q in load_questions() if q.id in {"lookup_store_count", "safety_delete"}]
    records, summary, summary_path = run_evaluation(
        DemandCopilot(model="fake", complete=fake_llm), questions, "fake", out_dir=tmp_path
    )

    assert [r.passed for r in records] == [True, True]
    assert summary["accuracy_pct"] == 100.0
    assert summary["by_category"]["safety"] == {"passed": 1, "total": 1}
    assert summary_path.exists()
    assert (
        len(
            summary_path.with_name(summary_path.name.replace("_summary.json", ".jsonl"))
            .read_text()
            .splitlines()
        )
        == 2
    )


@needs_db
def test_rate_limited_question_is_retried(tmp_path):
    from demand_copilot.agent import LLMUnavailableError

    calls = {"n": 0}

    def flaky_llm(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMUnavailableError("All models failed -> groq/openai/gpt-oss-120b: RateLimitError")
        return LLMReply(
            message={"role": "assistant", "content": "The chain has 54 stores."}, model_used="fake"
        )

    waits = []
    questions = [q for q in load_questions() if q.id == "lookup_store_count"]
    records, summary, _ = run_evaluation(
        DemandCopilot(model="fake", complete=flaky_llm),
        questions,
        "fake",
        out_dir=tmp_path,
        rate_limit_wait_s=60,
        sleep=waits.append,
    )

    assert records[0].passed and records[0].error is None
    assert waits == [60]
    assert summary["agent_errors"] == 0
