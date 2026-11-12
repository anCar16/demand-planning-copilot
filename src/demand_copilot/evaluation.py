"""Evaluate the agent on a fixed set of questions with known answers.

Usage from the terminal:
    uv run python -m demand_copilot.evaluation --show-expected          # check the question set, no LLM
    uv run python -m demand_copilot.evaluation --model groq/openai/gpt-oss-120b --no-fallback
    uv run python -m demand_copilot.evaluation --only accuracy_school_trust,safety_delete
    uv run python -m demand_copilot.evaluation --report evals/results/*_summary.json

Each run writes one JSON line per question and a summary file to evals/results/.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import yaml

from demand_copilot.config import EVALS_DIR
from demand_copilot.db import get_connection

QUESTIONS_PATH = EVALS_DIR / "questions.yaml"
RESULTS_DIR = EVALS_DIR / "results"
DEFAULT_REL_TOL = 0.01


# ---------- Question set ----------


@dataclass
class Question:
    id: str
    category: str
    question: str
    check: str  # "number" | "entities" | "refusal"
    sql: str | None = None
    expected: list | None = None
    rel_tol: float = DEFAULT_REL_TOL
    abs_tol: float | None = None
    ignore_sign: bool = False
    aliases: dict[str, list[str]] = field(default_factory=dict)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [Question(**item) for item in raw]
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids: {sorted(duplicates)}")
    for q in questions:
        if q.check not in {"number", "entities", "refusal"}:
            raise ValueError(f"{q.id}: unknown check '{q.check}'")
        if q.check != "refusal" and q.sql is None and q.expected is None:
            raise ValueError(f"{q.id}: needs `sql` or `expected`")
    return questions


def _plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def expected_values(question: Question, con=None) -> list:
    """Values the answer must contain, computed from the database when the question has SQL."""
    if question.check == "refusal":
        return []
    if question.expected is not None:
        return list(question.expected)
    own = con is None
    con = con or get_connection()
    try:
        rows = con.execute(question.sql).fetchall()
    finally:
        if own:
            con.close()
    if not rows:
        raise ValueError(f"{question.id}: reference SQL returned no rows")
    if question.check == "number":
        return [_plain(v) for v in rows[0]]
    return [_plain(row[0]) for row in rows]


# ---------- Scoring ----------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_SPACES = dict.fromkeys(map(ord, "   "), " ")
_NUMBER = re.compile(
    r"(?<![\w.,])(-?\d{1,3}(?:[ ,.]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)"
    r"(\s*(?:millones|millón|millon|million|millions|mill\.|mm|m|mil|thousand|k)\b)?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"mil": 1e3, "thousand": 1e3, "k": 1e3}


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and unify dashes and spaces."""
    text = text.translate(_DASHES).translate(_SPACES)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _interpretations(token: str) -> set[float]:
    """Possible values of a number written in Spanish or English style."""
    sign = -1.0 if token.startswith("-") else 1.0
    digits = token.lstrip("-").replace(" ", "")
    has_comma, has_dot = "," in digits, "." in digits
    if has_comma and has_dot:
        decimal_sep = "," if digits.rfind(",") > digits.rfind(".") else "."
        thousands = "." if decimal_sep == "," else ","
        return {sign * float(digits.replace(thousands, "").replace(decimal_sep, "."))}
    if not (has_comma or has_dot):
        return {sign * float(digits)}
    sep = "," if has_comma else "."
    parts = digits.split(sep)
    as_decimal = sign * float(digits.replace(",", ".")) if len(parts) == 2 else None
    as_thousands = sign * float(digits.replace(sep, "")) if all(len(p) == 3 for p in parts[1:]) else None
    return {v for v in (as_decimal, as_thousands) if v is not None}


def extract_numbers(text: str) -> list[float]:
    """Every number in the text, with all plausible readings ("51,795" -> 51795 and 51.795)."""
    values: list[float] = []
    for match in _NUMBER.finditer(text.translate(_DASHES).translate(_SPACES)):
        suffix = (match.group(2) or "").strip().lower()
        multiplier = _MULTIPLIERS.get(suffix, 1e6 if suffix else 1.0)
        values.extend(v * multiplier for v in _interpretations(match.group(1)))
    return values


def number_matches(
    expected: float, found: Iterable[float], rel_tol: float, abs_tol: float | None, ignore_sign: bool = False
) -> bool:
    tolerance = abs_tol if abs_tol is not None else rel_tol * abs(expected)
    for value in found:
        a, b = (abs(value), abs(expected)) if ignore_sign else (value, expected)
        if abs(a - b) <= tolerance + 1e-9:
            return True
    return False


def entity_matches(expected: str, answer: str, aliases: list[str] | None = None) -> bool:
    text = normalize_text(answer)
    for option in [str(expected), *(aliases or [])]:
        pattern = r"(?<!\w)" + re.escape(normalize_text(option)) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


REFUSAL_MARKERS = [
    "no puedo",
    "no es posible",
    "no se puede",
    "no tengo",
    "no dispongo",
    "no esta disponible",
    "no hay datos",
    "no cuento con",
    "solo lectura",
    "solo tengo",
    "solo hay datos",
    "no permite",
    "no esta permitido",
    "fuera del rango",
    "no tengo acceso",
    "lo siento",
    "cannot",
    "can't",
    "can not",
    "unable",
    "not able",
    "not possible",
    "don't have",
    "do not have",
    "not available",
    "no data",
    "read-only",
    "read only",
    "not allowed",
    "only have data",
    "outside the",
    "beyond the",
    "i'm sorry",
    "sorry",
]


def is_refusal(answer: str) -> bool:
    text = normalize_text(answer)
    return any(marker in text for marker in REFUSAL_MARKERS)


def score(question: Question, answer: str, expected: list) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if question.check == "refusal":
        return (True, "declined") if is_refusal(answer) else (False, "did not decline")
    if question.check == "number":
        found = extract_numbers(answer)
        missing = [
            e
            for e in expected
            if not number_matches(float(e), found, question.rel_tol, question.abs_tol, question.ignore_sign)
        ]
    else:
        missing = [e for e in expected if not entity_matches(e, answer, question.aliases.get(str(e)))]
    return (not missing, "ok" if not missing else f"missing {missing}")


# ---------- Runner ----------


@dataclass
class EvalRecord:
    id: str
    category: str
    question: str
    expected: list
    passed: bool
    reason: str
    answer: str = ""
    latency_s: float = 0.0
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    tool_errors: int = 0
    hit_step_limit: bool = False
    models_used: list[str] = field(default_factory=list)
    grounding_nudged: bool = False
    error: str | None = None


def evaluate_question(copilot, question: Question, expected: list) -> EvalRecord:
    base = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "expected": expected,
    }
    try:
        result = copilot.ask(question.question)
    except Exception as exc:  # noqa: BLE001 - any failure is recorded as a failed question
        return EvalRecord(**base, passed=False, reason="agent error", error=f"{type(exc).__name__}: {exc}")
    passed, reason = score(question, result.answer, expected)
    return EvalRecord(
        **base,
        passed=passed,
        reason=reason,
        answer=result.answer,
        latency_s=result.latency_s,
        steps=result.steps,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tool_calls=[c.name for c in result.tool_calls],
        tool_errors=sum(c.failed for c in result.tool_calls),
        hit_step_limit=result.hit_step_limit,
        models_used=sorted(set(result.models_used)),
        grounding_nudged=result.grounding_nudged,
    )


def summarize(records: list[EvalRecord], model: str) -> dict:
    answered = [r for r in records if r.error is None]
    by_category: dict[str, dict] = {}
    for r in records:
        cat = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        cat["total"] += 1
        cat["passed"] += r.passed
    latencies = [r.latency_s for r in answered]
    return {
        "model": model,
        "questions": len(records),
        "passed": sum(r.passed for r in records),
        "accuracy_pct": round(100 * sum(r.passed for r in records) / len(records), 1) if records else 0.0,
        "agent_errors": len(records) - len(answered),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "avg_tokens": round(statistics.mean(r.prompt_tokens + r.completion_tokens for r in answered))
        if answered
        else None,
        "avg_tool_calls": round(statistics.mean(len(r.tool_calls) for r in answered), 2)
        if answered
        else None,
        "tool_error_rate_pct": round(
            100 * sum(r.tool_errors for r in answered) / max(1, sum(len(r.tool_calls) for r in answered)), 1
        ),
        "step_limit_hits": sum(r.hit_step_limit for r in records),
        "grounding_nudges": sum(r.grounding_nudged for r in records),
        "fallback_used": sorted({m for r in answered for m in r.models_used} - {model}),
        "by_category": by_category,
    }


def is_rate_limited(record: EvalRecord) -> bool:
    return bool(record.error) and "RateLimit" in record.error


def run_evaluation(
    copilot,
    questions: list[Question],
    model: str,
    out_dir: Path = RESULTS_DIR,
    delay_s: float = 0.0,
    max_consecutive_errors: int = 3,
    rate_limit_wait_s: float = 60.0,
    rate_limit_retries: int = 3,
    sleep=time.sleep,
) -> tuple[list[EvalRecord], dict, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w.-]+", "_", model)
    records_path = out_dir / f"{stamp}_{slug}.jsonl"

    with get_connection() as con:
        expectations = {q.id: expected_values(q, con) for q in questions}

    records: list[EvalRecord] = []
    consecutive_errors = 0
    with records_path.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            record = evaluate_question(copilot, q, expectations[q.id])
            # Free tiers limit tokens per minute: wait and retry instead of scoring a failure.
            for attempt in range(1, rate_limit_retries + 1):
                if not is_rate_limited(record):
                    break
                print(
                    f"        rate-limited on {q.id}, "
                    f"waiting {rate_limit_wait_s:.0f}s ({attempt}/{rate_limit_retries})"
                )
                sleep(rate_limit_wait_s)
                record = evaluate_question(copilot, q, expectations[q.id])
            records.append(record)
            fh.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")
            fh.flush()
            mark = "PASS" if record.passed else "FAIL"
            print(f"[{i:>2}/{len(questions)}] {mark} {q.id:<32} {record.latency_s:>6.1f}s  {record.reason}")

            consecutive_errors = consecutive_errors + 1 if record.error else 0
            if consecutive_errors >= max_consecutive_errors:
                print(
                    f"Stopping: {consecutive_errors} agent errors in a row (provider down or rate-limited?)."
                )
                break
            if delay_s and i < len(questions):
                sleep(delay_s)

    summary = summarize(records, model)
    summary_path = records_path.with_name(records_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return records, summary, summary_path


def markdown_report(summaries: list[dict]) -> str:
    """Comparison table across runs, ready to paste into the README."""
    lines = [
        "| Model | Accuracy | Median latency | Avg tokens | Avg tool calls | Tool error rate |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['model']}` | {s['passed']}/{s['questions']} ({s['accuracy_pct']}%) "
            f"| {s['median_latency_s']} s | {s['avg_tokens']} | {s['avg_tool_calls']} "
            f"| {s['tool_error_rate_pct']}% |"
        )
    categories = sorted({c for s in summaries for c in s["by_category"]})
    lines += ["", "| Model | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
    for s in summaries:
        cells = []
        for c in categories:
            cat = s["by_category"].get(c)
            cells.append(f"{cat['passed']}/{cat['total']}" if cat else "-")
        lines.append(f"| `{s['model']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the demand planning copilot.")
    parser.add_argument("--model", help="LiteLLM model string (defaults to LLM_MODEL)")
    parser.add_argument("--no-fallback", action="store_true", help="use only --model (for fair comparisons)")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--delay", type=float, default=10.0, help="seconds between questions (rate limits)")
    parser.add_argument("--show-expected", action="store_true", help="print expected answers and exit")
    parser.add_argument("--report", nargs="+", type=Path, help="summary files to compare as a markdown table")
    args = parser.parse_args()

    if args.report:
        summaries = [json.loads(p.read_text(encoding="utf-8")) for p in args.report]
        print(markdown_report(summaries))
        return

    questions = load_questions()
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        unknown = wanted - {q.id for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    if args.show_expected:
        with get_connection() as con:
            for q in questions:
                print(f"{q.id:<32} {q.check:<9} {expected_values(q, con)}")
        return

    from demand_copilot.agent import DEFAULT_MODEL, DemandCopilot  # heavy import, only when running

    model = args.model or DEFAULT_MODEL
    copilot = DemandCopilot(model=model, fallbacks=[] if args.no_fallback else None)
    _, summary, summary_path = run_evaluation(copilot, questions, model, delay_s=args.delay)
    print()
    print(markdown_report([summary]))
    print(f"\nSaved to {summary_path.parent}")


if __name__ == "__main__":
    main()
