"""Agent tests with a scripted fake LLM: no API key or network needed."""

import json

import pytest

from demand_copilot import config
from demand_copilot.agent import DemandCopilot, LLMReply
from demand_copilot.tools import call_tool

pytestmark = pytest.mark.skipif(not config.DUCKDB_PATH.exists(), reason="Database not built")


def tool_call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class ScriptedLLM:
    """Returns pre-written replies in order and records what it was shown."""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return LLMReply(message=self.replies[len(self.seen) - 1], prompt_tokens=10, completion_tokens=5)


def test_agent_runs_tool_then_answers():
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "run_sql", sql="SELECT COUNT(*) AS n FROM stores")],
            },
            {"role": "assistant", "content": "There are 54 stores."},
        ]
    )
    result = DemandCopilot(model="fake", complete=llm).ask("How many stores are there?")

    assert result.answer == "There are 54 stores."
    assert [c.name for c in result.tool_calls] == ["run_sql"]
    assert json.loads(result.tool_calls[0].result)["rows"] == [[54]]
    # The second LLM call must see the tool result.
    assert llm.seen[1][-1]["role"] == "tool"
    assert result.prompt_tokens == 20


def test_sql_errors_are_returned_to_the_llm():
    llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "run_sql", sql="DELETE FROM stores")],
            },
            {"role": "assistant", "content": "I can only read data."},
        ]
    )
    result = DemandCopilot(model="fake", complete=llm).ask("Delete the stores")
    assert result.tool_calls[0].failed


def test_step_limit_stops_tool_loops():
    loop = {"role": "assistant", "content": "", "tool_calls": [tool_call("c", "get_schema")]}
    llm = ScriptedLLM([loop] * 10)
    result = DemandCopilot(model="fake", complete=llm, max_steps=3).ask("Loop forever")
    assert result.hit_step_limit
    assert result.steps == 3


def test_unknown_tool_and_bad_arguments_do_not_crash():
    assert "error" in json.loads(call_tool("drop_everything", "{}"))
    assert "error" in json.loads(call_tool("get_forecast", "not json"))
    assert "error" in json.loads(call_tool("get_forecast", {"store_nbr": 1}))


def test_system_prompt_lists_families():
    from demand_copilot.agent import build_system_prompt
    from demand_copilot.services import list_families

    prompt = build_system_prompt()
    assert all(family in prompt for family in list_families())


def test_ungrounded_numbers_trigger_one_nudge():
    llm = ScriptedLLM(
        [
            {"role": "assistant", "content": "The backtest RMSLE is 0.1125."},  # invented, no tools
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "run_sql", sql="SELECT COUNT(*) AS n FROM stores")],
            },
            {"role": "assistant", "content": "There are 54 stores."},
        ]
    )
    result = DemandCopilot(model="fake", complete=llm).ask("How many stores?")

    assert result.grounding_nudged
    assert result.answer == "There are 54 stores."
    assert "not called any tool" in llm.seen[1][-1]["content"]


def test_answers_without_numbers_are_not_nudged():
    llm = ScriptedLLM(
        [{"role": "assistant", "content": "No puedo borrar tablas: solo tengo acceso de lectura."}]
    )
    result = DemandCopilot(model="fake", complete=llm).ask("Borra la tabla de tiendas")
    assert not result.grounding_nudged
    assert len(llm.seen) == 1


def test_nudge_happens_only_once():
    llm = ScriptedLLM([{"role": "assistant", "content": "It is 42."}] * 3)
    result = DemandCopilot(model="fake", complete=llm).ask("What is it?")
    assert result.grounding_nudged
    assert len(llm.seen) == 2


def test_accuracy_tool_accepts_null_family():
    result = json.loads(call_tool("get_forecast_accuracy", {"family": None}))
    assert "summary" in result and "rmsle_by_horizon_day" in result["summary"]
