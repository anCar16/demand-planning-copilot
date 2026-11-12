"""Demand planning agent built as a LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

The "agent" node asks the LLM what to do next (through LiteLLM, so any provider works).
The "tools" node runs the requested tools and feeds the results back.

Usage from the terminal:
    uv run python -m demand_copilot.agent "Can I trust the forecast for school supplies?"
"""

from __future__ import annotations

import argparse
import operator
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from demand_copilot import (
    config,  # noqa: F401  (loads the .env file with API keys)
    services,
)
from demand_copilot.tools import TOOL_SCHEMAS, call_tool

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-3.8-flash")
# Comma-separated LiteLLM models tried in order when the main provider is down or rate-limited.
FALLBACK_MODELS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
MAX_STEPS = 8  # LLM calls per question; stops runaway tool loops

SYSTEM_PROMPT = """You are a demand planning copilot for Corporación Favorita, a grocery chain in Ecuador.
You help category managers and planners who do not know SQL understand sales history and forecasts.

Data you can reach through tools:
- Daily sales history by store and product family from 2013-01-01 to 2017-08-15.
- A LightGBM forecast for the next 16 days: 2017-08-16 to 2017-08-31.
- Backtest accuracy of that forecast (2017-07-31 to 2017-08-15).

Rules:
1. Every number in your answer must come from a tool result. Never estimate or invent figures.
   Always call a tool before answering a question about the data, even if you think you know the answer.
2. Before writing SQL, call get_schema unless you already know the schema in this conversation.
3. If a query fails, read the error, fix the SQL and try again.
4. If the question is ambiguous (for example "December" without a year), choose a reasonable
   interpretation and state it in one short sentence.
5. Explain accuracy in business terms: RMSLE is the size of the error, bias_pct is its direction.
   Negative bias means the model under-forecasts, which risks stockouts; positive bias risks overstock.
6. Association is not causation: promotions and holidays correlate with sales but you cannot prove causes.
7. Answer in the same language as the user, concisely, leading with the direct answer.
8. Never count, sum or rank many rows yourself from a tool result: let run_sql do the arithmetic.
9. When comparing periods, stores or families, lead with the totals and the difference (absolute and %),
   then add details only if they help.
"""

# Sent once when the model answers with figures without having called any tool (ungrounded answer).
GROUNDING_NUDGE = (
    "Your answer contains figures, but you have not called any tool in this conversation. "
    "Use the tools to verify every number before answering, or answer without figures."
)


def build_system_prompt() -> str:
    """System prompt plus the valid family names, so the agent does not spend steps looking them up."""
    try:
        families = services.list_families()
    except FileNotFoundError:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\nProduct families (always use these exact English names in tools and SQL; "
        + "translate the user's words, e.g. 'útiles escolares' -> SCHOOL AND OFFICE SUPPLIES):\n"
        + ", ".join(families)
        + "\n"
    )


# ---------- LLM call (injectable, so tests can run without an API key) ----------


@dataclass
class LLMReply:
    message: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_used: str = ""


class LLMUnavailableError(RuntimeError):
    """Raised when the main model and every fallback model failed."""


def _sampling_params(model: str) -> dict:
    # Gemini 3+ deprecates temperature; keep it at 0 elsewhere for reproducible evaluations.
    return {} if model.startswith("gemini/gemini-3") else {"temperature": 0}


COOLDOWN_SECONDS = 300  # how long a failing provider is skipped (circuit breaker)


def litellm_completion(
    model: str,
    fallbacks: list[str] | None = None,
    cooldown_s: float = COOLDOWN_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[list[dict], list[dict]], LLMReply]:
    """Build a completion function for any LiteLLM model string, with provider fallbacks.

    Circuit breaker: when a model fails, it is skipped for cooldown_s seconds so the
    following agent steps go straight to a healthy provider instead of waiting again.
    """
    import litellm  # imported lazily: tests use a fake completion and never load it

    litellm.suppress_debug_info = True
    # Errors where trying another provider makes sense (outages, rate limits, retired models, missing keys).
    switchable = (
        litellm.ServiceUnavailableError,
        litellm.RateLimitError,
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.InternalServerError,
        litellm.NotFoundError,
        litellm.AuthenticationError,
    )
    candidates = [model, *(fallbacks or [])]
    down_until: dict[str, float] = {}

    def complete(messages: list[dict], tools: list[dict]) -> LLMReply:
        now = clock()
        healthy = [m for m in candidates if down_until.get(m, 0) <= now]
        # If every provider is cooling down, try them all anyway rather than failing instantly.
        order = healthy or candidates
        last_resort = len(order) == 1
        errors = []
        for candidate in order:
            try:
                response = litellm.completion(
                    model=candidate,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    # Fail fast when a backup exists; retry harder only on the last option.
                    num_retries=2 if last_resort or candidate == order[-1] else 0,
                    **_sampling_params(candidate),
                )
            except switchable as exc:
                down_until[candidate] = clock() + cooldown_s
                # Keep the provider's message: it says which limit was hit (per minute vs per day).
                detail = " ".join(str(exc).split())[:300]
                errors.append(f"{candidate}: {type(exc).__name__}: {detail}")
                continue
            down_until.pop(candidate, None)
            return _to_reply(response, candidate)
        raise LLMUnavailableError("All models failed -> " + "; ".join(errors))

    return complete


def _to_reply(response, model_used: str) -> LLMReply:
    msg = response.choices[0].message
    message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in msg.tool_calls
        ]
    usage = getattr(response, "usage", None)
    return LLMReply(
        message=message,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model_used=model_used,
    )


# ---------- Graph ----------


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    steps: int
    prompt_tokens: int
    completion_tokens: int
    models_used: Annotated[list[str], operator.add]
    nudged: bool


def needs_grounding(messages: list[dict]) -> bool:
    """True when the last answer has digits but no tool was ever called (likely hallucinated numbers)."""
    last = messages[-1]
    if last.get("tool_calls"):
        return False
    used_tools = any(m.get("role") == "tool" for m in messages)
    return not used_tools and any(ch.isdigit() for ch in last.get("content") or "")


def build_graph(complete: Callable[[list[dict], list[dict]], LLMReply], max_steps: int = MAX_STEPS):
    def agent_node(state: AgentState) -> dict:
        reply = complete(state["messages"], TOOL_SCHEMAS)
        return {
            "messages": [reply.message],
            "steps": state["steps"] + 1,
            "prompt_tokens": state["prompt_tokens"] + reply.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + reply.completion_tokens,
            "models_used": [reply.model_used] if reply.model_used else [],
        }

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for tc in last["tool_calls"]:
            name = tc["function"]["name"]
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": call_tool(name, tc["function"]["arguments"]),
                }
            )
        return {"messages": results}

    def ground_node(state: AgentState) -> dict:
        return {"messages": [{"role": "user", "content": GROUNDING_NUDGE}], "nudged": True}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if state["steps"] >= max_steps:
            return END
        if last.get("tool_calls"):
            return "tools"
        if not state.get("nudged") and needs_grounding(state["messages"]):
            return "ground"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("ground", ground_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "ground": "ground", END: END})
    graph.add_edge("tools", "agent")
    graph.add_edge("ground", "agent")
    return graph.compile()


# ---------- Public interface ----------


@dataclass
class ToolCall:
    name: str
    arguments: str
    result: str

    @property
    def failed(self) -> bool:
        return self.result.startswith('{"error"')


@dataclass
class AgentResult:
    question: str
    answer: str
    model: str
    steps: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    hit_step_limit: bool = False
    models_used: list[str] = field(default_factory=list)
    grounding_nudged: bool = False


class DemandCopilot:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        complete: Callable | None = None,
        max_steps: int = MAX_STEPS,
        fallbacks: list[str] | None = None,
    ):
        self.model = model
        self.max_steps = max_steps
        fallbacks = FALLBACK_MODELS if fallbacks is None else fallbacks
        self.graph = build_graph(complete or litellm_completion(model, fallbacks), max_steps=max_steps)

    def ask(self, question: str) -> AgentResult:
        start = time.perf_counter()
        state = self.graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": question},
                ],
                "nudged": False,
                "steps": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "models_used": [],
            },
            {"recursion_limit": 2 * self.max_steps + 4},
        )
        latency = time.perf_counter() - start
        messages = state["messages"]
        last = messages[-1]
        hit_limit = bool(last.get("tool_calls"))
        answer = last.get("content") or ""
        if hit_limit or not answer:
            answer = (
                answer
                or "I could not finish the analysis within the step limit. Try a more specific question."
            )
        return AgentResult(
            question=question,
            answer=answer,
            model=self.model,
            steps=state["steps"],
            latency_s=round(latency, 2),
            prompt_tokens=state["prompt_tokens"],
            completion_tokens=state["completion_tokens"],
            tool_calls=_collect_tool_calls(messages),
            hit_step_limit=hit_limit,
            models_used=state["models_used"],
            grounding_nudged=state.get("nudged", False),
        )


def _collect_tool_calls(messages: list[dict]) -> list[ToolCall]:
    results = {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"}
    calls = []
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            calls.append(
                ToolCall(tc["function"]["name"], tc["function"]["arguments"], results.get(tc["id"], ""))
            )
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the demand planning copilot a question.")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model string")
    args = parser.parse_args()

    try:
        result = DemandCopilot(model=args.model).ask(args.question)
    except LLMUnavailableError as exc:
        raise SystemExit(
            f"\nNo LLM provider is available right now.\n{exc}\n"
            "Try again in a few minutes or set LLM_FALLBACK_MODELS in .env."
        ) from None
    print("\n" + result.answer + "\n")
    print("-" * 60)
    for i, call in enumerate(result.tool_calls, 1):
        status = "ERROR" if call.failed else "ok"
        print(f"{i}. {call.name}({call.arguments[:120]}) -> {status}")
    used = ", ".join(dict.fromkeys(result.models_used)) or result.model
    print(
        f"model={used} steps={result.steps} latency={result.latency_s}s "
        f"tokens={result.prompt_tokens}+{result.completion_tokens}"
    )


if __name__ == "__main__":
    main()
