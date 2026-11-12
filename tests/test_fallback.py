"""Provider fallback tests: litellm.completion is replaced, so no network or API key is used."""

from types import SimpleNamespace

import pytest

litellm = pytest.importorskip("litellm")

from demand_copilot.agent import LLMUnavailableError, litellm_completion  # noqa: E402


def fake_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def overloaded(model: str):
    return litellm.ServiceUnavailableError(message="high demand", llm_provider="gemini", model=model)


def test_falls_back_when_main_model_is_overloaded(monkeypatch):
    calls = []

    def fake_completion(model, **kwargs):
        calls.append((model, kwargs))
        if model.startswith("gemini/"):
            raise overloaded(model)
        return fake_response("ok from backup")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    complete = litellm_completion("gemini/gemini-3.8-flash", ["groq/openai/gpt-oss-120b"])
    reply = complete([{"role": "user", "content": "hi"}], [])

    assert reply.message["content"] == "ok from backup"
    assert reply.model_used == "groq/openai/gpt-oss-120b"
    assert [m for m, _ in calls] == ["gemini/gemini-3.8-flash", "groq/openai/gpt-oss-120b"]
    # Gemini 3 gets no temperature (deprecated); other providers get temperature=0.
    assert "temperature" not in calls[0][1]
    assert calls[1][1]["temperature"] == 0


def test_raises_clear_error_when_every_model_fails(monkeypatch):
    def always_down(model, **kwargs):
        raise overloaded(model)

    monkeypatch.setattr(litellm, "completion", always_down)
    complete = litellm_completion("gemini/gemini-3.8-flash", ["groq/openai/gpt-oss-120b"])
    with pytest.raises(LLMUnavailableError, match="ServiceUnavailableError"):
        complete([{"role": "user", "content": "hi"}], [])


def test_unexpected_errors_are_not_hidden(monkeypatch):
    def broken(model, **kwargs):
        raise ValueError("bug in our request")

    monkeypatch.setattr(litellm, "completion", broken)
    complete = litellm_completion("gemini/gemini-3.8-flash", ["groq/openai/gpt-oss-120b"])
    with pytest.raises(ValueError):
        complete([{"role": "user", "content": "hi"}], [])


def test_failed_provider_is_skipped_during_cooldown(monkeypatch):
    calls = []
    now = [0.0]

    def fake_completion(model, **kwargs):
        calls.append(model)
        if model.startswith("gemini/"):
            raise overloaded(model)
        return fake_response("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    complete = litellm_completion(
        "gemini/gemini-3.8-flash", ["groq/openai/gpt-oss-120b"], cooldown_s=300, clock=lambda: now[0]
    )
    messages = [{"role": "user", "content": "hi"}]

    complete(messages, [])  # Gemini fails -> Groq
    complete(messages, [])  # Gemini is cooling down -> straight to Groq
    assert calls == ["gemini/gemini-3.8-flash", "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-120b"]

    now[0] = 301  # cooldown over -> Gemini gets another chance
    complete(messages, [])
    assert calls[3] == "gemini/gemini-3.8-flash"


def test_main_model_fails_fast_when_backup_exists(monkeypatch):
    retries = {}

    def fake_completion(model, **kwargs):
        retries[model] = kwargs["num_retries"]
        if model.startswith("gemini/"):
            raise overloaded(model)
        return fake_response("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    litellm_completion("gemini/gemini-3.8-flash", ["groq/openai/gpt-oss-120b"])(
        [{"role": "user", "content": "hi"}], []
    )
    assert retries == {"gemini/gemini-3.8-flash": 0, "groq/openai/gpt-oss-120b": 2}
