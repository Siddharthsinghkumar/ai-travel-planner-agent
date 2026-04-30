import asyncio

import pytest

import agents.ollama_client as ollama_client
from core.circuit_breaker import CircuitBreakerOpenError


def test_extract_stream_token_or_heartbeat_prefers_visible_content():
    token, kind = ollama_client._extract_stream_token_or_heartbeat(
        {"message": {"content": "hello"}, "done": False}
    )
    assert token == "hello"
    assert kind == "visible"


def test_extract_stream_token_or_heartbeat_accepts_generate_response_shape():
    token, kind = ollama_client._extract_stream_token_or_heartbeat(
        {"response": "world", "done": False}
    )
    assert token == "world"
    assert kind == "visible"


def test_extract_stream_token_or_heartbeat_emits_thinking_liveness():
    token, kind = ollama_client._extract_stream_token_or_heartbeat(
        {"message": {"content": ""}, "thinking": "reasoning...", "done": False}
    )
    assert token == ""
    assert kind == "thinking_heartbeat"


@pytest.mark.asyncio
async def test_streaming_generate_does_not_enforce_extra_total_timeout(monkeypatch):
    async def fake_streaming_call(payload, request_id=None, timeout=30.0):
        # Simulate a stream that takes longer than the caller timeout overall,
        # but still yields chunks normally.
        await asyncio.sleep(0.02)
        yield "a"
        await asyncio.sleep(0.02)
        yield "b"

    monkeypatch.setattr(ollama_client, "_streaming_call", fake_streaming_call)

    stream = await ollama_client.generate(
        prompt="hello",
        stream=True,
        timeout=0.01,
    )
    chunks = []
    async for token in stream:
        chunks.append(token)

    assert "".join(chunks) == "ab"


@pytest.mark.asyncio
async def test_non_streaming_breaker_open_surfaces_as_ollama_error(monkeypatch):
    async def _raise_open(_fn, *, treat_cancelled_as_failure=False):
        raise CircuitBreakerOpenError("Circuit breaker is OPEN")

    monkeypatch.setattr(ollama_client.ollama_breaker, "call", _raise_open)

    with pytest.raises(ollama_client.OllamaError, match="Circuit breaker is open"):
        await ollama_client.generate(prompt="hello", stream=False, timeout=1.0)


@pytest.mark.asyncio
async def test_ollama_client_wrapper_forwards_timeout(monkeypatch):
    captured = {}

    async def fake_generate(
        *,
        prompt,
        system=None,
        model=None,
        temperature=0.2,
        stream=False,
        request_id=None,
        timeout=None,
    ):
        captured.update(
            {
                "prompt": prompt,
                "system": system,
                "model": model,
                "temperature": temperature,
                "stream": stream,
                "request_id": request_id,
                "timeout": timeout,
            }
        )
        return "ok"

    monkeypatch.setattr(ollama_client, "generate", fake_generate)
    client = ollama_client.OllamaClient()

    result = await client.generate(
        prompt="hello",
        stream=False,
        request_id="test-req",
        timeout=7.5,
    )

    assert result == "ok"
    assert captured["timeout"] == 7.5
    assert captured["request_id"] == "test-req"


@pytest.mark.asyncio
async def test_non_streaming_breaker_cancellation_is_neutral_by_default(monkeypatch):
    seen = {}

    async def _cancelled_call(_fn, *, treat_cancelled_as_failure=False):
        seen["treat_cancelled_as_failure"] = treat_cancelled_as_failure
        raise asyncio.CancelledError()

    monkeypatch.setattr(ollama_client.ollama_breaker, "call", _cancelled_call)

    with pytest.raises(asyncio.CancelledError):
        await ollama_client.generate(prompt="hello", stream=False, timeout=1.0)

    assert seen.get("treat_cancelled_as_failure") is False


def test_resolve_ollama_thinking_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("OLLAMA_THINKING_MODE", raising=False)
    assert ollama_client._resolve_ollama_thinking_mode() == "auto"
    monkeypatch.setenv("OLLAMA_THINKING_MODE", "invalid")
    assert ollama_client._resolve_ollama_thinking_mode() == "auto"


@pytest.mark.asyncio
async def test_generate_sets_think_option_when_disabled(monkeypatch):
    captured = {}

    async def fake_non_streaming(payload, request_id=None, timeout=30.0):
        captured["payload"] = payload
        return "ok"

    monkeypatch.setenv("OLLAMA_THINKING_MODE", "disable")
    monkeypatch.setattr(ollama_client, "_non_streaming_call", fake_non_streaming)

    result = await ollama_client.generate(prompt="hello", stream=False, timeout=2.0)

    assert result == "ok"
    assert captured["payload"]["options"]["think"] is False
