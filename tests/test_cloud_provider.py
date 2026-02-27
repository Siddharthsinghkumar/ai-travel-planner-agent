# tests/test_cloud_provider.py
import pytest
import types

import agents.cloud_llm as cloud_llm

class FakeAdapter:
    def __init__(self, name, response=None, raise_on_call=False):
        self.provider = name
        self._response = response
        self._raise = raise_on_call

    async def create_completion(self, model, messages, temperature, max_tokens, timeout):
        if self._raise:
            raise RuntimeError(f"{self.provider} failure")
        class Choice:
            def __init__(self, text):
                self.message = types.SimpleNamespace(content=text)
        class Resp:
            def __init__(self, text):
                self.choices = [Choice(text)]
                self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return Resp(self._response)

    async def open_stream(self, model, messages, temperature, max_tokens, timeout):
        # return an async generator that yields one chunk then stops
        async def gen():
            if self._raise:
                raise RuntimeError("open_stream failed")
            class FakeDelta:
                def __init__(self, content): self.content = content
            class FakeChoice:
                def __init__(self, delta): self.delta = delta
            class FakeChunk:
                def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
            yield FakeChunk(self._response)
        return gen()

@pytest.mark.asyncio
async def test_generate_prefers_first_provider():
    # gemini first, openai second
    gem = FakeAdapter("gemini", response="gemini-ok")
    oai = FakeAdapter("openai", response="openai-ok")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    res = await cloud_llm.generate(prompt="hi")
    assert "gemini-ok" in res

@pytest.mark.asyncio
async def test_generate_falls_back_to_openai_on_gemini_error():
    gem = FakeAdapter("gemini", raise_on_call=True)
    oai = FakeAdapter("openai", response="openai-ok")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    res = await cloud_llm.generate(prompt="hi")
    assert "openai-ok" in res

@pytest.mark.asyncio
async def test_generate_stream_fallback_before_first_token():
    gem = FakeAdapter("gemini", raise_on_call=True)
    oai = FakeAdapter("openai", response="openai-stream")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    # generate_stream returns an async generator — collect text
    agen = cloud_llm.generate_stream(prompt="stream me")
    collected = []
    async for chunk in agen:
        collected.append(chunk)
    joined = "".join(collected)
    assert "openai-stream" in joined