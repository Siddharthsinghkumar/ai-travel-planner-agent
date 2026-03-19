import pytest

from api.app import app


@pytest.mark.asyncio
async def test_lifespan_tolerates_legacy_llm_client_init_failure(monkeypatch):
    async def fail_init():
        raise ValueError("OPENAI_API_KEY missing")

    monkeypatch.setattr("api.app.init_llm_client", fail_init)

    async with app.router.lifespan_context(app):
        assert app.state.startup_complete is True
