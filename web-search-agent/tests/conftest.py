"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent import AgentTurnResult
from app.config import Settings
from app.database import Database
from app.exceptions import (
    AgentTimeoutError,
    ModelFailedError,
    ModelUnavailableError,
    OllamaUnavailableError,
    SearchFailedError,
    SearchRateLimitError,
    ToolCallingUnsupportedError,
)
from app.main import create_app
from app.schemas import Source


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=str(tmp_path / "test.db"),
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3.2:3b",
        max_message_length=200,
        conversation_history_limit=6,
        agent_timeout_seconds=5.0,
    )


@pytest.fixture
async def app(settings: Settings):
    application = create_app(settings)
    database = Database(settings.resolved_database_path())
    await database.initialize()
    application.state.db = database
    application.state.busy_guard = asyncio.Lock()
    application.state.busy_conversations = set()

    async def fake_run_turn(message: str, history, _settings, **_kwargs) -> AgentTurnResult:
        if message.startswith("SEARCH:"):
            return AgentTurnResult(
                answer="The score is 2-1 according to search snippets. [1]",
                sources=[
                    Source(title="Match report", url="https://example.com/match"),
                ],
            )
        if message == "FAIL_MODEL":
            raise ModelFailedError("The local model failed to complete the reply.")
        if message == "FAIL_OLLAMA":
            raise OllamaUnavailableError("Ollama is not reachable. Start Ollama and retry.")
        if message == "FAIL_MODEL_MISSING":
            raise ModelUnavailableError("Model llama3.2:3b was not found.")
        if message == "FAIL_TOOLS":
            raise ToolCallingUnsupportedError(
                "The selected model does not support tool calling. Choose llama3.2:3b."
            )
        if message == "FAIL_SEARCH":
            raise SearchFailedError("Web search is currently unavailable.")
        if message == "FAIL_RATE":
            raise SearchRateLimitError("Web search was rate-limited. Wait a moment and try again.")
        if message == "FAIL_TIMEOUT":
            raise AgentTimeoutError("The assistant timed out before finishing a reply.")
        return AgentTurnResult(answer=f"Direct answer for: {message}", sources=[])

    application.state.run_turn = fake_run_turn
    yield application
    await database.close()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as async_client:
        yield async_client
