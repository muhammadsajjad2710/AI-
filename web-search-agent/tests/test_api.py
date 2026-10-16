"""API tests using a mocked agent."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.agent import AgentTurnResult
from app.schemas import Source


async def _create_conversation(client) -> str:
    response = await client.post("/api/conversations")
    assert response.status_code == 200
    return response.json()["conversation_id"]


@pytest.mark.asyncio
async def test_health_reports_backend_status(client, monkeypatch):
    async def fake_status(_settings):
        from app.agent import OllamaStatus

        return OllamaStatus(
            available=True,
            model_available=True,
            supports_tools=True,
            detail="ok",
        )

    monkeypatch.setattr("app.main.check_ollama_status", fake_status)
    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ollama"]["model"] == "llama3.2:3b"
    assert payload["ollama"]["supports_tools"] is True


@pytest.mark.asyncio
async def test_create_conversation_and_isolation(client):
    first = await _create_conversation(client)
    second = await _create_conversation(client)
    assert first != second

    send_first = await client.post(
        "/api/chat",
        json={"conversation_id": first, "message": "Hello there"},
    )
    assert send_first.status_code == 200

    first_messages = await client.get(f"/api/conversations/{first}/messages")
    second_messages = await client.get(f"/api/conversations/{second}/messages")
    assert first_messages.status_code == 200
    assert second_messages.status_code == 200
    assert len(first_messages.json()["messages"]) == 2
    assert second_messages.json()["messages"] == []
    assert first_messages.json()["messages"][0]["content"] == "Hello there"


@pytest.mark.asyncio
async def test_save_and_retrieve_messages(client):
    conversation_id = await _create_conversation(client)
    response = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "What is 2+2?"},
    )
    assert response.status_code == 200
    stored = await client.get(f"/api/conversations/{conversation_id}/messages")
    messages = stored.json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is 2+2?"
    assert "Direct answer" in messages[1]["content"]


@pytest.mark.asyncio
async def test_direct_answer_without_search(client):
    conversation_id = await _create_conversation(client)
    response = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "Explain gravity simply."},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["sources"] == []
    assert payload["answer"].startswith("Direct answer")


@pytest.mark.asyncio
async def test_tool_invocation_and_source_handling(client):
    conversation_id = await _create_conversation(client)
    response = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "SEARCH: latest match score"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["sources"] == [
        {"title": "Match report", "url": "https://example.com/match"}
    ]
    stored = await client.get(f"/api/conversations/{conversation_id}/messages")
    assistant = stored.json()["messages"][1]
    assert assistant["sources"][0]["url"] == "https://example.com/match"


@pytest.mark.asyncio
async def test_unknown_conversation(client):
    missing = str(uuid4())
    chat = await client.post(
        "/api/chat",
        json={"conversation_id": missing, "message": "Hello"},
    )
    messages = await client.get(f"/api/conversations/{missing}/messages")
    assert chat.status_code == 404
    assert messages.status_code == 404
    assert chat.json()["code"] == "CONVERSATION_NOT_FOUND"


@pytest.mark.asyncio
async def test_empty_and_oversized_message(client):
    conversation_id = await _create_conversation(client)
    empty = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "   "},
    )
    oversized = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "x" * 201},
    )
    assert empty.status_code == 400
    assert oversized.status_code == 400
    assert "too long" in oversized.json()["error"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "status", "code"),
    [
        ("FAIL_OLLAMA", 503, "OLLAMA_UNAVAILABLE"),
        ("FAIL_MODEL_MISSING", 503, "MODEL_UNAVAILABLE"),
        ("FAIL_TOOLS", 503, "TOOL_CALLING_UNSUPPORTED"),
        ("FAIL_SEARCH", 502, "SEARCH_FAILED"),
        ("FAIL_RATE", 429, "SEARCH_RATE_LIMIT"),
        ("FAIL_TIMEOUT", 504, "AGENT_TIMEOUT"),
        ("FAIL_MODEL", 502, "MODEL_FAILED"),
    ],
)
async def test_readable_failure_errors(client, message, status, code):
    conversation_id = await _create_conversation(client)
    response = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": message},
    )
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert body["error"]
    stored = await client.get(f"/api/conversations/{conversation_id}/messages")
    assert stored.json()["messages"] == []


@pytest.mark.asyncio
async def test_overlapping_conversation_is_rejected(app, client):
    conversation_id = await _create_conversation(client)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_run_turn(message, history, settings, **kwargs):
        started.set()
        await release.wait()
        return AgentTurnResult(answer="done", sources=[])

    app.state.run_turn = slow_run_turn
    first = asyncio.create_task(
        client.post(
            "/api/chat",
            json={"conversation_id": conversation_id, "message": "first"},
        )
    )
    await started.wait()
    second = await client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "second"},
    )
    assert second.status_code == 409
    release.set()
    first_response = await first
    assert first_response.status_code == 200


@pytest.mark.asyncio
async def test_frontend_is_served(client):
    home = await client.get("/")
    css = await client.get("/styles.css")
    js = await client.get("/app.js")
    assert home.status_code == 200
    assert "MSA AI Search Assistant" in home.text
    assert css.status_code == 200
    assert js.status_code == 200
