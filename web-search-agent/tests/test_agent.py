"""Agent tests with a mocked model graph and mocked search."""

from __future__ import annotations

import json

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import (
    AgentTurnResult,
    _detect_tool_support,
    _extract_answer,
    _extract_sources,
    _rewrite_runtime_error,
    run_turn,
)
from app.config import Settings
from app.exceptions import (
    AgentTimeoutError,
    ModelFailedError,
    OllamaUnavailableError,
    ToolCallingUnsupportedError,
)
from app.schemas import MessageRecord, SearchResult
from app.tools import begin_run_collection, set_search_runtime, web_search
from datetime import datetime, timezone


class FakeGraph:
    def __init__(self, messages):
        self.messages = messages
        self.calls = []

    async def ainvoke(self, payload, config=None):
        self.calls.append((payload, config))
        return {"messages": list(payload["messages"]) + self.messages}


def _now():
    return datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_direct_answer_path_does_not_require_search(settings: Settings):
    graph = FakeGraph([AIMessage(content="Photosynthesis converts light into chemical energy.")])
    result = await run_turn(
        "Explain photosynthesis.",
        [],
        settings,
        agent=graph,
        skip_ollama_check=True,
    )
    assert isinstance(result, AgentTurnResult)
    assert "Photosynthesis" in result.answer
    assert result.sources == []
    user_message = graph.calls[0][0]["messages"][-1]
    assert isinstance(user_message, HumanMessage)


@pytest.mark.asyncio
async def test_tool_invocation_collects_real_sources(settings: Settings):
    tool_payload = json.dumps(
        {
            "results": [
                {
                    "title": "World Cup final",
                    "url": "https://example.com/world-cup",
                    "snippet": "Argentina won the 2022 FIFA World Cup.",
                }
            ]
        }
    )
    graph = FakeGraph(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "latest world cup winner"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content=tool_payload, tool_call_id="call-1", name="web_search"),
            AIMessage(content="Argentina won in 2022 according to the snippet. [1]"),
        ]
    )
    result = await run_turn(
        "Who won the latest World Cup?",
        [MessageRecord(role="user", content="Hi", sources=None, created_at=_now())],
        settings,
        agent=graph,
        skip_ollama_check=True,
    )
    assert result.sources[0].url == "https://example.com/world-cup"
    assert "[1]" in result.answer


@pytest.mark.asyncio
async def test_run_turn_uses_injected_search_tool(settings: Settings):
    begin_run_collection()

    def fake_search(query: str) -> list[SearchResult]:
        return [
            SearchResult(
                title="Ollama",
                url="https://ollama.com",
                snippet=f"Results for {query}",
            )
        ]

    set_search_runtime(fake_search, max_results=5, timeout_seconds=5)
    tool_output = web_search.invoke({"query": "ollama"})
    graph = FakeGraph(
        [
            ToolMessage(content=tool_output, tool_call_id="call-1", name="web_search"),
            AIMessage(content="Ollama is a local model runner. [1]"),
        ]
    )
    result = await run_turn(
        "What is Ollama?",
        [],
        settings,
        agent=graph,
        search_fn=fake_search,
        skip_ollama_check=True,
    )
    assert result.sources[0].url == "https://ollama.com"


def test_extract_answer_strips_private_thoughts():
    messages = [
        AIMessage(content="<think>secret plan</think>Hello, I can help with questions."),
    ]
    assert _extract_answer(messages) == "Hello, I can help with questions."


def test_extract_sources_ignores_non_http_links():
    begin_run_collection()
    payload = json.dumps(
        {
            "results": [
                {"title": "bad", "url": "javascript:alert(1)", "snippet": "nope"},
                {"title": "good", "url": "https://example.com/ok", "snippet": "ok"},
            ]
        }
    )
    sources = _extract_sources(
        [ToolMessage(content=payload, tool_call_id="1", name="web_search")]
    )
    assert [item.url for item in sources] == ["https://example.com/ok"]


def test_detect_tool_support_from_capabilities():
    assert _detect_tool_support({"capabilities": ["completion", "tools"]}) is True
    assert _detect_tool_support({"capabilities": ["completion"]}) is False
    assert _detect_tool_support({"template": "{{ .Tools }}"}) is True
    assert _detect_tool_support({}) is None


def test_rewrite_runtime_error_maps_known_failures():
    assert isinstance(
        _rewrite_runtime_error(RuntimeError("connection refused")),
        OllamaUnavailableError,
    )
    assert isinstance(
        _rewrite_runtime_error(RuntimeError("model does not support tools")),
        ToolCallingUnsupportedError,
    )
    assert isinstance(
        _rewrite_runtime_error(RuntimeError("Recursion limit reached")),
        AgentTimeoutError,
    )
    assert isinstance(_rewrite_runtime_error(RuntimeError("boom")), ModelFailedError)


@pytest.mark.asyncio
async def test_agent_timeout(settings: Settings):
    class SlowGraph:
        async def ainvoke(self, payload, config=None):
            import asyncio

            await asyncio.sleep(0.2)
            return {"messages": [AIMessage(content="late")]}

    tight = settings.model_copy(update={"agent_timeout_seconds": 0.01})
    with pytest.raises(AgentTimeoutError):
        await run_turn(
            "Hello",
            [],
            tight,
            agent=SlowGraph(),
            skip_ollama_check=True,
        )
