"""Unit tests for the DuckDuckGo search tool."""

from __future__ import annotations

import json

import pytest

from app.exceptions import SearchFailedError, SearchRateLimitError
from app.schemas import SearchResult
from app.tools import (
    begin_run_collection,
    default_web_search,
    get_collected_sources,
    set_search_runtime,
    web_search,
)


def test_web_search_returns_structured_results():
    begin_run_collection()

    def fake_search(query: str) -> list[SearchResult]:
        assert query == "ollama llama3.2"
        return [
            SearchResult(
                title="Ollama library",
                url="https://ollama.com/library/llama3.2",
                snippet="Llama 3.2 is available in 1B and 3B sizes.",
            )
        ]

    set_search_runtime(fake_search, max_results=5, timeout_seconds=5)
    raw = web_search.invoke({"query": "ollama llama3.2"})
    payload = json.loads(raw)
    assert payload["results"][0]["title"] == "Ollama library"
    assert payload["results"][0]["url"].startswith("https://")
    assert "snippet" in payload["results"][0]
    sources = get_collected_sources()
    assert sources[0].url == "https://ollama.com/library/llama3.2"


def test_web_search_reports_rate_limit_without_inventing_results():
    begin_run_collection()

    def fake_search(_query: str) -> list[SearchResult]:
        raise SearchRateLimitError("Web search was rate-limited. Wait a moment and try again.")

    set_search_runtime(fake_search, max_results=5, timeout_seconds=5)
    payload = json.loads(web_search.invoke({"query": "today news"}))
    assert payload["error"] == "rate_limit"
    assert payload["results"] == []
    assert get_collected_sources() == []


def test_web_search_reports_search_failure():
    begin_run_collection()

    def fake_search(_query: str) -> list[SearchResult]:
        raise SearchFailedError("DuckDuckGo did not return results.")

    set_search_runtime(fake_search, max_results=5, timeout_seconds=5)
    payload = json.loads(web_search.invoke({"query": "latest scores"}))
    assert payload["error"] == "search_failed"
    assert payload["results"] == []


def test_default_web_search_uses_ddgs(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout):
            self.timeout = timeout

        def text(self, query, max_results):
            assert query == "python fastapi"
            assert max_results == 5
            return [
                {
                    "title": "FastAPI",
                    "href": "https://fastapi.tiangolo.com/",
                    "body": "FastAPI framework, high performance.",
                }
            ]

    fake_exceptions = type(
        "exc",
        (),
        {
            "DDGSException": type("DDGSException", (Exception,), {}),
            "RatelimitException": type("RatelimitException", (Exception,), {}),
            "TimeoutException": type("TimeoutException", (Exception,), {}),
        },
    )

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    monkeypatch.setitem(__import__("sys").modules, "ddgs.exceptions", fake_exceptions)
    set_search_runtime(None, max_results=5, timeout_seconds=9)
    results = default_web_search("python fastapi")
    assert results[0].title == "FastAPI"
    assert results[0].url == "https://fastapi.tiangolo.com/"
    assert "high performance" in results[0].snippet
