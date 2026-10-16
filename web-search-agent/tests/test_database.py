"""Database isolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.database import Database
from app.schemas import Source


@pytest.mark.asyncio
async def test_conversations_are_isolated(tmp_path: Path):
    db = Database(tmp_path / "isolated.db")
    await db.initialize()
    first = await db.create_conversation()
    second = await db.create_conversation()
    await db.add_message(first, "user", "alpha")
    await db.add_message(
        first,
        "assistant",
        "reply",
        sources=[Source(title="Doc", url="https://example.com/doc")],
    )
    first_messages = await db.list_messages(first)
    second_messages = await db.list_messages(second)
    assert len(first_messages) == 2
    assert second_messages == []
    assert first_messages[1].sources[0].url == "https://example.com/doc"
    history = await db.recent_history(first, 1)
    assert len(history) == 1
    assert history[0].role == "assistant"
    await db.close()
