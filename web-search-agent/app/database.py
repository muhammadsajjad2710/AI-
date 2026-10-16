"""SQLite persistence for conversations and messages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.schemas import MessageRecord, Source

CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
"""

CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
"""

CREATE_MESSAGES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON messages (conversation_id, id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._connection = await aiosqlite.connect(self.path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys = ON;")
        await self._connection.execute(CREATE_CONVERSATIONS)
        await self._connection.execute(CREATE_MESSAGES)
        await self._connection.execute(CREATE_MESSAGES_INDEX)
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database is not initialized.")
        return self._connection

    async def create_conversation(self) -> str:
        conversation_id = str(uuid4())
        await self.connection.execute(
            "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
            (conversation_id, utc_now()),
        )
        await self.connection.commit()
        return conversation_id

    async def conversation_exists(self, conversation_id: str) -> bool:
        cursor = await self.connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return row is not None

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[Source] | None = None,
    ) -> None:
        payload = None
        if sources:
            payload = json.dumps([source.model_dump() for source in sources])
        await self.connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, payload, utc_now()),
        )
        await self.connection.commit()

    async def list_messages(self, conversation_id: str) -> list[MessageRecord]:
        cursor = await self.connection.execute(
            """
            SELECT role, content, sources_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        messages: list[MessageRecord] = []
        for row in rows:
            sources = None
            if row["sources_json"]:
                sources = [Source.model_validate(item) for item in json.loads(row["sources_json"])]
            messages.append(
                MessageRecord(
                    role=row["role"],
                    content=row["content"],
                    sources=sources,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return messages

    async def recent_history(
        self,
        conversation_id: str,
        limit: int,
    ) -> list[MessageRecord]:
        messages = await self.list_messages(conversation_id)
        if limit <= 0:
            return []
        return messages[-limit:]
