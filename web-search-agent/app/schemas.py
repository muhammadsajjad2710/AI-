"""Pydantic request and response models."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationCreateResponse(BaseModel):
    conversation_id: UUID


class ChatRequest(BaseModel):
    conversation_id: UUID
    message: str = Field(..., min_length=1, max_length=20000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned


class Source(BaseModel):
    title: str
    url: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    sources: list[Source] = Field(default_factory=list)


class MessageRecord(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] | None = None
    created_at: datetime


class MessagesResponse(BaseModel):
    conversation_id: UUID
    messages: list[MessageRecord]


class OllamaHealth(BaseModel):
    available: bool
    base_url: str
    model: str
    model_available: bool
    supports_tools: bool | None
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: OllamaHealth


class ErrorResponse(BaseModel):
    error: str
    code: str
