"""FastAPI application for the local MSA AI Search Assistant."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.agent import AgentTurnResult, check_ollama_status, run_turn
from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import (
    AppError,
    ConversationBusyError,
    ConversationNotFoundError,
    MessageValidationError,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreateResponse,
    ErrorResponse,
    HealthResponse,
    MessagesResponse,
    OllamaHealth,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _error_response(status_code: int, message: str, code: str) -> JSONResponse:
    payload = ErrorResponse(error=message, code=code)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    database = Database(settings.resolved_database_path())
    await database.initialize()
    app.state.db = database
    app.state.busy_guard = asyncio.Lock()
    app.state.busy_conversations = set()
    if not hasattr(app.state, "run_turn"):
        app.state.run_turn = run_turn
    yield
    await database.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    application = FastAPI(
        title="MSA AI Search Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.run_turn = run_turn
    _register_exception_handlers(application)
    _register_routes(application)
    return application


def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.message, exc.code)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        messages: list[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
            detail = error.get("msg", "Invalid input.")
            if location:
                messages.append(f"{location}: {detail}")
            else:
                messages.append(detail)
        readable = " ".join(messages) if messages else "Invalid request."
        if "message" in readable.lower() and (
            "empty" in readable.lower() or "at least" in readable.lower()
        ):
            readable = "Message cannot be empty."
        return _error_response(400, readable, "INVALID_REQUEST")


def _register_routes(application: FastAPI) -> None:
    @application.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/styles.css")
    async def styles() -> FileResponse:
        return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

    @application.get("/app.js")
    async def script() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        settings: Settings = request.app.state.settings
        status = await check_ollama_status(settings)
        overall = "ok" if status.available and status.model_available and status.supports_tools is not False else "degraded"
        return HealthResponse(
            status=overall,
            ollama=OllamaHealth(
                available=status.available,
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                model_available=status.model_available,
                supports_tools=status.supports_tools,
                detail=status.detail,
            ),
        )

    @application.post("/api/conversations", response_model=ConversationCreateResponse)
    async def create_conversation(request: Request) -> ConversationCreateResponse:
        db: Database = request.app.state.db
        conversation_id = await db.create_conversation()
        return ConversationCreateResponse(conversation_id=UUID(conversation_id))

    @application.get(
        "/api/conversations/{conversation_id}/messages",
        response_model=MessagesResponse,
    )
    async def get_messages(conversation_id: UUID, request: Request) -> MessagesResponse:
        db: Database = request.app.state.db
        key = str(conversation_id)
        if not await db.conversation_exists(key):
            raise ConversationNotFoundError("That conversation was not found.")
        messages = await db.list_messages(key)
        return MessagesResponse(conversation_id=conversation_id, messages=messages)

    @application.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        settings: Settings = request.app.state.settings
        db: Database = request.app.state.db
        conversation_id = str(payload.conversation_id)

        if len(payload.message) > settings.max_message_length:
            raise MessageValidationError(
                f"Message is too long. Maximum length is {settings.max_message_length} characters."
            )
        if not await db.conversation_exists(conversation_id):
            raise ConversationNotFoundError("That conversation was not found.")

        busy_guard: asyncio.Lock = request.app.state.busy_guard
        busy: set[str] = request.app.state.busy_conversations
        async with busy_guard:
            if conversation_id in busy:
                raise ConversationBusyError(
                    "This conversation is already generating a reply. Wait for it to finish."
                )
            busy.add(conversation_id)

        try:
            history = await db.recent_history(
                conversation_id,
                settings.conversation_history_limit,
            )
            runner = request.app.state.run_turn
            result: AgentTurnResult = await runner(
                payload.message,
                history,
                settings,
            )
            await db.add_message(conversation_id, "user", payload.message)
            await db.add_message(
                conversation_id,
                "assistant",
                result.answer,
                sources=result.sources or None,
            )
            return ChatResponse(
                conversation_id=payload.conversation_id,
                answer=result.answer,
                sources=result.sources,
            )
        finally:
            async with busy_guard:
                busy.discard(conversation_id)


app = create_app()
