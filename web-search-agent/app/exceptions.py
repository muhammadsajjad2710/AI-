"""Domain errors mapped to readable HTTP responses."""


class AppError(Exception):
    """Base application error with an HTTP status and public message."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConversationNotFoundError(AppError):
    status_code = 404
    code = "CONVERSATION_NOT_FOUND"


class MessageValidationError(AppError):
    status_code = 400
    code = "INVALID_MESSAGE"


class ConversationBusyError(AppError):
    status_code = 409
    code = "CONVERSATION_BUSY"


class OllamaUnavailableError(AppError):
    status_code = 503
    code = "OLLAMA_UNAVAILABLE"


class ModelUnavailableError(AppError):
    status_code = 503
    code = "MODEL_UNAVAILABLE"


class ToolCallingUnsupportedError(AppError):
    status_code = 503
    code = "TOOL_CALLING_UNSUPPORTED"


class SearchRateLimitError(AppError):
    status_code = 429
    code = "SEARCH_RATE_LIMIT"


class SearchFailedError(AppError):
    status_code = 502
    code = "SEARCH_FAILED"


class AgentTimeoutError(AppError):
    status_code = 504
    code = "AGENT_TIMEOUT"


class ModelFailedError(AppError):
    status_code = 502
    code = "MODEL_FAILED"
