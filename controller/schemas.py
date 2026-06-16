from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MessagePart(BaseModel):
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | list[MessagePart]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = 0.7
    top_k: int = Field(default=40, ge=0, le=512)
    stop: str | list[str] | None = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]


class StreamDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[StreamChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
