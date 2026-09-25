"""OpenAI compatible request/response types (used for docs and error bodies)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_payload(
    message: str,
    *,
    type: str = "invalid_request_error",  # noqa: A002 - mirrors the OpenAI field name
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> dict:
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }


class TranscriptionSegment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class UsageDuration(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: int


class TranscriptionVerboseJson(BaseModel):
    task: str
    language: Optional[str] = None
    duration: float
    text: str
    segments: list[TranscriptionSegment]
    words: list[TranscriptionWord] = Field(default_factory=list)
    usage: UsageDuration


class TranscriptionJson(BaseModel):
    text: str


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "whisper-local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    device: Optional[str] = None
    compute_type: Optional[str] = None
    words_available: bool = False
    notes: list[str] = Field(default_factory=list)
