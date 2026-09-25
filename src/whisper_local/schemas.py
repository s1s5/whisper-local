"""Response formatting for the OpenAI compatible endpoints.

``format_srt`` owns the subtitle rendering; this module owns the mapping of
faster-whisper results onto OpenAI JSON shapes and re-exports the subtitle
helpers so the API layer has a single import.
"""

from __future__ import annotations

from typing import Any, Iterable

from faster_whisper.transcribe import Segment

from whisper_local.server.format_srt import srt_timestamp, to_srt, to_vtt, vtt_timestamp
from whisper_local.transcriber import TranscriptionResult

GRANULARITY_SEGMENT = "segment"
GRANULARITY_WORD = "word"

RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")

__all__ = [
    "GRANULARITY_SEGMENT",
    "GRANULARITY_WORD",
    "RESPONSE_FORMATS",
    "collect_words",
    "full_text",
    "segment_to_dict",
    "srt_timestamp",
    "to_json",
    "to_srt",
    "to_text",
    "to_verbose_json",
    "to_vtt",
    "vtt_timestamp",
    "word_to_dict",
]


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def segment_to_dict(segment: Segment, index: int) -> dict[str, Any]:
    """Map a faster-whisper segment onto the OpenAI ``verbose_json`` shape."""
    return {
        "id": index,
        "seek": int(getattr(segment, "seek", 0) or 0),
        "start": _round(segment.start),
        "end": _round(segment.end),
        "text": segment.text,
        "tokens": [int(token) for token in (segment.tokens or [])],
        "temperature": _round(getattr(segment, "temperature", 0.0)),
        "avg_logprob": _round(segment.avg_logprob, 6),
        "compression_ratio": _round(segment.compression_ratio, 6),
        "no_speech_prob": _round(segment.no_speech_prob, 6),
    }


def word_to_dict(word) -> dict[str, Any]:
    return {
        "word": word.word,
        "start": _round(word.start),
        "end": _round(word.end),
    }


def full_text(segments: Iterable[Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def collect_words(result: TranscriptionResult) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.segments:
        for word in getattr(segment, "words", None) or []:
            words.append(word_to_dict(word))
    return words


def to_json(result: TranscriptionResult) -> dict[str, Any]:
    return {"text": full_text(result.segments)}


def to_text(result: TranscriptionResult) -> str:
    return full_text(result.segments)


def to_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    """OpenAI ``verbose_json`` payload (plan section 4.1)."""
    duration = float(getattr(result.info, "duration", 0.0) or 0.0)
    return {
        "task": result.task,
        "language": getattr(result.info, "language", None),
        "duration": _round(duration),
        "text": full_text(result.segments),
        "segments": [
            segment_to_dict(segment, index)
            for index, segment in enumerate(result.segments)
        ],
        "words": collect_words(result),
        "usage": {"type": "duration", "seconds": int(round(duration))},
    }


def segments_srt(result: TranscriptionResult) -> str:
    return to_srt(result.segments)


def segments_vtt(result: TranscriptionResult) -> str:
    return to_vtt(result.segments)
