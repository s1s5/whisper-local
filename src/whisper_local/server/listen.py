"""Deepgram compatible ``WebSocket /v1/listen`` protocol.

This module owns everything about the streaming endpoint except the route
registration (which lives in :mod:`whisper_local.server.app`):

* query parameter parsing (plan section 2.2)
* authentication via ``Authorization`` / ``?api_key=`` /
  ``Sec-WebSocket-Protocol`` (plan section 2.1)
* raw ``linear16`` PCM frames -> 16 kHz mono float32 (see
  :mod:`whisper_local.audio`)
* endpointing / VAD and the resulting ``Results`` / ``SpeechStarted`` /
  ``UtteranceEnd`` / ``Metadata`` / ``Error`` messages (plan sections 2.4 / 6)

faster-whisper has no incremental decoding, so this is a *pseudo* stream: a
buffer plus a simple RMS based endpointing that reproduces Deepgram's
``interim`` (``is_final=false``) and ``final`` (``is_final=true``) model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from whisper_local import audio as audio_utils
from whisper_local.config import Settings
from whisper_local.transcriber import ModelLoadError, ModelManager

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = audio_utils.TARGET_SAMPLE_RATE

# Endpointing / VAD (plan section 4.5). Deepgram's `endpointing` default is
# 10 ms which is far too twitchy for a buffer + batch decoder, so the PoC uses
# a longer internal default when the client does not ask for one.
DEFAULT_ENDPOINTING_MS = 500
SILENCE_RMS_THRESHOLD = 0.01
MIN_SEGMENT_SECONDS = 0.15
MIN_INTERIM_SECONDS = 0.3
# Drop a buffer that has grown this long without any speech (avoids feeding
# minutes of silence to the decoder at the start of a session).
MAX_LEADING_SILENCE_SECONDS = 2.0

SUPPORTED_ENCODING = "linear16"
SUPPORTED_CHANNELS = 1
SUBPROTOCOL_TOKEN = "token"

# Query parameters this endpoint understands (everything else is ignored).
_KNOWN_PARAMS = frozenset(
    {
        "model",
        "language",
        "encoding",
        "sample_rate",
        "channels",
        "interim_results",
        "endpointing",
        "utterance_end_ms",
        "vad_events",
        "punctuate",
        "smart_format",
        "version",
    }
)


class ListenProtocolError(Exception):
    """A client error surfaced as a ``type:"Error"`` message then a close."""

    def __init__(self, err_code: str, err_msg: str) -> None:
        super().__init__(err_msg)
        self.err_code = err_code
        self.err_msg = err_msg


@dataclass(frozen=True)
class ListenOptions:
    """Parsed ``/v1/listen`` query parameters."""

    model: Optional[str] = None
    language: str = "ja"
    encoding: str = SUPPORTED_ENCODING
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = SUPPORTED_CHANNELS
    interim_results: bool = False
    endpointing_ms: Optional[int] = DEFAULT_ENDPOINTING_MS
    utterance_end_ms: Optional[int] = None
    vad_events: bool = False
    punctuate: bool = False
    smart_format: bool = False
    version: Optional[str] = None
    ignored: Mapping[str, str] = field(default_factory=dict)


def _to_int(raw: str, param: str) -> int:
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        raise ListenProtocolError("Bad Request", f"Invalid value for {param!r}: {raw!r}") from None


def _to_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return default


def parse_query(params: Mapping[str, str]) -> ListenOptions:
    """Validate the query string, raising :class:`ListenProtocolError` on bad values."""
    ignored = {key: value for key, value in params.items() if key not in _KNOWN_PARAMS}

    raw_channels = params.get("channels", str(SUPPORTED_CHANNELS))
    channels = _to_int(raw_channels, "channels")
    if channels != SUPPORTED_CHANNELS:
        raise ListenProtocolError(
            "Bad Request",
            f"Unsupported channels={channels}; only mono (channels=1) is supported.",
        )

    encoding = (params.get("encoding") or SUPPORTED_ENCODING).strip().lower()
    if encoding != SUPPORTED_ENCODING:
        raise ListenProtocolError(
            "Bad Request",
            f"Unsupported encoding={encoding!r}; only 'linear16' is supported.",
        )

    raw_sample_rate = params.get("sample_rate", str(TARGET_SAMPLE_RATE))
    sample_rate = _to_int(raw_sample_rate, "sample_rate")
    if sample_rate <= 0:
        raise ListenProtocolError("Bad Request", f"Invalid sample_rate={sample_rate}")

    raw_endpointing = params.get("endpointing")
    if raw_endpointing is None or raw_endpointing.strip() == "":
        endpointing_ms: Optional[int] = DEFAULT_ENDPOINTING_MS
    elif raw_endpointing.strip().lower() in {"false", "off"}:
        endpointing_ms = None
    else:
        endpointing_ms = _to_int(raw_endpointing, "endpointing")
        if endpointing_ms <= 0:
            endpointing_ms = None

    raw_utterance_end = params.get("utterance_end_ms")
    utterance_end_ms: Optional[int] = None
    if raw_utterance_end is not None and raw_utterance_end.strip() != "":
        utterance_end_ms = _to_int(raw_utterance_end, "utterance_end_ms")
        if utterance_end_ms <= 0:
            utterance_end_ms = None

    language = (params.get("language") or "").strip() or "ja"

    return ListenOptions(
        model=(params.get("model") or "").strip() or None,
        language=language,
        encoding=encoding,
        sample_rate=sample_rate,
        channels=channels,
        interim_results=_to_bool(params.get("interim_results"), False),
        endpointing_ms=endpointing_ms,
        utterance_end_ms=utterance_end_ms,
        vad_events=_to_bool(params.get("vad_events"), False),
        punctuate=_to_bool(params.get("punctuate"), False),
        smart_format=_to_bool(params.get("smart_format"), False),
        version=(params.get("version") or "").strip() or None,
        ignored=ignored,
    )


# -- authentication --------------------------------------------------------


def _offered_subprotocols(websocket: WebSocket) -> list[str]:
    header = websocket.headers.get("sec-websocket-protocol")
    if not header:
        return []
    return [part.strip() for part in header.split(",") if part.strip()]


def _extract_token(websocket: WebSocket) -> Optional[str]:
    """Pick the token from ``Authorization`` / ``?api_key=`` / subprotocol."""
    authorization = websocket.headers.get("authorization")
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() in {"token", "bearer"}:
            return parts[1].strip() or None

    api_key = websocket.query_params.get("api_key")
    if api_key:
        return api_key.strip() or None

    offered = _offered_subprotocols(websocket)
    if offered:
        lowered = [item.lower() for item in offered]
        if SUBPROTOCOL_TOKEN in lowered:
            index = lowered.index(SUBPROTOCOL_TOKEN)
            if index + 1 < len(offered):
                return offered[index + 1].strip() or None
            return None
        return offered[0].strip() or None
    return None


# -- session ---------------------------------------------------------------


class ListenSession:
    """Per-connection state machine: buffering, VAD and Deepgram messages."""

    def __init__(
        self,
        *,
        websocket: WebSocket,
        manager: ModelManager,
        settings: Settings,
        options: ListenOptions,
        request_id: str,
    ) -> None:
        self.websocket = websocket
        self.manager = manager
        self.settings = settings
        self.options = options
        self.request_id = request_id
        self.model_uuid = str(uuid.uuid4())

        self.sample_rate = options.sample_rate
        self.endpointing_ms = options.endpointing_ms
        self.interim_interval_ms = max(1, settings.listen_interim_interval_ms)

        self._hash = hashlib.sha256()
        self._buffer: list[np.ndarray] = []
        self._buffered_samples = 0
        self._total_samples = 0
        self._buffered_start = 0
        self._last_interim_sample = 0

        self._speech_active = False
        self._speech_started_sent = False
        self._last_speech_sample: Optional[int] = None
        self._last_word_end: Optional[float] = None
        self._utterance_end_sent = False

    # -- message helpers ---------------------------------------------------

    async def _send(self, message: dict[str, Any]) -> None:
        await self.websocket.send_json(message)

    async def send_error(self, err_code: str, err_msg: str) -> None:
        await self._send(
            {
                "type": "Error",
                "err_code": err_code,
                "err_msg": err_msg,
                "request_id": self.request_id,
            }
        )

    def _results_metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_info": {
                "name": self.options.model or self.settings.model,
                "version": self.options.version or "latest",
                "arch": "faster-whisper",
            },
            "model_uuid": self.model_uuid,
        }

    def _results_message(
        self,
        *,
        transcript: str,
        confidence: float,
        words: list[dict[str, Any]],
        start: float,
        duration: float,
        is_final: bool,
        speech_final: bool,
        from_finalize: bool,
    ) -> dict[str, Any]:
        return {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": round(duration, 3),
            "start": round(start, 3),
            "is_final": is_final,
            "speech_final": speech_final,
            "from_finalize": from_finalize,
            "channel": {
                "alternatives": [
                    {
                        "transcript": transcript,
                        "confidence": round(confidence, 4),
                        "words": words,
                    }
                ]
            },
            "metadata": self._results_metadata(),
        }

    def _metadata_message(self) -> dict[str, Any]:
        return {
            "type": "Metadata",
            "transaction_key": "deprecated",
            "request_id": self.request_id,
            "sha256": self._hash.hexdigest(),
            "created": datetime.now(timezone.utc).isoformat(),
            "duration": round(self._total_samples / TARGET_SAMPLE_RATE, 3),
            "channels": self.options.channels,
        }

    # -- buffer ------------------------------------------------------------

    def _clear_buffer(self) -> None:
        self._buffer = []
        self._buffered_samples = 0
        self._buffered_start = self._total_samples

    def _reset_speech(self) -> None:
        self._speech_active = False
        self._speech_started_sent = False

    @property
    def _buffer_duration(self) -> float:
        return self._buffered_samples / TARGET_SAMPLE_RATE

    # -- inference ---------------------------------------------------------

    async def _transcribe(self, samples: np.ndarray, *, start: float):
        """Run the batch decoder over *samples*.

        Consumes the segment generator added to :class:`ModelManager` so the
        interim / final paths share one code path.
        """

        def run():
            segments = list(
                self.manager.transcribe_stream(
                    samples, language=self.options.language, want_words=True
                )
            )
            transcript = "".join(segment.text for segment in segments).strip()
            words: list[dict[str, Any]] = []
            for segment in segments:
                for word in getattr(segment, "words", None) or []:
                    words.append(
                        {
                            "word": word.word,
                            "start": round(start + float(word.start), 3),
                            "end": round(start + float(word.end), 3),
                            "confidence": round(float(getattr(word, "probability", 0.0) or 0.0), 4),
                        }
                    )
            if words:
                confidence = sum(word["confidence"] for word in words) / len(words)
            elif segments:
                logprobs = [float(getattr(s, "avg_logprob", 0.0) or 0.0) for s in segments]
                confidence = math.exp(sum(logprobs) / len(logprobs))
            else:
                confidence = 0.0
            last_end = words[-1]["end"] if words else start + samples.size / TARGET_SAMPLE_RATE
            return transcript, max(0.0, min(1.0, confidence)), words, last_end

        return await run_in_threadpool(run)

    async def _finalize(self, *, from_finalize: bool, speech_final: bool) -> None:
        if self._buffered_samples < MIN_SEGMENT_SECONDS * TARGET_SAMPLE_RATE:
            self._clear_buffer()
            self._reset_speech()
            return

        samples = np.concatenate(self._buffer)
        start = self._buffered_start / TARGET_SAMPLE_RATE
        duration = samples.size / TARGET_SAMPLE_RATE
        try:
            transcript, confidence, words, last_end = await self._transcribe(samples, start=start)
        except ModelLoadError as exc:
            log.warning("listen finalize failed: %s", exc)
            self._clear_buffer()
            self._reset_speech()
            await self.send_error("Internal Server Error", f"Model is not available: {exc}")
            return

        self._clear_buffer()
        self._reset_speech()

        if not transcript:
            # Never emit a final for pure silence (hallucination guard).
            return

        self._last_word_end = last_end
        self._utterance_end_sent = False
        await self._send(
            self._results_message(
                transcript=transcript,
                confidence=confidence,
                words=words,
                start=start,
                duration=duration,
                is_final=True,
                speech_final=speech_final,
                from_finalize=from_finalize,
            )
        )

    async def _emit_interim(self) -> None:
        samples = np.concatenate(self._buffer)
        start = self._buffered_start / TARGET_SAMPLE_RATE
        duration = samples.size / TARGET_SAMPLE_RATE
        try:
            transcript, confidence, words, _last_end = await self._transcribe(samples, start=start)
        except ModelLoadError as exc:
            log.warning("listen interim failed: %s", exc)
            await self.send_error("Internal Server Error", f"Model is not available: {exc}")
            return
        if not transcript:
            return
        await self._send(
            self._results_message(
                transcript=transcript,
                confidence=confidence,
                words=words,
                start=start,
                duration=duration,
                is_final=False,
                speech_final=False,
                from_finalize=False,
            )
        )

    # -- events ------------------------------------------------------------

    async def on_media(self, payload: bytes) -> None:
        self._hash.update(payload)
        try:
            frame = audio_utils.decode_linear16(payload, sample_rate=self.sample_rate)
        except Exception as exc:  # noqa: BLE001 - malformed frame is a client error
            await self.send_error("Bad Request", f"Failed to decode audio frame: {exc}")
            return
        if frame.size == 0:
            return

        self._buffer.append(frame)
        self._buffered_samples += frame.size
        self._total_samples += frame.size
        now = self._total_samples

        level = audio_utils.rms(frame)
        if level >= SILENCE_RMS_THRESHOLD:
            if not self._speech_active:
                self._speech_active = True
                if self.options.vad_events and not self._speech_started_sent:
                    await self._send(
                        {
                            "type": "SpeechStarted",
                            "channel": [0],
                            "timestamp": round(self._buffered_start / TARGET_SAMPLE_RATE, 3),
                        }
                    )
                    self._speech_started_sent = True
            self._last_speech_sample = now
            self._utterance_end_sent = False
        else:
            if self._speech_active and self.endpointing_ms is not None:
                silence = now - (self._last_speech_sample or now)
                if silence >= self.endpointing_ms * TARGET_SAMPLE_RATE / 1000:
                    await self._finalize(from_finalize=False, speech_final=True)
                    return
            elif not self._speech_active and self._buffer_duration > MAX_LEADING_SILENCE_SECONDS:
                self._clear_buffer()

        if (
            self.options.utterance_end_ms is not None
            and not self._utterance_end_sent
            and self._last_speech_sample is not None
            and now - self._last_speech_sample
            >= self.options.utterance_end_ms * TARGET_SAMPLE_RATE / 1000
        ):
            last_end = (
                self._last_word_end
                if self._last_word_end is not None
                else self._last_speech_sample / TARGET_SAMPLE_RATE
            )
            await self._send(
                {"type": "UtteranceEnd", "channel": [0], "last_word_end": round(last_end, 3)}
            )
            self._utterance_end_sent = True

        if (
            self.options.interim_results
            and self._speech_active
            and self._buffer_duration >= MIN_INTERIM_SECONDS
            and now - self._last_interim_sample
            >= self.interim_interval_ms * TARGET_SAMPLE_RATE / 1000
        ):
            self._last_interim_sample = now
            await self._emit_interim()

    async def on_text(self, text: str) -> bool:
        """Handle a client JSON message. Returns ``True`` when the session ends."""
        try:
            payload = json.loads(text)
        except ValueError:
            await self.send_error("Bad Request", "Invalid JSON submitted.")
            return False
        if not isinstance(payload, dict):
            await self.send_error("Bad Request", "Invalid JSON submitted.")
            return False

        message_type = payload.get("type")
        if message_type == "Finalize":
            await self._finalize(from_finalize=True, speech_final=True)
            return False
        if message_type == "CloseStream":
            await self._finalize(from_finalize=True, speech_final=True)
            await self._send(self._metadata_message())
            await self.websocket.close(code=1000)
            return True
        if message_type == "KeepAlive":
            # Accepted; intentionally not buffered as audio.
            return False
        await self.send_error("Bad Request", f"Unknown message type: {message_type!r}")
        return False

    async def serve(self) -> None:
        """Receive loop until the client disconnects or sends CloseStream."""
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is not None:
                    await self.on_media(data)
                    continue
                text = message.get("text")
                if text is not None:
                    if await self.on_text(text):
                        return
        except WebSocketDisconnect:
            return


# -- entry point -----------------------------------------------------------


async def handle_listen(websocket: WebSocket, manager: ModelManager, settings: Settings) -> None:
    """Serve a single ``/v1/listen`` connection."""
    request_id = str(uuid.uuid4())

    if settings.api_key:
        provided = _extract_token(websocket)
        if provided != settings.api_key:
            await websocket.send_denial_response(
                JSONResponse(
                    {
                        "err_code": "INVALID_AUTH",
                        "err_msg": "Invalid credentials.",
                        "request_id": request_id,
                    },
                    status_code=401,
                )
            )
            return

    offered = [item.lower() for item in _offered_subprotocols(websocket)]
    subprotocol = SUBPROTOCOL_TOKEN if SUBPROTOCOL_TOKEN in offered else None
    await websocket.accept(subprotocol=subprotocol)

    try:
        options = parse_query(websocket.query_params)
    except ListenProtocolError as exc:
        session = ListenSession(
            websocket=websocket,
            manager=manager,
            settings=settings,
            options=ListenOptions(),
            request_id=request_id,
        )
        await session.send_error(exc.err_code, exc.err_msg)
        await websocket.close(code=1000)
        return

    log.info(
        "listen session %s: model=%s language=%s encoding=%s sample_rate=%s "
        "channels=%s interim=%s endpointing=%s utterance_end=%s vad_events=%s ignored=%s",
        request_id,
        options.model,
        options.language,
        options.encoding,
        options.sample_rate,
        options.channels,
        options.interim_results,
        options.endpointing_ms,
        options.utterance_end_ms,
        options.vad_events,
        dict(options.ignored),
    )

    if not manager.loaded:
        try:
            await run_in_threadpool(manager.load)
        except ModelLoadError as exc:
            session = ListenSession(
                websocket=websocket,
                manager=manager,
                settings=settings,
                options=options,
                request_id=request_id,
            )
            await session.send_error("Internal Server Error", f"Model is not available: {exc}")
            await websocket.close(code=1000)
            return

    session = ListenSession(
        websocket=websocket,
        manager=manager,
        settings=settings,
        options=options,
        request_id=request_id,
    )
    await session.serve()
