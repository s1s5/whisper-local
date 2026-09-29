"""Raw PCM helpers for the Deepgram compatible listen endpoint.

Deepgram's streaming API receives raw binary audio frames. faster-whisper
expects a 16 kHz mono float32 numpy array, so this module owns the pure
conversion functions:

* :func:`decode_linear16` - ``linear16`` bytes -> float32 in ``[-1, 1]``
* :func:`decode_ulaw` / :func:`decode_alaw` - G.711 companded bytes -> float32
* :func:`decode_frame` - dispatch by Deepgram/OpenAI encoding name
* :func:`resample_linear` - numpy linear interpolation resampler
* :func:`rms` - frame loudness used by the endpointing VAD

G.711 is implemented here in numpy because Python 3.13 removed the stdlib
``audioop`` module (PEP 594). The tables were validated against ``ffmpeg``
mulaw/alaw decoding (all 256 byte values, exact match).

Everything here is a pure function (no state, no I/O) so the protocol layer in
:mod:`whisper_local.server.listen` can stay focused on state and events.
"""

from __future__ import annotations

import numpy as np

from whisper_local.transcriber import AudioDecodeError

TARGET_SAMPLE_RATE = 16000
_INT16_SCALE = 32768.0

# Deepgram's canonical encoding names plus the OpenAI/OpenClaw spellings
# (``g711_ulaw`` / ``g711_alaw``) are accepted interchangeably.
ENCODING_ALIASES = {
    "linear16": "linear16",
    "pcm16": "linear16",
    "mulaw": "mulaw",
    "ulaw": "mulaw",
    "mu-law": "mulaw",
    "g711_ulaw": "mulaw",
    "alaw": "alaw",
    "a-law": "alaw",
    "g711_alaw": "alaw",
}

# Default wire sample rate per encoding: G.711 telephony is 8 kHz, linear16 is 16 kHz.
DEFAULT_SAMPLE_RATE_BY_ENCODING = {
    "linear16": TARGET_SAMPLE_RATE,
    "mulaw": 8000,
    "alaw": 8000,
}


def normalize_encoding(raw: str) -> str:
    """Map a client encoding name to our canonical one (or raise)."""
    key = (raw or "").strip().lower()
    canonical = ENCODING_ALIASES.get(key)
    if canonical is None:
        raise AudioDecodeError(f"unsupported encoding: {raw!r}")
    return canonical


def decode_linear16(payload: bytes, *, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a ``linear16`` frame into ``target_rate`` mono float32 samples.

    Raises :class:`AudioDecodeError` when the frame cannot be interpreted as
    little-endian signed 16-bit PCM (an odd number of bytes).
    """
    if len(payload) % 2 != 0:
        raise AudioDecodeError(
            "linear16 frame has an odd number of bytes and cannot be decoded"
        )
    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    audio = samples.astype(np.float32) / _INT16_SCALE
    if sample_rate != target_rate:
        audio = resample_linear(audio, sample_rate, target_rate)
    return audio


def _ulaw_to_int16(codes: np.ndarray) -> np.ndarray:
    """G.711 µ-law bytes -> int16 (matches ffmpeg's mulaw decoder exactly)."""
    u = ~codes.astype(np.int32) & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t = t << ((u & 0x70) >> 4)
    return np.where((u & 0x80) != 0, 0x84 - t, t - 0x84).astype(np.int16)


def _alaw_to_int16(codes: np.ndarray) -> np.ndarray:
    """G.711 A-law bytes -> int16 (matches ffmpeg's alaw decoder exactly)."""
    a = codes.astype(np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(
        seg == 0,
        t + 8,
        np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)),
    )
    return np.where((a & 0x80) != 0, t, -t).astype(np.int16)


def decode_ulaw(payload: bytes, *, sample_rate: int = 8000, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a ``mulaw``/``g711_ulaw`` frame (one byte per sample)."""
    if not payload:
        return np.empty(0, dtype=np.float32)
    codes = np.frombuffer(payload, dtype=np.uint8)
    audio = _ulaw_to_int16(codes).astype(np.float32) / _INT16_SCALE
    if sample_rate != target_rate:
        audio = resample_linear(audio, sample_rate, target_rate)
    return audio


def decode_alaw(payload: bytes, *, sample_rate: int = 8000, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode an ``alaw``/``g711_alaw`` frame (one byte per sample)."""
    if not payload:
        return np.empty(0, dtype=np.float32)
    codes = np.frombuffer(payload, dtype=np.uint8)
    audio = _alaw_to_int16(codes).astype(np.float32) / _INT16_SCALE
    if sample_rate != target_rate:
        audio = resample_linear(audio, sample_rate, target_rate)
    return audio


def decode_frame(payload: bytes, *, encoding: str, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a raw audio frame using the client's canonical encoding name."""
    if encoding == "linear16":
        return decode_linear16(payload, sample_rate=sample_rate, target_rate=target_rate)
    if encoding == "mulaw":
        return decode_ulaw(payload, sample_rate=sample_rate, target_rate=target_rate)
    if encoding == "alaw":
        return decode_alaw(payload, sample_rate=sample_rate, target_rate=target_rate)
    raise AudioDecodeError(f"unsupported encoding: {encoding!r}")


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample via linear interpolation (adequate for a PoC, no new deps)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    if src_rate <= 0 or dst_rate <= 0:
        raise AudioDecodeError(f"invalid sample rate: src={src_rate} dst={dst_rate}")

    duration = samples.size / src_rate
    target_length = max(1, int(round(duration * dst_rate)))
    src_positions = np.linspace(0.0, duration, samples.size, endpoint=False)
    dst_positions = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square loudness in ``[0, 1]``; ``0.0`` for empty frames."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
