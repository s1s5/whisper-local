"""Raw PCM helpers for the Deepgram compatible listen endpoint.

Deepgram's streaming API receives raw binary audio frames (``encoding=linear16``
means little-endian signed 16-bit PCM). faster-whisper expects a 16 kHz mono
float32 numpy array, so this module owns the pure conversion functions:

* :func:`decode_linear16` - ``linear16`` bytes -> float32 in ``[-1, 1]``
* :func:`resample_linear` - numpy linear interpolation resampler
* :func:`rms` - frame loudness used by the endpointing VAD

Everything here is a pure function (no state, no I/O) so the protocol layer in
:mod:`whisper_local.server.listen` can stay focused on state and events.
"""

from __future__ import annotations

import numpy as np

from whisper_local.transcriber import AudioDecodeError

TARGET_SAMPLE_RATE = 16000
_INT16_SCALE = 32768.0


def decode_linear16(payload: bytes, *, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a ``linear16`` frame into 16 kHz mono float32 samples.

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
