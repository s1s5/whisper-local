"""Contract tests for the Deepgram compatible ``WS /v1/listen`` endpoint.

Uses a stub :class:`ModelManager` (no 1.5 GB model, no inference) plus the
Starlette/FastAPI ``TestClient.websocket_connect`` harness. The stub returns a
canned transcript only for frames that actually contain energy, so the
"pure silence never produces a final" rule is exercised end to end.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from whisper_local.audio import (
    decode_alaw,
    decode_linear16,
    decode_ulaw,
    normalize_encoding,
    resample_linear,
    rms,
)
from whisper_local.config import Settings
from whisper_local.server.app import create_app

SAMPLE_RATE = 16000
TRANSCRIPT = "こんにちは世界"


# -- stubs -----------------------------------------------------------------


@dataclasses.dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.97


@dataclasses.dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.1
    words: list = dataclasses.field(default_factory=list)


class ListenStubManager:
    """Minimal stand-in that only transcribes frames containing energy."""

    def __init__(self) -> None:
        self.loaded = True
        self.words_enabled = True
        self.device = "cpu"
        self.compute_type = "int8"
        self.load_notes: list[str] = []
        self.calls: list[dict] = []

    def load(self) -> None:
        self.loaded = True

    def transcribe_stream(self, samples, *, language=None, want_words=False):
        self.calls.append({"samples": int(samples.size), "language": language, "want_words": want_words})
        if rms(np.asarray(samples)) < 0.01:
            return
        duration = samples.size / SAMPLE_RATE
        yield FakeSegment(
            start=0.0,
            end=duration,
            text=TRANSCRIPT,
            words=[FakeWord(TRANSCRIPT, 0.0, duration)],
        )


def make_client(**kwargs):
    kwargs.setdefault("save_audio", False)
    api_key = kwargs.pop("api_key", None)
    settings = Settings(api_key=api_key, **kwargs)
    manager = ListenStubManager()
    return TestClient(create_app(settings=settings, manager=manager)), manager


# -- helpers ---------------------------------------------------------------


def speech(seconds: float, amplitude: float = 0.3) -> bytes:
    count = int(seconds * SAMPLE_RATE)
    return (np.full(count, amplitude, dtype=np.float32) * 32767.0).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    return b"\x00" * (int(seconds * SAMPLE_RATE) * 2)


# G.711 is 8 kHz, one byte per sample. 0x80 = full positive, 0xFF = silence.
G711_RATE = 8000

def mulaw_speech(seconds: float, byte: int = 0x80) -> bytes:
    return bytes([byte]) * int(seconds * G711_RATE)

def alaw_speech(seconds: float, byte: int = 0xAA) -> bytes:
    return bytes([byte]) * int(seconds * G711_RATE)


# -- happy paths -----------------------------------------------------------


def test_endpointing_emits_final_then_metadata_on_close_stream():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=linear16&sample_rate=16000") as ws:
        ws.send_bytes(speech(1.0))
        ws.send_bytes(silence(1.0))  # > default 500 ms endpointing
        final = ws.receive_json()
        assert final["type"] == "Results"
        assert final["is_final"] is True
        assert final["speech_final"] is True
        assert final["from_finalize"] is False
        assert final["channel_index"] == [0, 1]
        assert final["channel"]["alternatives"][0]["transcript"] == TRANSCRIPT
        assert final["metadata"]["request_id"]

        ws.send_json({"type": "CloseStream"})
        metadata = ws.receive_json()

    assert metadata["type"] == "Metadata"
    assert metadata["channels"] == 1
    assert metadata["transaction_key"]
    assert len(metadata["sha256"]) == 64


def test_finalize_message_marks_from_finalize():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_bytes(speech(0.8))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"
    assert final["is_final"] is True
    assert final["speech_final"] is True
    assert final["from_finalize"] is True


def test_results_start_duration_and_words():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_bytes(speech(1.0))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["start"] == pytest.approx(0.0, abs=0.01)
    assert final["duration"] == pytest.approx(1.0, abs=0.05)
    alternative = final["channel"]["alternatives"][0]
    assert alternative["transcript"] == TRANSCRIPT
    assert 0.0 < alternative["confidence"] <= 1.0
    assert alternative["words"]
    assert set(alternative["words"][0]) == {"word", "start", "end", "confidence"}


def test_interim_results_then_final():
    client, _ = make_client(listen_interim_interval_ms=0)
    with client.websocket_connect("/v1/listen?interim_results=true&endpointing=100000") as ws:
        ws.send_bytes(speech(0.5))
        interim = ws.receive_json()
        assert interim["type"] == "Results"
        assert interim["is_final"] is False
        assert interim["speech_final"] is False
        assert interim["channel"]["alternatives"][0]["transcript"] == TRANSCRIPT

        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["is_final"] is True
    assert final["speech_final"] is True
    assert final["from_finalize"] is True


def test_speech_started_and_utterance_end_events():
    client, _ = make_client()
    url = "/v1/listen?vad_events=true&utterance_end_ms=500&endpointing=false"
    with client.websocket_connect(url) as ws:
        ws.send_bytes(speech(0.5))
        started = ws.receive_json()
        assert started["type"] == "SpeechStarted"
        assert started["channel"] == [0]
        assert started["timestamp"] == pytest.approx(0.0, abs=0.01)

        ws.send_bytes(silence(1.0))
        end = ws.receive_json()
    assert end["type"] == "UtteranceEnd"
    assert end["channel"] == [0]
    assert end["last_word_end"] == pytest.approx(0.5, abs=0.05)


def test_keep_alive_is_accepted_and_ignored():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_json({"type": "KeepAlive"})
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"
    assert final["channel"]["alternatives"][0]["transcript"] == TRANSCRIPT


def test_metadata_is_emitted_even_without_audio():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_json({"type": "CloseStream"})
        metadata = ws.receive_json()
    assert metadata["type"] == "Metadata"
    assert metadata["duration"] == 0


def test_pure_silence_never_emits_a_final():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_bytes(silence(1.0))
        ws.send_json({"type": "Finalize"})
        # No Results expected for silence: the next message is Metadata.
        ws.send_json({"type": "CloseStream"})
        message = ws.receive_json()
    assert message["type"] == "Metadata"


def test_unknown_query_parameters_are_ignored():
    client, manager = make_client()
    url = "/v1/listen?diarize=true&smart_format=true&punctuate=true&foo=bar&language=en"
    with client.websocket_connect(url) as ws:
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"
    assert manager.calls[0]["language"] == "en"


# -- error handling --------------------------------------------------------


def test_channels_greater_than_one_is_rejected():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?channels=2") as ws:
        error = ws.receive_json()
    assert error["type"] == "Error"
    assert error["err_code"] == "Bad Request"
    assert "channels" in error["err_msg"]


def test_unsupported_encoding_is_rejected():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=opus") as ws:
        error = ws.receive_json()
    assert error["type"] == "Error"
    assert "encoding" in error["err_msg"]


def test_mulaw_encoding_is_accepted_and_transcribes():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=mulaw&sample_rate=8000") as ws:
        ws.send_bytes(mulaw_speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"
    assert final["channel"]["alternatives"][0]["transcript"] == TRANSCRIPT


def test_g711_ulaw_alias_is_accepted():
    # OpenClaw's Dictation relay sends encoding=g711_ulaw at 8000 Hz.
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=g711_ulaw&sample_rate=8000") as ws:
        ws.send_bytes(mulaw_speech(0.6))
        ws.send_json({"type": "Finalize"})
        assert ws.receive_json()["type"] == "Results"


def test_g711_alaw_alias_is_accepted_and_transcribes():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=g711_alaw&sample_rate=8000") as ws:
        ws.send_bytes(alaw_speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"
    assert final["channel"]["alternatives"][0]["transcript"] == TRANSCRIPT


def test_g711_defaults_to_8000_when_sample_rate_omitted():
    client, manager = make_client()
    with client.websocket_connect("/v1/listen?encoding=mulaw") as ws:
        ws.send_bytes(mulaw_speech(0.6))
        ws.send_json({"type": "Finalize"})
        assert ws.receive_json()["type"] == "Results"
    # 0.6 s of 8 kHz mulaw resampled to 16 kHz -> ~9600 samples at the decoder.
    assert 9000 <= manager.calls[-1]["samples"] <= 10000


def test_pure_silence_in_mulaw_never_emits_a_final():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen?encoding=mulaw&sample_rate=8000") as ws:
        ws.send_bytes(mulaw_speech(0.5, byte=0xFF))  # 0xFF decodes to 0.0
        ws.send_json({"type": "CloseStream"})
        metadata = ws.receive_json()
    assert metadata["type"] == "Metadata"


def test_unknown_message_type_is_rejected():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_json({"type": "Wat"})
        error = ws.receive_json()
    assert error["type"] == "Error"
    assert error["err_code"] == "Bad Request"


def test_invalid_json_is_rejected():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_text("{not json")
        error = ws.receive_json()
    assert error["type"] == "Error"
    assert "Invalid JSON" in error["err_msg"]


def test_odd_length_linear16_frame_is_rejected():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_bytes(b"\x01\x02\x03")  # 3 bytes is not a whole int16 sample
        error = ws.receive_json()
    assert error["type"] == "Error"


# -- authentication --------------------------------------------------------


def test_auth_denied_without_token_when_api_key_set():
    client, _ = make_client(api_key="secret")
    with pytest.raises(WebSocketDenialResponse) as excinfo:
        with client.websocket_connect("/v1/listen"):
            pass
    assert excinfo.value.status_code == 401
    assert excinfo.value.json()["err_code"] == "INVALID_AUTH"


def test_auth_accepts_authorization_token_header():
    client, _ = make_client(api_key="secret")
    with client.websocket_connect(
        "/v1/listen", headers={"Authorization": "Token secret"}
    ) as ws:
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"


def test_auth_accepts_api_key_query_parameter():
    client, _ = make_client(api_key="secret")
    with client.websocket_connect("/v1/listen?api_key=secret") as ws:
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"


def test_auth_accepts_sec_websocket_protocol():
    client, _ = make_client(api_key="secret")
    with client.websocket_connect("/v1/listen", subprotocols=["token", "secret"]) as ws:
        assert ws.accepted_subprotocol == "token"
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        final = ws.receive_json()
    assert final["type"] == "Results"


def test_auth_rejects_wrong_token():
    client, _ = make_client(api_key="secret")
    with pytest.raises(WebSocketDenialResponse) as excinfo:
        with client.websocket_connect("/v1/listen", headers={"Authorization": "Token nope"}):
            pass
    assert excinfo.value.status_code == 401


def test_auth_is_skipped_when_api_key_unset():
    client, _ = make_client()
    with client.websocket_connect("/v1/listen") as ws:
        ws.send_bytes(speech(0.6))
        ws.send_json({"type": "Finalize"})
        assert ws.receive_json()["type"] == "Results"


# -- audio helpers (unit) --------------------------------------------------


def test_decode_linear16_scales_to_float32():
    payload = (np.full(100, 0.5, dtype=np.float32) * 32767.0).astype("<i2").tobytes()
    decoded = decode_linear16(payload, sample_rate=SAMPLE_RATE)
    assert decoded.dtype == np.float32
    assert decoded.size == 100
    assert 0.45 < float(decoded.mean()) < 0.55


def test_decode_linear16_resamples_8k_to_16k():
    payload = (np.full(8000, 0.25, dtype=np.float32) * 32767.0).astype("<i2").tobytes()
    decoded = decode_linear16(payload, sample_rate=8000)
    assert decoded.size == 16000


def test_decode_linear16_rejects_odd_length():
    with pytest.raises(Exception):
        decode_linear16(b"\x01\x02\x03", sample_rate=SAMPLE_RATE)


def test_decode_ulaw_matches_ffmpeg_reference_values():
    # Table validated byte-for-byte against ffmpeg's mulaw decoder.
    def d(code: int) -> int:
        return round(float(decode_ulaw(bytes([code]), sample_rate=8000, target_rate=8000)[0]) * 32768)

    assert d(0x00) == -32124
    assert d(0x80) == 32124
    assert d(0xFF) == 0


def test_decode_alaw_matches_ffmpeg_reference_values():
    def d(code: int) -> int:
        return round(float(decode_alaw(bytes([code]), sample_rate=8000, target_rate=8000)[0]) * 32768)

    assert d(0x00) == -5504
    assert d(0xAA) == 32256
    # A-law has no exact zero: the smallest magnitudes are -8 and +8.
    assert d(0x55) == -8
    assert d(0xD5) == 8


def test_decode_g711_resamples_8k_to_16k():
    assert decode_ulaw(bytes([0x80]) * 8000, sample_rate=8000).size == 16000
    assert decode_alaw(bytes([0xAA]) * 8000, sample_rate=8000).size == 16000


def test_normalize_encoding_aliases():
    assert normalize_encoding("linear16") == "linear16"
    assert normalize_encoding("pcm16") == "linear16"
    assert normalize_encoding("mulaw") == "mulaw"
    assert normalize_encoding("G711_ULAW") == "mulaw"
    assert normalize_encoding("ulaw") == "mulaw"
    assert normalize_encoding("g711_alaw") == "alaw"
    with pytest.raises(Exception):
        normalize_encoding("opus")


def test_resample_linear_downsample_length():
    source = np.zeros(16000, dtype=np.float32)
    assert resample_linear(source, 16000, 8000).size == 8000


def test_rms_silence_and_signal():
    assert rms(np.zeros(64, dtype=np.float32)) == 0.0
    assert rms(np.full(64, 0.5, dtype=np.float32)) == pytest.approx(0.5)
    assert rms(np.empty(0, dtype=np.float32)) == 0.0


# -- static UI + root ------------------------------------------------------


def test_ui_listen_page_is_served():
    client, _ = make_client()
    with client:
        res = client.get("/ui/listen")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "ストリーミング文字起こし" in res.text
    assert "/v1/listen" in res.text


def test_root_advertises_listen_endpoint():
    client, _ = make_client()
    with client:
        body = client.get("/").json()
    assert "WS /v1/listen" in body["endpoints"]
    assert "GET /ui/listen" in body["endpoints"]
