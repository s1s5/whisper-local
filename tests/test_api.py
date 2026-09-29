"""API contract tests.

These tests use a stub :class:`ModelManager` so the suite stays fast and does
not need the 1.5 GB model. The real model is exercised by the smoke checks in
``thoughts/implementation-report.md``.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from whisper_local.config import Settings
from whisper_local.server.app import create_app
from whisper_local.transcriber import AudioDecodeError, TranscriptionResult

AUDIO = b"RIFF....WAVEfmt "  # contents are irrelevant for the stub manager


@dataclasses.dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.99


@dataclasses.dataclass
class FakeSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list


@dataclasses.dataclass
class FakeInfo:
    language: str = "ja"
    language_probability: float = 0.99
    duration: float = 3.288


class StubManager:
    """Minimal stand-in for ModelManager."""

    def __init__(self, words_enabled: bool = True, fail: bool = False) -> None:
        self.words_enabled = words_enabled
        self.loaded = True
        self.device = "cpu"
        self.compute_type = "int8"
        self.load_notes: list[str] = []
        self.fail = fail
        self.decode_error: Exception | None = None
        self.calls: list[dict] = []

    def load(self) -> None:
        self.loaded = True

    def decode(self, audio):
        if self.decode_error is not None:
            raise self.decode_error
        return audio

    def transcribe(self, audio, **kwargs) -> TranscriptionResult:
        self.calls.append({"audio": audio, **kwargs})
        if self.decode_error is not None:
            raise self.decode_error
        if self.fail:
            raise RuntimeError("boom")
        words = [FakeWord("ニュース", 0.0, 0.6), FakeWord("と", 0.6, 0.74)]
        segment = FakeSegment(
            id=0,
            seek=0,
            start=0.0,
            end=2.38,
            text="ニュースと天気をミラーに表示して",
            tokens=[50365, 34737],
            temperature=0.0,
            avg_logprob=-0.0258,
            compression_ratio=1.2,
            no_speech_prob=0.01,
            words=words if self.words_enabled and kwargs.get("want_words") else [],
        )
        return TranscriptionResult(
            segments=[segment],
            info=FakeInfo(),
            task=kwargs.get("task", "transcribe"),
            words_available=self.words_enabled,
            word_timestamps_requested=bool(kwargs.get("want_words")),
        )


def make_client(**kwargs):
    # Tests opt into recording persistence explicitly so the suite never
    # writes into the repository's ``var/recordings`` directory.
    kwargs.setdefault("save_audio", False)
    settings = Settings(api_key=kwargs.pop("api_key", None), **kwargs)
    manager = StubManager()
    app = create_app(settings=settings, manager=manager)
    return TestClient(app), manager


def post_audio(client, data=None, path="/v1/audio/transcriptions", **kwargs):
    data = dict(data or {})
    return client.post(
        path,
        files={"file": ("sample.wav", AUDIO, "audio/wav")},
        data=data,
        **kwargs,
    )


# -- happy paths -----------------------------------------------------------


def test_json_response_format():
    client, _ = make_client()
    with client:
        res = post_audio(client)
    assert res.status_code == 200
    assert res.json() == {"text": "ニュースと天気をミラーに表示して"}


def test_text_response_format():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"response_format": "text"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert res.text == "ニュースと天気をミラーに表示して"


def test_verbose_json_shape():
    client, _ = make_client()
    with client:
        res = post_audio(
            client, {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
        )
    assert res.status_code == 200
    body = res.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "ja"
    assert body["duration"] == 3.288
    assert body["usage"] == {"type": "duration", "seconds": 3}
    assert len(body["segments"]) == 1
    segment = body["segments"][0]
    for key in (
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "temperature",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
    ):
        assert key in segment, key
    assert segment["seek"] == 0
    assert segment["tokens"] == [50365, 34737]
    assert body["words"][0]["word"] == "ニュース"
    assert set(body["words"][0]) == {"word", "start", "end"}


def test_verbose_json_without_word_granularity_has_empty_words():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"response_format": "verbose_json"})
    assert res.status_code == 200
    assert res.json()["words"] == []


def test_srt_response_format():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"response_format": "srt"})
    assert res.status_code == 200
    assert res.text.startswith("1\n00:00:00,000 --> 00:00:02,380\n")
    assert "ニュースと天気をミラーに表示して" in res.text


def test_vtt_response_format():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"response_format": "vtt"})
    assert res.status_code == 200
    assert res.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.380" in res.text


def test_language_is_passed_through_and_optional():
    client, manager = make_client()
    with client:
        post_audio(client, {"language": "ja"})
        post_audio(client)
    assert manager.calls[0]["language"] == "ja"
    assert manager.calls[1]["language"] is None


def test_prompt_maps_to_initial_prompt():
    client, manager = make_client()
    with client:
        post_audio(client, {"prompt": "HMI"})
    assert manager.calls[0]["initial_prompt"] == "HMI"


def test_translations_endpoint_sets_task_translate():
    client, manager = make_client()
    with client:
        res = post_audio(client, {"response_format": "verbose_json"}, path="/v1/audio/translations")
    assert res.status_code == 200
    assert manager.calls[0]["task"] == "translate"
    assert res.json()["task"] == "translate"


def test_models_endpoint():
    client, _ = make_client()
    with client:
        res = client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == Settings().model
    assert body["data"][0]["object"] == "model"


def test_healthz_endpoint():
    client, _ = make_client()
    with client:
        res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["words_available"] is True


def test_healthcheck_endpoint_with_trailing_slash():
    client, _ = make_client()
    with client:
        res = client.get("/-/healthcheck/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_healthcheck_endpoint_without_trailing_slash_is_not_a_redirect():
    client, _ = make_client()
    with client:
        res = client.get("/-/healthcheck", follow_redirects=False)
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_healthcheck_endpoint_skips_auth_when_api_key_set():
    client, _ = make_client(api_key="secret")
    with client:
        with_slash = client.get("/-/healthcheck/")
        without_slash = client.get("/-/healthcheck")
    assert with_slash.status_code == 200
    assert with_slash.json() == {"status": "ok"}
    assert without_slash.status_code == 200


def test_root_endpoint():
    client, _ = make_client()
    with client:
        res = client.get("/")
    assert res.status_code == 200
    assert res.json()["name"] == "whisper-local"


def test_root_endpoint_advertises_ui():
    client, _ = make_client()
    with client:
        res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["ui"] == "/ui"
    assert "GET /ui" in body["endpoints"]


# -- recording UI (plan section 6) -----------------------------------------


def test_ui_endpoint_serves_html():
    client, _ = make_client()
    with client:
        res = client.get("/ui")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<title>whisper-local 録音</title>" in res.text
    assert "録音開始" in res.text
    assert "停止して認識" in res.text


def test_ui_directory_index_is_not_a_redirect():
    client, _ = make_client()
    with client:
        res = client.get("/ui/", follow_redirects=False)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")


def test_ui_is_not_protected_by_auth():
    client, _ = make_client(api_key="secret")
    with client:
        ui = client.get("/ui")
        api = post_audio(client)
    assert ui.status_code == 200
    assert api.status_code == 401


def test_recording_is_saved_with_sidecar_by_default(tmp_path):
    import json

    client, _ = make_client(save_audio=True, save_audio_dir=tmp_path)
    with client:
        res = post_audio(client, {"response_format": "verbose_json"})
    assert res.status_code == 200

    audio_files = sorted(tmp_path.rglob("*.wav"))
    assert len(audio_files) == 1
    saved = audio_files[0]
    assert saved.read_bytes() == AUDIO
    # <dir>/<YYYY-MM-DD>/<HHMMSS>-<6 digits>.<ext>
    assert saved.parent.name.count("-") == 2
    assert saved.stem[-7] == "-"
    assert saved.stem[-6:].isdigit()

    sidecar = saved.with_suffix(".json")
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["model"] == Settings().model
    assert payload["audio_original_filename"] == "sample.wav"
    assert payload["audio_content_type"] == "audio/wav"
    assert payload["audio_bytes"] == len(AUDIO)
    assert payload["result"]["text"] == "ニュースと天気をミラーに表示して"


def test_recording_save_can_be_disabled(tmp_path):
    client, _ = make_client(save_audio=False, save_audio_dir=tmp_path)
    with client:
        res = post_audio(client)
    assert res.status_code == 200
    assert list(tmp_path.rglob("*")) == []


def test_save_audio_defaults_to_enabled(tmp_path):
    from whisper_local.config import DEFAULT_SAVE_AUDIO_DIR

    settings = Settings.from_env({})
    assert settings.save_audio is True
    assert settings.save_audio_dir == DEFAULT_SAVE_AUDIO_DIR


def test_save_audio_env_overrides():
    settings = Settings.from_env(
        {"WHISPER_SAVE_AUDIO": "false", "WHISPER_SAVE_AUDIO_DIR": "/tmp/whisper-recordings"}
    )
    assert settings.save_audio is False
    assert str(settings.save_audio_dir) == "/tmp/whisper-recordings"


def test_saved_recording_is_downloadable_and_requires_auth(tmp_path):
    client, _ = make_client(save_audio=True, save_audio_dir=tmp_path)
    with client:
        post_audio(client)
        saved = next(tmp_path.rglob("*.wav"))
        url = f"/api/recordings/{saved.parent.name}/{saved.name}"
        res = client.get(url)
    assert res.status_code == 200
    assert res.content == AUDIO

    protected, _ = make_client(api_key="secret", save_audio_dir=tmp_path)
    with protected:
        denied = protected.get(url)
        allowed = protected.get(url, headers={"Authorization": "Bearer secret"})
    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_recording_url_rejects_path_traversal(tmp_path):
    client, _ = make_client(save_audio_dir=tmp_path)
    with client:
        res = client.get("/api/recordings/2026-09-29/..%2F..%2F.env")
        missing = client.get("/api/recordings/2026-09-29/nope.wav")
    assert res.status_code == 404
    assert missing.status_code == 404


# -- condition_on_previous_text (long-audio truncation fix) -----------------


class RecordingModel:
    """Stand-in ``WhisperModel`` that records the kwargs passed to ``transcribe``."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs
        return [], FakeInfo()


def test_condition_on_previous_text_defaults_to_false():
    assert Settings().condition_on_previous_text is False
    assert Settings.from_env({}).condition_on_previous_text is False


def test_condition_on_previous_text_env_overrides():
    assert (
        Settings.from_env({"WHISPER_CONDITION_ON_PREVIOUS_TEXT": "true"}).condition_on_previous_text
        is True
    )
    assert (
        Settings.from_env({"WHISPER_CONDITION_ON_PREVIOUS_TEXT": "false"}).condition_on_previous_text
        is False
    )


def test_condition_on_previous_text_is_passed_to_transcribe_by_default():
    from whisper_local.transcriber import ModelManager

    manager = ModelManager(Settings(save_audio=False))
    model = RecordingModel()
    manager._model = model

    manager.transcribe_array([0.0] * 16, language="ja")

    assert model.kwargs is not None
    assert model.kwargs["condition_on_previous_text"] is False


def test_condition_on_previous_text_override_is_passed_to_transcribe():
    from whisper_local.transcriber import ModelManager

    manager = ModelManager(Settings(save_audio=False, condition_on_previous_text=True))
    model = RecordingModel()
    manager._model = model

    manager.transcribe_array([0.0] * 16, language="ja")

    assert model.kwargs is not None
    assert model.kwargs["condition_on_previous_text"] is True


# -- error paths -----------------------------------------------------------


def test_temperature_nonzero_is_400_with_envelope():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"temperature": "1.0"})
    assert res.status_code == 400
    body = res.json()
    assert body["error"]["param"] == "temperature"
    assert set(body["error"]) == {"message", "type", "param", "code"}


def test_invalid_response_format_is_400():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"response_format": "xml"})
    assert res.status_code == 400
    assert res.json()["error"]["param"] == "response_format"


def test_invalid_granularity_is_400():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"timestamp_granularities[]": "phoneme"})
    assert res.status_code == 400
    assert res.json()["error"]["param"] == "timestamp_granularities[]"


def test_stream_is_rejected():
    client, _ = make_client()
    with client:
        res = post_audio(client, {"stream": "true"})
    assert res.status_code == 400


def test_empty_upload_is_400():
    client, _ = make_client()
    with client:
        res = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
    assert res.status_code == 400
    assert res.json()["error"]["param"] == "file"


def test_missing_file_is_422():
    client, _ = make_client()
    with client:
        res = client.post("/v1/audio/transcriptions", data={"model": "x"})
    assert res.status_code == 422


def test_auth_required_when_api_key_set():
    client, _ = make_client(api_key="secret")
    with client:
        res = post_audio(client)
        ok = post_audio(client, headers={"Authorization": "Bearer secret"})
        bad = post_audio(client, headers={"Authorization": "Bearer nope"})
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "invalid_api_key"
    assert ok.status_code == 200
    assert bad.status_code == 401


def test_auth_skipped_when_api_key_unset():
    client, _ = make_client()
    with client:
        res = post_audio(client)
    assert res.status_code == 200


def test_upload_limit_is_413():
    client, _ = make_client(max_upload_mb=0)
    with client:
        res = post_audio(client)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "file_too_large"


def test_inference_failure_returns_500_envelope_and_keeps_server_alive():
    client, manager = make_client()
    manager.fail = True
    with client:
        res = post_audio(client)
        assert res.status_code == 500
        body = res.json()
        assert body["error"]["type"] == "server_error"
        manager.fail = False
        ok = post_audio(client)
    assert ok.status_code == 200


def test_undecodable_audio_is_400_with_envelope():
    client, manager = make_client()
    manager.decode_error = AudioDecodeError("cannot decode audio: bogus")
    with client:
        res = post_audio(client)
        assert res.status_code == 400
        body = res.json()
        assert body["error"]["param"] == "file"
        assert body["error"]["code"] == "invalid_audio"
        manager.decode_error = None
        ok = post_audio(client)
    assert ok.status_code == 200


def test_words_disabled_degrades_without_error():
    settings = Settings(save_audio=False)
    manager = StubManager(words_enabled=False)
    client = TestClient(create_app(settings=settings, manager=manager))
    with client:
        res = post_audio(
            client, {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
        )
    assert res.status_code == 200
    assert res.json()["words"] == []
