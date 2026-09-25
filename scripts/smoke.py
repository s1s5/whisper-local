"""Smoke checks against a running server (httpx, OpenAI-SDK-equivalent calls).

Usage: ``uv run python scripts/smoke.py [audio]``
Requires the server to be listening on 127.0.0.1:8000.
"""

from __future__ import annotations

import sys

import httpx

BASE = "http://127.0.0.1:8000/v1"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/shogo/projects/tomody/d-sha/wavs_ja/01_disp_hmi_scene.wav"
)
# The OpenAI SDK always sends an Authorization header, even for local servers.
HEADERS = {"Authorization": "Bearer dummy"}


def sdk_equivalent_transcription() -> None:
    """Same call shape openai-python uses for audio.transcriptions.create()."""
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=120.0) as client:
        with open(AUDIO, "rb") as fh:
            response = client.post(
                "/audio/transcriptions",
                files={"file": ("01_disp_hmi_scene.wav", fh, "audio/wav")},
                data={"model": "whisper-1", "response_format": "json"},
            )
    response.raise_for_status()
    print("[sdk-equivalent] status", response.status_code, "body", response.json())


def models_listing() -> None:
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30.0) as client:
        response = client.get("/models")
    response.raise_for_status()
    print("[models]", response.json())


def ten_consecutive_requests() -> None:
    codes = []
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=180.0) as client:
        for index in range(10):
            with open(AUDIO, "rb") as fh:
                response = client.post(
                    "/audio/transcriptions",
                    files={"file": ("01_disp_hmi_scene.wav", fh, "audio/wav")},
                    data={
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                    },
                )
            text = ""
            if response.status_code == 200:
                text = response.json().get("text", "")
            codes.append(response.status_code)
            print(f"[{index + 1:02d}/10] status={response.status_code} text={text!r}")
    failures = [code for code in codes if code != 200]
    print("[10x] statuses", codes, "failures", failures)
    if failures:
        raise SystemExit("FAILED: non-200 responses in the ten-request loop")


if __name__ == "__main__":
    models_listing()
    sdk_equivalent_transcription()
    ten_consecutive_requests()
    print("ALL SMOKE CHECKS PASSED")
