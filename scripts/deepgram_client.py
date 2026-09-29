"""Deepgram compatible streaming client for the local ``/v1/listen`` endpoint.

Usage::

    uv run python scripts/deepgram_client.py /path/to/audio.wav
    uv run python scripts/deepgram_client.py audio.wav --language ja \
        --interim-results true --finalize --api-key secret

The audio is decoded to 16 kHz mono, converted to little-endian ``linear16``
PCM and streamed as raw binary WebSocket frames (exactly what Deepgram's
``Media`` message expects). Control messages (``Finalize`` / ``CloseStream``)
are sent as JSON, and every server message is printed as it arrives.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from whisper_local.transcriber import AudioDecodeError
from faster_whisper.audio import decode_audio

DEFAULT_URL = "ws://127.0.0.1:8000/v1/listen"
SAMPLE_RATE = 16000
FRAME_SECONDS = 0.5


def decode_to_pcm(path: str) -> bytes:
    """Decode any PyAV-supported file into 16 kHz mono ``linear16`` bytes."""
    samples = decode_audio(path, sampling_rate=SAMPLE_RATE)
    audio = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def build_url(args: argparse.Namespace) -> str:
    params = [
        ("encoding", "linear16"),
        ("sample_rate", str(SAMPLE_RATE)),
        ("channels", "1"),
        ("language", args.language),
        ("interim_results", "true" if args.interim_results else "false"),
        ("vad_events", "true" if args.vad_events else "false"),
    ]
    if args.endpointing is not None:
        params.append(("endpointing", str(args.endpointing)))
    if args.utterance_end_ms is not None:
        params.append(("utterance_end_ms", str(args.utterance_end_ms)))
    if args.model:
        params.append(("model", args.model))
    query = "&".join(f"{key}={value}" for key, value in params)
    separator = "&" if "?" in args.url else "?"
    return f"{args.url}{separator}{query}"


def print_message(raw: str) -> None:
    try:
        payload = json.loads(raw)
    except ValueError:
        print(f"[server] {raw}")
        return
    kind = payload.get("type")
    if kind == "Results":
        alternatives = payload.get("channel", {}).get("alternatives", [])
        transcript = alternatives[0].get("transcript", "") if alternatives else ""
        flags = [
            f"is_final={payload.get('is_final')}",
            f"speech_final={payload.get('speech_final')}",
            f"from_finalize={payload.get('from_finalize', False)}",
        ]
        print(
            f"[Results] {' '.join(flags)} start={payload.get('start')} "
            f"dur={payload.get('duration')} text={transcript!r}"
        )
    elif kind == "Metadata":
        print(f"[Metadata] {json.dumps(payload, ensure_ascii=False)}")
    elif kind == "UtteranceEnd":
        print(f"[UtteranceEnd] channel={payload.get('channel')} last_word_end={payload.get('last_word_end')}")
    elif kind == "SpeechStarted":
        print(f"[SpeechStarted] channel={payload.get('channel')} timestamp={payload.get('timestamp')}")
    elif kind == "Error":
        err = json.dumps(payload, ensure_ascii=False)
        print(f"[Error] {err}")
        raise SystemExit(f"server returned an error: {err}")
    else:
        print(f"[server] {json.dumps(payload, ensure_ascii=False)}")


def run(args: argparse.Namespace) -> int:
    try:
        pcm = decode_to_pcm(args.audio)
    except AudioDecodeError as exc:
        print(f"failed to decode {args.audio!r}: {exc}", file=sys.stderr)
        return 2

    url = build_url(args)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Token {args.api_key}"
        if args.extra_header_token:
            headers["x-deepgram-session-id"] = "local-poc"

    total_samples = len(pcm) // 2
    print(f"connecting: {url}")
    print(f"audio: {args.audio} -> {total_samples / SAMPLE_RATE:.2f}s of 16 kHz mono linear16")

    frame_bytes = int(FRAME_SECONDS * SAMPLE_RATE) * 2
    with connect(url, additional_headers=headers) as ws:
        print("[client] connected; streaming audio")
        for offset in range(0, len(pcm), frame_bytes):
            ws.send(pcm[offset : offset + frame_bytes])
            if args.realtime:
                time.sleep(FRAME_SECONDS)

        if args.finalize:
            print("[client] send Finalize")
            ws.send(json.dumps({"type": "Finalize"}))
            time.sleep(0.2)

        print("[client] send CloseStream")
        ws.send(json.dumps({"type": "CloseStream"}))

        try:
            while True:
                print_message(ws.recv(timeout=args.recv_timeout))
        except TimeoutError:
            print("[client] no more messages (timeout)")
        except ConnectionClosed as exc:
            print(f"[client] closed: code={exc.rcvd.code if exc.rcvd else '?'} "
                  f"reason={exc.rcvd.reason if exc.rcvd else ''!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="audio file (wav and anything PyAV can open)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"listen endpoint (default {DEFAULT_URL})")
    parser.add_argument("--api-key", default=None, help="sent as 'Authorization: Token <key>'")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpointing", type=int, default=None, help="endpointing ms")
    parser.add_argument("--utterance-end-ms", type=int, default=None)
    parser.add_argument("--interim-results", action="store_true")
    parser.add_argument("--vad-events", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="send Finalize before CloseStream")
    parser.add_argument("--realtime", action="store_true", help="pace frames in real time")
    parser.add_argument("--recv-timeout", type=float, default=30.0)
    parser.add_argument(
        "--extra-header-token",
        action="store_true",
        help="also send x-deepgram-session-id (SDK-like extra header)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
