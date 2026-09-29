"""Phase 4 foundation: point the official ``deepgram-python-sdk`` at this server.

The SDK builds its URL as ``environment.production + "/v1/listen"``, so we
override ``DeepgramClientEnvironment.production`` with the local host. This
script is the *tooling* for Phase 4 (SDK compatibility verification); it is not
part of Phase 1-3 and is expected to surface incompatibilities rather than to
pass silently.

Usage::

    uv add deepgram
    uv run python scripts/deepgram_sdk_check.py /path/to/audio.wav

If the SDK is not installed the script prints an install hint and exits 0 so it
never breaks CI when only Phase 1-3 are in scope.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.5


def decode_to_pcm(path: str) -> bytes:
    samples = np.asarray(decode_audio(path, sampling_rate=SAMPLE_RATE), dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def run(args: argparse.Namespace) -> int:
    try:
        from deepgram import DeepgramClient
        from deepgram.environment import DeepgramClientEnvironment
    except ImportError:
        print(
            "deepgram SDK is not installed. Install it to run the Phase 4 check:\n"
            "  uv add deepgram",
            file=sys.stderr,
        )
        return 0

    pcm = decode_to_pcm(args.audio)
    print(f"audio: {args.audio} -> {len(pcm) // 2 / SAMPLE_RATE:.2f}s of linear16 PCM")

    environment = DeepgramClientEnvironment(
        base=f"http://{args.host}:{args.port}",
        production=f"ws://{args.host}:{args.port}",
        agent=f"ws://{args.host}:{args.port}",
        agent_rest=f"http://{args.host}:{args.port}",
    )
    client = DeepgramClient(api_key=args.api_key, environment=environment)

    frame_bytes = int(FRAME_SECONDS * SAMPLE_RATE) * 2
    received: list[str] = []

    try:
        with client.listen.v1.connect(
            model=args.model,
            language=args.language,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
            interim_results=True,
            vad_events=True,
            endpointing=args.endpointing,
            utterance_end_ms=args.utterance_end_ms,
        ) as socket:
            for offset in range(0, len(pcm), frame_bytes):
                socket.send_media(pcm[offset : offset + frame_bytes])
            socket.send_finalize()
            socket.send_close_stream()

            try:
                for message in socket:
                    kind = getattr(message, "type", type(message).__name__)
                    received.append(str(kind))
                    print(f"[sdk] {kind}: {message!r}")
            except Exception as exc:  # noqa: BLE001 - report, do not hide
                print(f"[sdk] stream ended: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - this is a compatibility probe
        print(
            f"SDK connection failed: {type(exc).__name__}: {exc}\n"
            "Record this as an SDK incompatibility point in the Phase 4 report.",
            file=sys.stderr,
        )
        return 1

    print(f"[sdk] received {len(received)} messages: {received}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="local-poc")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--endpointing", type=int, default=500)
    parser.add_argument("--utterance-end-ms", type=int, default=1500)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
