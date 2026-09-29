"""FastAPI application exposing an OpenAI compatible Audio API."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, AsyncIterator, Iterable, Optional

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from whisper_local import __version__, schemas
from whisper_local.config import Settings
from whisper_local.server.openai_types import (
    RESPONSE_FORMATS,
    HealthResponse,
    ModelCard,
    ModelList,
    error_payload,
)
from whisper_local.transcriber import AudioDecodeError, ModelManager, ModelLoadError

log = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
MULTIPART_OVERHEAD_ALLOWANCE = 1024 * 1024

STATIC_DIR = Path(__file__).parent / "static"

# Content-Type -> file extension for saved recordings (plan section 5.4).
_AUDIO_EXTENSIONS = {
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "video/mp4": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}
_SAFE_EXTENSION_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def _extension_from_content_type(content_type: Optional[str]) -> str:
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in _AUDIO_EXTENSIONS:
            return _AUDIO_EXTENSIONS[base]
    return "bin"


def _extension_from_filename(filename: Optional[str]) -> Optional[str]:
    """Return a sanitized extension (no dot) from the uploaded file name."""
    if not filename:
        return None
    suffix = Path(filename).suffix.lstrip(".").lower()
    if not suffix or len(suffix) > 8 or not set(suffix) <= _SAFE_EXTENSION_CHARS:
        return None
    return suffix


def _recording_extension(upload: UploadFile) -> str:
    return (
        _extension_from_filename(upload.filename)
        or _extension_from_content_type(upload.content_type)
    )


def _recording_path(directory: Path, extension: str) -> Path:
    """<dir>/<YYYY-MM-DD>/<HHMMSS>-<6 digits>.<ext> (plan section 5.4)."""
    now = datetime.now()
    stamp = f"{now:%H%M%S}-{uuid.uuid4().int % 1_000_000:06d}"
    return directory / f"{now:%Y-%m-%d}" / f"{stamp}.{extension}"


class ApiError(Exception):
    """An error that maps straight onto an OpenAI error envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str = "invalid_request_error",  # noqa: A002 - OpenAI field name
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.param = param
        self.code = code


def _error_response(err: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=err.status_code,
        content=error_payload(err.message, type=err.type, param=err.param, code=err.code),
    )


def create_app(settings: Settings | None = None, manager: ModelManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = manager or ModelManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await run_in_threadpool(manager.load)
        except ModelLoadError:
            log.exception("model failed to load; the API will return 500 on inference")
        yield
        log.info("shutting down")

    app = FastAPI(
        title="whisper-local",
        version=__version__,
        description="OpenAI compatible Audio API backed by faster-whisper",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.manager = manager

    # -- CORS (plan section 5.2) -------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,  # "*" and credentials cannot be combined
    )

    # -- auth --------------------------------------------------------------

    async def require_auth(authorization: Annotated[Optional[str], Header()] = None) -> None:
        expected = settings.api_key
        if not expected:
            return
        provided = None
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if provided != expected:
            raise ApiError(
                401,
                "Incorrect API key provided.",
                type="invalid_request_error",
                param=None,
                code="invalid_api_key",
            )

    # -- error handling ----------------------------------------------------

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error while serving %s", request.url.path)
        return _error_response(
            ApiError(500, f"Internal server error: {type(exc).__name__}", type="server_error")
        )

    # -- helpers -----------------------------------------------------------

    def parse_temperature(raw: Optional[str], param: str = "temperature") -> float:
        if raw is None or raw == "":
            return 0.0
        try:
            value = float(raw)
        except ValueError:
            raise ApiError(400, f"Invalid value for {param!r}: {raw!r}", param=param) from None
        if value != 0:
            raise ApiError(
                400,
                "Only temperature=0 is supported (greedy decoding).",
                param=param,
                code="unsupported_value",
            )
        return value

    def parse_response_format(raw: Optional[str]) -> str:
        value = (raw or "json").strip().lower()
        if value not in RESPONSE_FORMATS:
            raise ApiError(
                400,
                f"Invalid response_format {raw!r}; expected one of "
                f"{', '.join(RESPONSE_FORMATS)}.",
                param="response_format",
            )
        return value

    def parse_granularities(values: Iterable[str]) -> set[str]:
        granularities: set[str] = set()
        for value in values:
            normalized = (value or "").strip().lower()
            if not normalized:
                continue
            if normalized not in (schemas.GRANULARITY_SEGMENT, schemas.GRANULARITY_WORD):
                raise ApiError(
                    400,
                    f"Invalid timestamp_granularities value {value!r}; expected "
                    "'segment' or 'word'.",
                    param="timestamp_granularities[]",
                )
            granularities.add(normalized)
        return granularities or {schemas.GRANULARITY_SEGMENT}

    async def save_upload(upload: UploadFile) -> str:
        """Stream the upload to a temp file, enforcing the size limit."""
        limit_bytes = settings.max_upload_mb * 1024 * 1024
        fd, path = tempfile.mkstemp(prefix="whisper-local-", suffix=".upload")
        total = 0
        try:
            with os.fdopen(fd, "wb") as sink:
                while True:
                    chunk = await upload.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit_bytes:
                        raise ApiError(
                            413,
                            f"Audio file exceeds the maximum size of "
                            f"{settings.max_upload_mb} MB.",
                            param="file",
                            code="file_too_large",
                        )
                    sink.write(chunk)
        except ApiError:
            os.unlink(path)
            raise
        except Exception as exc:  # noqa: BLE001
            os.unlink(path)
            raise ApiError(400, f"Failed to read uploaded file: {exc}", param="file") from exc
        finally:
            await upload.close()

        if total == 0:
            os.unlink(path)
            raise ApiError(400, "Uploaded file is empty.", param="file")
        return path

    def check_request_size(request: Request) -> None:
        """Reject obviously oversized bodies before buffering (best effort)."""
        raw_length = request.headers.get("content-length")
        if not raw_length:
            return
        try:
            length = int(raw_length)
        except ValueError:
            return
        limit = (settings.max_upload_mb * 1024 * 1024) + MULTIPART_OVERHEAD_ALLOWANCE
        if length > limit:
            raise ApiError(
                413,
                f"Request body exceeds the maximum size of {settings.max_upload_mb} MB.",
                param="file",
                code="file_too_large",
            )

    def build_response(result, response_format: str) -> Response:
        if response_format == "json":
            return JSONResponse(content=schemas.to_json(result))
        if response_format == "text":
            return PlainTextResponse(content=schemas.to_text(result))
        if response_format == "srt":
            return Response(
                content=schemas.segments_srt(result),
                media_type="application/x-subrip",
            )
        if response_format == "vtt":
            return Response(content=schemas.segments_vtt(result), media_type="text/vtt")
        return JSONResponse(content=schemas.to_verbose_json(result))

    def verbose_payload(result) -> dict:
        """The sidecar JSON body handed to :func:`save_recording`."""
        return schemas.to_verbose_json(result)

    def save_recording(
        temp_path: str,
        upload: UploadFile,
        *,
        processing_seconds: float,
        result,
    ) -> Optional[str]:
        """Copy the (already transcribed) upload into ``var/recordings``.

        Returns the browser-facing URL of the saved audio, or ``None`` when
        saving is disabled or fails. Failures are logged and swallowed: saving
        must never change the outcome of a transcription (plan section 5.4).
        """
        try:
            directory = Path(settings.save_audio_dir)
            destination = _recording_path(directory, _recording_extension(upload))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temp_path, destination)
            audio_url = f"/api/recordings/{destination.parent.name}/{destination.name}"
            sidecar = destination.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "model": settings.model,
                        "created": int(time.time()),
                        "processing_seconds": round(processing_seconds, 3),
                        "audio_url": audio_url,
                        "audio_original_filename": upload.filename,
                        "audio_content_type": upload.content_type,
                        "audio_bytes": destination.stat().st_size,
                        "result": verbose_payload(result),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.info("saved recording: %s -> %s", audio_url, destination)
            return audio_url
        except Exception:  # noqa: BLE001 - saving is best effort
            log.warning(
                "failed to save recording %s", getattr(upload, "filename", "?"), exc_info=True
            )
            return None

    async def handle_audio_request(
        request: Request,
        file: UploadFile,
        model: Optional[str],
        language: Optional[str],
        prompt: Optional[str],
        response_format: Optional[str],
        temperature: Optional[str],
        timestamp_granularities: list[str],
        task: str,
    ) -> Response:
        check_request_size(request)
        fmt = parse_response_format(response_format)
        temp_value = parse_temperature(temperature)
        granularities = parse_granularities(timestamp_granularities)
        want_words = schemas.GRANULARITY_WORD in granularities

        if not manager.loaded:
            # One more attempt: a transient failure at startup should not
            # permanently break the endpoint.
            try:
                await run_in_threadpool(manager.load)
            except ModelLoadError as exc:
                raise ApiError(500, f"Model is not available: {exc}", type="server_error") from exc

        path = await save_upload(file)
        try:
            started = time.perf_counter()
            try:
                result = await run_in_threadpool(
                    manager.transcribe,
                    path,
                    language=language,
                    task=task,
                    initial_prompt=prompt,
                    temperature=temp_value,
                    want_words=want_words,
                )
            except ModelLoadError as exc:
                raise ApiError(500, f"Model is not available: {exc}", type="server_error") from exc
            except AudioDecodeError as exc:
                raise ApiError(
                    400,
                    f"Could not decode the uploaded audio: {exc}",
                    param="file",
                    code="invalid_audio",
                ) from exc
            except ApiError:
                raise
            except Exception as exc:  # noqa: BLE001 - never crash the server
                log.exception("transcription failed")
                raise ApiError(
                    500,
                    f"Transcription failed: {type(exc).__name__}: {exc}",
                    type="server_error",
                ) from exc
            elapsed = time.perf_counter() - started
            if settings.save_audio:
                await run_in_threadpool(
                    save_recording,
                    path,
                    file,
                    processing_seconds=elapsed,
                    result=result,
                )
        finally:
            try:
                os.unlink(path)
            except OSError:  # pragma: no cover - best effort cleanup
                log.warning("failed to remove temp file %s", path)

        return build_response(result, fmt)

    # -- routes ------------------------------------------------------------

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
    async def transcriptions(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[Optional[str], Form()] = None,
        language: Annotated[Optional[str], Form()] = None,
        prompt: Annotated[Optional[str], Form()] = None,
        response_format: Annotated[Optional[str], Form()] = None,
        temperature: Annotated[Optional[str], Form()] = None,
        timestamp_granularities: Annotated[list[str], Form(alias="timestamp_granularities[]")] = [],
        stream: Annotated[Optional[str], Form()] = None,
        include: Annotated[list[str], Form(alias="include[]")] = [],
    ) -> Response:
        if stream not in (None, "", "false", "0"):
            raise ApiError(
                400,
                "Streaming responses are not supported by this server.",
                param="stream",
                code="unsupported_value",
            )
        return await handle_audio_request(
            request,
            file,
            model,
            language,
            prompt,
            response_format,
            temperature,
            timestamp_granularities,
            task="transcribe",
        )

    @app.post("/v1/audio/translations", dependencies=[Depends(require_auth)])
    async def translations(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[Optional[str], Form()] = None,
        prompt: Annotated[Optional[str], Form()] = None,
        response_format: Annotated[Optional[str], Form()] = None,
        temperature: Annotated[Optional[str], Form()] = None,
    ) -> Response:
        return await handle_audio_request(
            request,
            file,
            model,
            None,
            prompt,
            response_format,
            temperature,
            [],
            task="translate",
        )

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=settings.model)])

    @app.get("/-/healthcheck/")
    @app.get("/-/healthcheck")
    async def healthcheck() -> dict:
        # Unauthenticated on purpose: the startup probe must succeed while the
        # model is still loading and even when WHISPER_API_KEY is set.
        # Both spellings are registered so no 307 redirect is involved.
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok" if manager.loaded else "loading",
            model=settings.model,
            model_loaded=manager.loaded,
            device=manager.device,
            compute_type=manager.compute_type,
            words_available=manager.words_enabled,
            notes=list(manager.load_notes),
        )

    @app.get("/api/recordings/{date}/{name}", dependencies=[Depends(require_auth)])
    async def recording(date: str, name: str) -> Response:
        directory = Path(settings.save_audio_dir)
        if not _is_safe_recording_part(date) or not _is_safe_recording_part(name):
            raise ApiError(404, "Recording not found.", type="not_found_error")
        path = (directory / date / name).resolve()
        if not path.is_file() or directory.resolve() not in path.parents:
            raise ApiError(404, "Recording not found.", type="not_found_error")
        return FileResponse(path, filename=name, content_disposition_type="inline")

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "whisper-local",
            "version": __version__,
            "model": settings.model,
            "model_loaded": manager.loaded,
            "endpoints": [
                "POST /v1/audio/transcriptions",
                "POST /v1/audio/translations",
                "GET /v1/models",
                "GET /healthz",
                "GET /ui",
            ],
            "ui": "/ui",
        }

    # -- UI (static) -------------------------------------------------------

    if STATIC_DIR.is_dir():
        index = STATIC_DIR / "index.html"

        @app.get("/ui", include_in_schema=False)
        async def ui_index() -> Response:
            # Registered before the mount so ``GET /ui`` is 200 instead of
            # Starlette's 307 redirect to ``/ui/``.
            return FileResponse(index, media_type="text/html; charset=utf-8")

        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    return app


def _is_safe_recording_part(part: str) -> bool:
    """Reject path traversal / nested segments in recording URLs."""
    return bool(part) and part not in {".", ".."} and "/" not in part and "\\" not in part
