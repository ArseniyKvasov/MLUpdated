from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
import shutil

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from groq import APIError
from dotenv import load_dotenv

from app.groq_service import GroqService
from app.schemas import (
    ErrorResponse,
    ChunkAnalyzeRequest,
    GradeOpenAnswersRequest,
    HealthResponse,
    LessonSummaryRequest,
    MiniSummaryRequest,
    PracticeSummaryRequest,
    QuizRequest,
    TaskAcceptedResponse,
    TaskQueueDebugResponse,
    TaskStatusResponse,
    TaskType,
    TeacherAnalysisAggregateRequest,
    TranscribeChunkJobRequest,
)
from app.task_queue import TaskQueueManager

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml_service")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, value)


def _task_queue_worker_counts() -> dict[str, int]:
    return {
        "transcribe": _int_env("TRANSCRIBE_WORKERS", 4),
        "analysis": _int_env("ANALYSIS_WORKERS", 4),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.groq_service = None
    app.state.task_queue = TaskQueueManager(
        executor=_execute_task,
        worker_counts=_task_queue_worker_counts(),
    )
    await app.state.task_queue.start()
    logger.info("ML Service starting up, task queue initialized.")
    yield
    await app.state.task_queue.stop()
    if app.state.groq_service:
        await app.state.groq_service.close()
    logger.info("ML Service shutting down, task queue stopped and GroqService closed.")


app = FastAPI(title="ML Service", version="1.0.0", lifespan=lifespan)


async def _get_groq_service() -> GroqService:
    """Возвращает GroqService, пересоздавая при необходимости."""
    service = app.state.groq_service
    if service is None:
        try:
            service = GroqService()
        except RuntimeError as exc:
            logger.error(f"Failed to create GroqService: {exc}")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app.state.groq_service = service
    return service


def get_groq_service() -> GroqService:
    """Синхронная обёртка для получения сервиса в синхронном контексте."""
    service = app.state.groq_service
    if service is None:
        try:
            service = GroqService()
        except RuntimeError as exc:
            logger.error(f"Failed to create GroqService: {exc}")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app.state.groq_service = service
    return service


def reset_groq_service() -> None:
    """Сбрасывает сервис, чтобы при следующем вызове создать новый экземпляр."""
    logger.info("Resetting GroqService instance.")
    app.state.groq_service = None


def get_task_queue() -> TaskQueueManager:
    task_queue = app.state.task_queue
    if task_queue is None:
        raise HTTPException(status_code=503, detail="Очередь задач не инициализирована.")
    return task_queue


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key or x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Некорректный API ключ.")


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error={"code": code, "message": message}).model_dump(),
    )


def _format_validation_error_location(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for part in loc:
        if part == "body":
            continue
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
            continue
        parts.append(str(part))
    return ".".join(parts) if parts else "request"


def _format_validation_error_message(exc: RequestValidationError) -> str:
    details: list[str] = []
    for error in exc.errors():
        loc = error.get("loc", ())
        if not isinstance(loc, tuple):
            loc = tuple(loc) if isinstance(loc, list) else ("request",)
        field = _format_validation_error_location(loc)
        message = error.get("msg", "Неверное значение.")
        details.append(f"{field}: {message}")
    if not details:
        return "Некорректный формат запроса."
    return "Некорректный формат запроса: " + "; ".join(details)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    service = app.state.groq_service
    groq_ok = False
    google_ok = False
    if service is not None:
        groq_ok, google_ok = await asyncio.gather(
            service.health_check(),
            service.google_health_check(),
        )
    return HealthResponse(groq_ok=groq_ok, google_ok=google_ok)


@app.get("/debug/queue", response_model=TaskQueueDebugResponse)
async def debug_queue(_: None = Depends(require_api_key)) -> Any:
    task_queue = get_task_queue()
    stats = await task_queue.get_queue_debug()
    return TaskQueueDebugResponse(
        transcribe=stats["transcribe"],
        analysis=stats["analysis"],
    )


async def _execute_task(task_type: TaskType, payload: dict[str, Any]) -> Any:
    logger.info(f"Executing task: {task_type}")
    service = get_groq_service()
    if task_type == "transcribe-chunk":
        logger.debug(f"Transcription payload: {payload}")
        return await service.transcribe_chunk(TranscribeChunkJobRequest.model_validate(payload))
    if task_type == "chunk-analyze":
        return await service.chunk_analyze(ChunkAnalyzeRequest.model_validate(payload))
    if task_type == "mini-summary":
        return await service.mini_summary(MiniSummaryRequest.model_validate(payload))
    if task_type == "teacher-analysis":
        return await service.teacher_analysis(ChunkAnalyzeRequest.model_validate(payload))
    if task_type == "teacher-analysis-aggregate":
        return await service.teacher_analysis_aggregate(TeacherAnalysisAggregateRequest.model_validate(payload))
    if task_type == "lesson-summary":
        return {"summary": await service.lesson_summary(LessonSummaryRequest.model_validate(payload))}
    if task_type == "practice-summary":
        return {"summary": await service.practice_summary(PracticeSummaryRequest.model_validate(payload))}
    if task_type == "quiz":
        return {"quiz": await service.quiz(QuizRequest.model_validate(payload))}
    if task_type == "grade-open-answers":
        return await service.grade_open_answers(GradeOpenAnswersRequest.model_validate(payload))
    raise RuntimeError(f"Неизвестный тип задачи: {task_type}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(f"RequestValidationError at {request.url}: {exc.errors()}")
    return error_response("invalid_request", _format_validation_error_message(exc), 422)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTPException {exc.status_code} at {request.url}: {exc.detail}")
    return error_response("invalid_request", exc.detail if isinstance(exc.detail, str) else "Ошибка запроса.", exc.status_code)


@app.exception_handler(APIError)
async def groq_exception_handler(_: Request, exc: APIError) -> JSONResponse:
    logger.error(f"Groq APIError: {exc}")
    reset_groq_service()
    return error_response("ml_provider_error", "Ошибка при обращении к Groq.", 502)


@app.exception_handler(RuntimeError)
async def runtime_exception_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    logger.error(f"RuntimeError: {exc}")
    return error_response("ml_provider_error", str(exc), 502)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled Exception: {exc}")
    return error_response("internal_error", "Внутренняя ошибка сервиса.", 500)


async def _submit_task(task_type: TaskType, payload: dict[str, Any]) -> TaskAcceptedResponse:
    task_queue = get_task_queue()
    try:
        job = await task_queue.submit(task_type, payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return job.to_task_accepted()


def _suffix_from_name(file_name: str, mime_type: str) -> str:
    if "." in file_name:
        return "." + file_name.rsplit(".", 1)[-1]
    if "/" in mime_type:
        return "." + mime_type.rsplit("/", 1)[-1]
    return ".bin"


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def _store_upload_file_sync(audio_file: UploadFile, suffix: str) -> Path:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        audio_file.file.seek(0)
        shutil.copyfileobj(audio_file.file, temp)
        temp.flush()
    finally:
        temp.close()
    return Path(temp.name)


async def _store_upload_file(audio_file: UploadFile) -> Path:
    suffix = _suffix_from_name(audio_file.filename or "chunk.bin", audio_file.content_type or "application/octet-stream")
    return await asyncio.to_thread(_store_upload_file_sync, audio_file, suffix)


@app.post("/transcribe-chunk", response_model=TaskAcceptedResponse, status_code=202)
async def transcribe_chunk(
    chunk_id: int = Form(...),
    start_ms: int = Form(...),
    end_ms: int = Form(...),
    audio_file: UploadFile = File(...),
    _: None = Depends(require_api_key),
) -> Any:
    logger.info(f"/transcribe-chunk called: chunk_id={chunk_id}, start_ms={start_ms}, end_ms={end_ms}, filename={audio_file.filename}")
    file_path = await _store_upload_file(audio_file)
    try:
        payload = {
            "file_path": str(file_path),
            "file_name": audio_file.filename or "chunk.bin",
            "mime_type": audio_file.content_type or "application/octet-stream",
            "chunk_id": chunk_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        return await _submit_task("transcribe-chunk", payload)
    except Exception:
        await asyncio.to_thread(_safe_unlink, file_path)
        raise
    finally:
        await audio_file.close()


@app.post("/chunk-analyze", response_model=TaskAcceptedResponse, status_code=202)
async def chunk_analyze(
    payload: ChunkAnalyzeRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("chunk-analyze", payload.model_dump())


@app.post("/teacher-analysis-aggregate", response_model=TaskAcceptedResponse, status_code=202)
async def teacher_analysis_aggregate(
    payload: TeacherAnalysisAggregateRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("teacher-analysis-aggregate", payload.model_dump())


@app.post("/lesson-summary", response_model=TaskAcceptedResponse, status_code=202)
async def lesson_summary(
    payload: LessonSummaryRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("lesson-summary", payload.model_dump())


@app.post("/quiz", response_model=TaskAcceptedResponse, status_code=202)
async def quiz(
    payload: QuizRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("quiz", payload.model_dump())


@app.post("/practice-summary", response_model=TaskAcceptedResponse, status_code=202)
async def practice_summary(
    payload: PracticeSummaryRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("practice-summary", payload.model_dump())


@app.post("/grade-open-answers", response_model=TaskAcceptedResponse, status_code=202)
async def grade_open_answers(
    payload: GradeOpenAnswersRequest,
    _: None = Depends(require_api_key),
) -> Any:
    return await _submit_task("grade-open-answers", payload.model_dump())


@app.get("/tasks/{job_id}", response_model=TaskStatusResponse)
async def task_status(job_id: str, _: None = Depends(require_api_key)) -> Any:
    task_queue = get_task_queue()
    job = await task_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return job.to_task_status()
