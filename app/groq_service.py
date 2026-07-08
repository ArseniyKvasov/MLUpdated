from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from groq import AsyncGroq
from google import genai
from google.genai import types

from app.prompts import (
    GRADE_PROMPT,
    CHUNK_ANALYZE_PROMPT,
    LESSON_SUMMARY_PROMPT,
    MINI_SUMMARY_PROMPT,
    PRACTICE_SUMMARY_PROMPT,
    QUIZ_PROMPT,
    TEACHER_ANALYSIS_AGGREGATE_PROMPT,
    TEACHER_AGGREGATE_PART_STRUCTURE_PROMPT,
    TEACHER_AGGREGATE_PART_ENGAGEMENT_PROMPT,
    TEACHER_AGGREGATE_PART_FLAGS_PROMPT,
)
from app.schemas import (
    GradeAnswerItem,
    GradeOpenAnswersRequest,
    ChunkAnalyzeRequest,
    TeacherAnalysisAggregateRequest,
    LessonSummaryRequest,
    MiniSummaryRequest,
    PracticeSummaryRequest,
    QuizRequest,
    TeacherSpeechExampleItem,
    TeacherSpeechQuestionItem,
    TeacherSpeechSpanItem,
    TranscriptItem,
    TranscribeChunkJobRequest,
)

from app.utils.json_utils import extract_json
from app.utils.normalization import (
    ensure_text,
    coerce_score,
    normalize_question_type,
    normalize_example_type,
)
from app.services.transcription import TranscriptionService, safe_unlink
from app.services.generation import GenerationService
from app.services.storage import get_chunk_storage
from dataclasses import dataclass

@dataclass(frozen=True)
class GenerationBackend:
    provider: str
    model: str


class GroqService:
    _ANALYSIS_TEMPERATURE = 0.0
    _ANALYSIS_TOP_P = 1.0
    _ANALYSIS_TOP_K = 1
    _TRANSCRIPTION_PROMPT = "Убедись, что язык транскрибации соответствует языку на записи"

    def __init__(self) -> None:
        self.transcriber_type = os.getenv("TRANSCRIBER_TYPE", "groq").lower()
        self.llm_type = os.getenv("LLM_TYPE", "gemini").lower()
        
        self.local_whisper_url = os.getenv("LOCAL_WHISPER_URL", "http://localhost:8080/v1")
        self.local_whisper_model = os.getenv("LOCAL_WHISPER_MODEL", "whisper-large-v3")
        self.local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.local_llm_model = os.getenv("LOCAL_LLM_MODEL", "gemma")
        
        # Cloudflare Worker AI proxy: when configured, Groq/Gemini requests are
        # routed through a Worker that holds the real provider keys as secrets.
        # This service only needs a shared secret to authenticate to the Worker.
        self._cf_worker_url = os.getenv("CLOUDFLARE_AI_WORKER_URL", "").strip().rstrip("/")
        self._cf_worker_secret = os.getenv("CLOUDFLARE_AI_WORKER_SECRET", "").strip()
        self._use_cf_worker = bool(self._cf_worker_url and self._cf_worker_secret)
        if bool(self._cf_worker_url) != bool(self._cf_worker_secret):
            raise RuntimeError(
                "CLOUDFLARE_AI_WORKER_URL и CLOUDFLARE_AI_WORKER_SECRET должны быть заданы вместе."
            )
        _worker_headers = {"X-Worker-Secret": self._cf_worker_secret} if self._use_cf_worker else None

        self._groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if self.transcriber_type == "groq" and not self._groq_api_key and not self._use_cf_worker:
            raise RuntimeError(
                "Не передан GROQ_API_KEY и не сконфигурирован Cloudflare Worker (CLOUDFLARE_AI_WORKER_URL/SECRET). "
                "Добавьте один из вариантов в .env."
            )

        self._groq_client = AsyncGroq(
            api_key=self._groq_api_key or "cloudflare-worker-proxy",
            base_url=f"{self._cf_worker_url}/groq" if self._use_cf_worker else None,
            default_headers=_worker_headers,
            max_retries=0,
        ) if (self._groq_api_key or self._use_cf_worker) else None

        self._google_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if self.llm_type == "gemini" and not self._google_api_key and not self._use_cf_worker:
            raise RuntimeError(
                "Не передан GEMINI_API_KEY и не сконфигурирован Cloudflare Worker (CLOUDFLARE_AI_WORKER_URL/SECRET). "
                "Добавьте один из вариантов в .env."
            )

        self._google_client = genai.Client(
            api_key=self._google_api_key or "cloudflare-worker-proxy",
            http_options=types.HttpOptions(
                base_url=f"{self._cf_worker_url}/google" if self._use_cf_worker else None,
                headers=_worker_headers,
                client_args={'http2': False, 'timeout': 600},
                async_client_args={'http2': False, 'timeout': 600}
            )
        ) if (self._google_api_key or self._use_cf_worker) else None
        self._concurrency_limit_sem = asyncio.Semaphore(10)
        
        # Instantiate split services
        self._transcription_service = TranscriptionService(
            transcriber_type=self.transcriber_type,
            local_whisper_url=self.local_whisper_url,
            local_whisper_model=self.local_whisper_model,
            groq_client=self._groq_client,
            transcription_prompt=self._TRANSCRIPTION_PROMPT,
        )
        
        self._generation_service = GenerationService(
            llm_type=self.llm_type,
            local_llm_url=self.local_llm_url,
            local_llm_model=self.local_llm_model,
            google_client=self._google_client,
            raw_google_models=os.getenv("GEMINI_GENERATION_MODELS"),
            timeout_seconds=600,
        )

        self.transcription_models = self._transcription_service.transcription_models
        self.google_generation_models = self._generation_service.google_generation_models

    @property
    def transcription_service(self) -> TranscriptionService:
        if not hasattr(self, '_transcription_service') or self._transcription_service is None:
            transcriber_type = getattr(self, 'transcriber_type', os.getenv("TRANSCRIBER_TYPE", "groq"))
            local_whisper_url = getattr(self, 'local_whisper_url', os.getenv("LOCAL_WHISPER_URL", "http://localhost:8080/v1"))
            local_whisper_model = getattr(self, 'local_whisper_model', os.getenv("LOCAL_WHISPER_MODEL", "whisper-large-v3"))
            groq_client = getattr(self, '_groq_client', None)
            self._transcription_service = TranscriptionService(
                transcriber_type=transcriber_type,
                local_whisper_url=local_whisper_url,
                local_whisper_model=local_whisper_model,
                groq_client=groq_client,
                transcription_prompt=self._TRANSCRIPTION_PROMPT,
            )
            if hasattr(self, 'transcription_models'):
                self._transcription_service.transcription_models = self.transcription_models
        return self._transcription_service

    @transcription_service.setter
    def transcription_service(self, val):
        self._transcription_service = val

    @property
    def generation_service(self) -> GenerationService:
        if not hasattr(self, '_generation_service') or self._generation_service is None:
            llm_type = getattr(self, 'llm_type', os.getenv("LLM_TYPE", "gemini"))
            local_llm_url = getattr(self, 'local_llm_url', os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"))
            local_llm_model = getattr(self, 'local_llm_model', os.getenv("LOCAL_LLM_MODEL", "gemma"))
            google_client = getattr(self, '_google_client', None)
            self._generation_service = GenerationService(
                llm_type=llm_type,
                local_llm_url=local_llm_url,
                local_llm_model=local_llm_model,
                google_client=google_client,
                raw_google_models=os.getenv("GEMINI_GENERATION_MODELS"),
                timeout_seconds=600,
            )
            if hasattr(self, 'google_generation_models'):
                self._generation_service.google_generation_models = self.google_generation_models
        return self._generation_service

    @generation_service.setter
    def generation_service(self, val):
        self._generation_service = val

    @property
    def _concurrency_limit(self):
        if not hasattr(self, '_concurrency_limit_sem') or self._concurrency_limit_sem is None:
            self._concurrency_limit_sem = asyncio.Semaphore(10)
        return self._concurrency_limit_sem

    @_concurrency_limit.setter
    def _concurrency_limit(self, val):
        self._concurrency_limit_sem = val

    async def _transcribe_with_retry(self, audio_path: Path) -> Any:
        self.transcription_service.transcriber_type = getattr(self, 'transcriber_type', 'groq')
        self.transcription_service.local_whisper_url = getattr(self, 'local_whisper_url', 'http://localhost:8080/v1')
        self.transcription_service.local_whisper_model = getattr(self, 'local_whisper_model', 'whisper-large-v3')
        self.transcription_service.groq_client = getattr(self, '_groq_client', None)
        if hasattr(self, 'transcription_models'):
            self.transcription_service.transcription_models = self.transcription_models
        return await self.transcription_service.transcribe_with_retry(audio_path)

    async def _chat_json(self, prompt: str, *, request_type: str = "unknown", is_retry: bool = False, allow_google: bool = True) -> Any:
        self.generation_service.llm_type = getattr(self, 'llm_type', 'gemini')
        self.generation_service.local_llm_url = getattr(self, 'local_llm_url', 'http://localhost:11434/v1')
        self.generation_service.local_llm_model = getattr(self, 'local_llm_model', 'gemma')
        self.generation_service.google_client = getattr(self, '_google_client', None)
        if hasattr(self, 'google_generation_models'):
            self.generation_service.google_generation_models = self.google_generation_models
        return await self.generation_service.chat_json(prompt, request_type=request_type, is_retry=is_retry)

    async def _chat_json_with_validation(self, prompt: str, validator: Callable[[Any], None], *, request_type: str = "unknown", max_retries: int = 3, allow_google: bool = True) -> Any:
        self.generation_service.llm_type = getattr(self, 'llm_type', 'gemini')
        self.generation_service.local_llm_url = getattr(self, 'local_llm_url', 'http://localhost:11434/v1')
        self.generation_service.local_llm_model = getattr(self, 'local_llm_model', 'gemma')
        self.generation_service.google_client = getattr(self, '_google_client', None)
        if hasattr(self, 'google_generation_models'):
            self.generation_service.google_generation_models = self.google_generation_models
        return await self.generation_service.chat_json_with_validation(prompt, validator, request_type=request_type, max_retries=max_retries)

    @classmethod
    def _analysis_groq_kwargs(cls) -> dict[str, Any]:
        return {
            "temperature": cls._ANALYSIS_TEMPERATURE,
            "top_p": cls._ANALYSIS_TOP_P,
        }

    @classmethod
    def _analysis_generation_config(
        cls,
        *,
        json_response: bool = False,
        max_output_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": cls._ANALYSIS_TEMPERATURE,
            "top_p": cls._ANALYSIS_TOP_P,
            "top_k": cls._ANALYSIS_TOP_K,
        }
        if json_response:
            config["response_mime_type"] = "application/json"
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        return config

    async def close(self) -> None:
        if self._google_client:
            try:
                await self._google_client.aio.aclose()
            except Exception as e:
                import logging
                logging.getLogger('ml_service').warning(f'Error closing Google client: {e}')

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        safe_unlink(path)

    @staticmethod
    async def _resolve_audio_path(payload: TranscribeChunkJobRequest) -> tuple[Path, Optional[str]]:
        """Returns a local path to the audio chunk plus the R2 key to clean up (if any).

        Chunks uploaded through R2 are downloaded once here so the rest of the
        transcription pipeline keeps working with a plain local file.
        """
        if payload.storage_key:
            storage = get_chunk_storage()
            if storage is None:
                raise RuntimeError(
                    "Задача ссылается на storage_key, но R2 не сконфигурирован (CLOUDFLARE_ACCOUNT_ID/R2_*)."
                )
            suffix = Path(payload.file_name or "chunk.bin").suffix or ".bin"
            audio_path = await asyncio.to_thread(storage.download_to_file, payload.storage_key, suffix)
            return audio_path, payload.storage_key
        if not payload.file_path:
            raise RuntimeError("В задаче транскрибации не указан ни file_path, ни storage_key.")
        return Path(payload.file_path), None

    async def health_check(self) -> bool:
        """Проверяет доступность транскрибатора."""
        if self.transcriber_type == "local":
            try:
                async with self._concurrency_limit:
                    async with httpx.AsyncClient(timeout=5) as client:
                        import httpx
                        resp = await client.get(self.local_whisper_url)
                        return resp.status_code < 500
            except Exception:
                return False
        else:
            if not self._groq_client:
                return False
            try:
                async with self._concurrency_limit:
                    response = await self._groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=1,
                        temperature=0,
                    )
                return response.choices[0].message.content is not None
            except Exception:
                return False

    async def google_health_check(self) -> bool:
        """Проверяет доступность LLM."""
        if self.llm_type == "local":
            try:
                async with self._concurrency_limit:
                    import httpx
                    base_url = self.local_llm_url
                    if "/v1" in base_url:
                        base_url = base_url.replace("/v1", "")
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(base_url)
                        return resp.status_code == 200 or "Ollama" in resp.text
            except Exception:
                return False
        else:
            if self._google_client is None:
                return False
            try:
                async with self._concurrency_limit:
                    from functools import partial
                    model = self.google_generation_models[0]
                    response = await self._google_client.aio.models.generate_content(
                        model=model,
                        contents="Hi",
                        config=self.generation_service._analysis_generation_config(
                            max_output_tokens=16,
                            disable_thinking=True,
                        ),
                    )
                return bool(getattr(response, "text", None))
            except Exception:
                return False

    async def transcribe_chunk(self, payload: TranscribeChunkJobRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            audio_path, storage_key = await self._resolve_audio_path(payload)
            try:
                transcription = await self._transcribe_with_retry(audio_path)
            finally:
                await asyncio.to_thread(self._safe_unlink, audio_path)
                if storage_key is not None:
                    storage = get_chunk_storage()
                    if storage is not None:
                        await asyncio.to_thread(storage.delete, storage_key)

        transcript: list[dict[str, Any]] = []

        def parse_segment(seg: Any) -> Optional[dict[str, Any]]:
            if isinstance(seg, dict):
                start_sec = seg.get("start") or 0
                text_val = (seg.get("text") or "").strip()
                if not text_val:
                    return None
                item = {"start_ms": int(float(start_sec) * 1000), "text": text_val}
                if "speaker" in seg and seg["speaker"] is not None:
                    item["speaker"] = str(seg["speaker"]).strip()
                return item
            else:
                start_sec = getattr(seg, "start", 0) or 0
                text_val = (getattr(seg, "text", "") or "").strip()
                if not text_val:
                    return None
                item = {"start_ms": int(float(start_sec) * 1000), "text": text_val}
                if hasattr(seg, "speaker") and getattr(seg, "speaker") is not None:
                    item["speaker"] = str(getattr(seg, "speaker")).strip()
                elif hasattr(seg, "get") and seg.get("speaker") is not None:
                    item["speaker"] = str(seg.get("speaker")).strip()
                return item

        if isinstance(transcription, dict):
            segments = transcription.get("segments") or []
            for segment in segments:
                item_dict = parse_segment(segment)
                if item_dict:
                    transcript.append(item_dict)
            if not transcript:
                text = (transcription.get("text") or "").strip()
                if text:
                    item_dict = {"start_ms": payload.start_ms, "text": text}
                    if "speaker" in transcription and transcription["speaker"] is not None:
                        item_dict["speaker"] = str(transcription["speaker"]).strip()
                    transcript.append(item_dict)
                else:
                    raise RuntimeError(
                        "Пустая транскрибация: в аудио не распознано речи или файл пришел в неподдерживаемом формате."
                    )
        else:
            segments = getattr(transcription, "segments", None) or []
            for segment in segments:
                item_dict = parse_segment(segment)
                if item_dict:
                    transcript.append(item_dict)
            if not transcript:
                text = (getattr(transcription, "text", "") or "").strip()
                if text:
                    item_dict = {"start_ms": payload.start_ms, "text": text}
                    if hasattr(transcription, "speaker") and getattr(transcription, "speaker") is not None:
                        item_dict["speaker"] = str(getattr(transcription, "speaker")).strip()
                    elif isinstance(transcription, dict) and "speaker" in transcription and transcription["speaker"] is not None:
                        item_dict["speaker"] = str(transcription["speaker"]).strip()
                    transcript.append(item_dict)
                else:
                    raise RuntimeError(
                        "Пустая транскрибация: в аудио не распознано речи или файл пришел в неподдерживаемом формате."
                    )

        return {
            "chunk_id": payload.chunk_id,
            "start_ms": payload.start_ms,
            "end_ms": payload.end_ms,
            "transcript": transcript,
        }

    async def mini_summary(self, payload: MiniSummaryRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            text = self._transcript_text(payload.transcript)
            prompt = (
                f"{MINI_SUMMARY_PROMPT}\n"
                f"Чанк: {payload.chunk_id}\n"
                f"Время: {payload.start_time} - {payload.end_time}\n"
                f"Транскрипт:\n{text}\n"
                "Верни JSON с ключом key_points."
            )
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_mini_summary,
                max_retries=3,
                request_type="mini-summary"
            )
            key_points_source = data.get("key_points") if isinstance(data, dict) else data
            return {
                "chunk_id": payload.chunk_id,
                "start_time": payload.start_time,
                "end_time": payload.end_time,
                "key_points": self._ensure_list(key_points_source, ["Основная идея"]),
            }

    @staticmethod
    def _build_chunk_analysis_prompt(payload: ChunkAnalyzeRequest) -> str:
        transcript_payload = [item.model_dump() for item in payload.transcript]
        return (
            f"{CHUNK_ANALYZE_PROMPT}\n\n"
            f"Чанк: {payload.chunk_id}\n"
            f"Время: {payload.start_time} - {payload.end_time}\n"
            f"Транскрипт:\n{json.dumps(transcript_payload, ensure_ascii=False)}\n"
        )

    def _normalize_chunk_analysis_output(self, payload: ChunkAnalyzeRequest, data: Any) -> dict[str, Any]:
        data_dict = data if isinstance(data, dict) else {}
        key_points = self._ensure_list(data_dict.get("key_points"), [])
        teacher_questions = self._extract_teacher_question_items(data_dict.get("teacher_questions"))
        student_answers = self._extract_span_items(data_dict.get("student_answers"))
        examples_and_analogies = self._extract_example_span_items(data_dict.get("examples_and_analogies"))
        lesson_events = self._extract_lesson_events(data_dict.get("lesson_events"))
        goals_and_summary = self._extract_goals_and_summary(data_dict.get("goals_and_summary"))
        flags = self._extract_flags(data_dict.get("flags"))

        return {
            "chunk_id": payload.chunk_id,
            "start_time": payload.start_time,
            "end_time": payload.end_time,
            "key_points": key_points,
            "teacher_questions": teacher_questions,
            "student_answers": student_answers,
            "examples_and_analogies": examples_and_analogies,
            "lesson_events": lesson_events,
            "goals_and_summary": goals_and_summary,
            "flags": flags,
        }

    async def chunk_analyze(self, payload: ChunkAnalyzeRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            prompt = self._build_chunk_analysis_prompt(payload)
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_chunk_analysis,
                max_retries=3,
                request_type="chunk-analyze",
            )
            return self._normalize_chunk_analysis_output(payload, data)

    async def teacher_analysis(self, payload: ChunkAnalyzeRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            prompt = self._build_chunk_analysis_prompt(payload)
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_teacher_analysis,
                max_retries=3,
                request_type="teacher-analysis",
            )
            return self._normalize_chunk_analysis_output(payload, data)

    async def teacher_analysis_aggregate(self, payload: TeacherAnalysisAggregateRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            # Split aggregation into sectors to handle long recordings and prevent context overflow
            # We increase the chunk limit since each sector has less work to do.
            analyses_payload = [item.model_dump() for item in payload.chunk_analyses[:30]]
            chunks_json = json.dumps({"chunk_analyses": analyses_payload}, ensure_ascii=False)
            
            async def call_sector(prompt_base, validator, req_type):
                prompt = f"{prompt_base}\n\njson {chunks_json}\n"
                return await self._chat_json_with_validation(
                    prompt,
                    validator=validator,
                    max_retries=3,
                    request_type=req_type
                )
            
            tasks = [
                call_sector(TEACHER_AGGREGATE_PART_STRUCTURE_PROMPT, self._validate_aggregate_structure, "teacher-aggregate-structure"),
                call_sector(TEACHER_AGGREGATE_PART_ENGAGEMENT_PROMPT, self._validate_aggregate_engagement, "teacher-aggregate-engagement"),
                call_sector(TEACHER_AGGREGATE_PART_FLAGS_PROMPT, self._validate_aggregate_flags, "teacher-aggregate-flags"),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data_dict = {}
            for res in results:
                if isinstance(res, Exception):
                    import logging
                    logging.getLogger("ml_service").error(f"Sectoral aggregation task failed: {res}")
                    continue
                if isinstance(res, dict):
                    data_dict.update(res)

            lesson_format = {
                "format": ensure_text(
                    data_dict.get("lesson_format", {}).get("format") if isinstance(data_dict.get("lesson_format"), dict) else None,
                    "лекция",
                ),
                "comment": ensure_text(
                    data_dict.get("lesson_format", {}).get("comment") if isinstance(data_dict.get("lesson_format"), dict) else None,
                    "На основе чанков урок носит объяснительный характер и опирается на вопросы, примеры и структурированный разбор.",
                ),
            }

            audience_engagement = {
                "questions_to_students": self._normalize_question_fragments_group(
                    data_dict.get("audience_engagement", {}).get("questions_to_students")
                    if isinstance(data_dict.get("audience_engagement"), dict)
                    else None,
                    fallback_title="Преподаватель активно задаёт вопросы студентам",
                    fallback_comment="Преобладают проверочные, уточняющие и риторические вопросы.",
                ),
                "student_answers": self._normalize_title_fragments_group(
                    data_dict.get("audience_engagement", {}).get("student_answers")
                    if isinstance(data_dict.get("audience_engagement"), dict)
                    else None,
                    fallback_title="Преподаватель реагирует на ответы студентов и развивает обсуждение",
                    fallback_comment="Преподаватель поддерживает ответы студентов и развивает их мысль.",
                ),
            }

            lesson_structure = {
                "step_by_step_explanation": {
                    "timeline": self._extract_timeline_items(
                        data_dict.get("lesson_structure", {}).get("step_by_step_explanation", {}).get("timeline")
                        if isinstance(data_dict.get("lesson_structure"), dict)
                        and isinstance(data_dict.get("lesson_structure", {}).get("step_by_step_explanation"), dict)
                        else None
                    )
                },
                "goals_and_summary": self._extract_goals_and_summary(
                    data_dict.get("lesson_structure", {}).get("goals_and_summary")
                    if isinstance(data_dict.get("lesson_structure"), dict)
                    else None
                ),
            }

            material_explanation = {
                "examples_and_analogies": self._normalize_example_fragments_group(
                    data_dict.get("material_explanation", {}).get("examples_and_analogies")
                    if isinstance(data_dict.get("material_explanation"), dict)
                    else None,
                    fallback_title="Преподаватель активно использует примеры и аналогии",
                    fallback_comment="В анализ попадают только реальные примеры, аналогии и истории, оформленные как связный пересказ.",
                )
            }

            teacher_recommendation = self._normalize_recommendation_block(
                data_dict.get("teacher_recommendation"),
                fallback_title="Рекомендация преподавателю",
                fallback_comment="Продолжайте опираться на конкретные примеры и завершать блоки короткой проверкой понимания.",
            )

            flags = {
                "profanity": self._normalize_flag_group(
                    data_dict.get("flags", {}).get("profanity") if isinstance(data_dict.get("flags"), dict) else None,
                    fallback_present=False,
                ),
                "overly_familiar_tone": self._normalize_flag_group(
                    data_dict.get("flags", {}).get("overly_familiar_tone") if isinstance(data_dict.get("flags"), dict) else None,
                    fallback_present=False,
                ),
            }

            return {
                "lesson_format": lesson_format,
                "audience_engagement": audience_engagement,
                "lesson_structure": lesson_structure,
                "material_explanation": material_explanation,
                "teacher_recommendation": teacher_recommendation,
                "flags": flags,
            }

    async def lesson_summary(self, payload: LessonSummaryRequest) -> list[dict[str, Any]]:
        async with self._concurrency_limit:
            prompt = (
                f"{LESSON_SUMMARY_PROMPT}\n"
                f"Тема: {payload.topic_hint or ''}\n"
                f"Ключевые пункты: {json.dumps(payload.key_points, ensure_ascii=False)}\n"
            )
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_lesson_summary,
                max_retries=3,
                request_type="lesson-summary"
            )
            data_dict = data if isinstance(data, dict) else {}
            sections = data if isinstance(data, list) else data_dict.get("summary", [])
            return [
                {
                    "subtopic": ensure_text(section.get("subtopic"), "Тема"),
                    "content": ensure_text(section.get("content"), "Краткий итоговый конспект по ключевым пунктам."),
                }
                for section in sections
                if isinstance(section, dict)
            ]

    async def quiz(self, payload: QuizRequest) -> list[dict[str, Any]]:
        async with self._concurrency_limit:
            prompt = (
                f"{QUIZ_PROMPT}\n"
                f"Конспект: {json.dumps([item.model_dump() for item in payload.summary], ensure_ascii=False)}\n"
                "Верни JSON-массив вопросов с полями question_id, question_text, question_type, options, correct_answer, explanation, subtopic."
            )
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_quiz,
                max_retries=3,
                request_type="quiz"
            )
            data_dict = data if isinstance(data, dict) else {}
            quiz_items = data if isinstance(data, list) else data_dict.get("quiz", [])
            if not isinstance(quiz_items, list) or not quiz_items:
                raise RuntimeError(
                    "Не удалось сгенерировать корректный quiz: отсутствуют вопросы."
                )
            return [
                {
                    "question_id": int(item.get("question_id", index + 1)),
                    "question_text": ensure_text(item.get("question_text"), "Вопрос"),
                    "question_type": item.get("question_type", "multiple_choice"),
                    "options": item.get("options"),
                    "correct_answer": item.get("correct_answer", 0),
                    "explanation": ensure_text(item.get("explanation"), "Пояснение."),
                    "subtopic": ensure_text(item.get("subtopic"), "Тема"),
                }
                for index, item in enumerate(quiz_items)
                if isinstance(item, dict)
            ]

    async def practice_summary(self, payload: PracticeSummaryRequest) -> list[dict[str, Any]]:
        async with self._concurrency_limit:
            topics_payload = []
            for topic in payload.topics:
                topics_payload.append(
                    {
                        "subtopic": topic.subtopic,
                        "summary_section": topic.summary_section.model_dump() if topic.summary_section else None,
                        "mini_summaries": [
                            {
                                "chunk_id": item.chunk_id,
                                "start_time": item.start_time,
                                "end_time": item.end_time,
                                "key_points": item.key_points,
                                "terms": item.terms,
                                "examples": item.examples,
                            }
                            for item in topic.mini_summaries
                        ],
                    }
                )

            questions_payload = [item.model_dump() for item in payload.questions]
            prompt = (
                f"{PRACTICE_SUMMARY_PROMPT}\n"
                f"weak_subtopics: {json.dumps(payload.weak_subtopics, ensure_ascii=False)}\n"
                f"topics: {json.dumps(topics_payload, ensure_ascii=False)}\n"
                f"questions: {json.dumps(questions_payload, ensure_ascii=False)}\n"
            )
            data = await self._chat_json_with_validation(
                prompt,
                validator=self._validate_practice_summary,
                max_retries=3,
                request_type="practice-summary"
            )
            data_dict = data if isinstance(data, dict) else {}
            sections = data if isinstance(data, list) else data_dict.get("summary", [])
            return [
                {
                    "subtopic": ensure_text(section.get("subtopic"), "Подтема"),
                    "content": ensure_text(
                        section.get("content"),
                        "Краткий практический конспект с акцентом на ошибки и разбор похожих примеров.",
                    ),
                }
                for section in sections
                if isinstance(section, dict)
            ]

    async def grade_open_answers(self, payload: GradeOpenAnswersRequest) -> dict[str, Any]:
        async with self._concurrency_limit:
            answers_payload = [
                {
                    "question_id": item.question_id,
                    "question_text": item.question_text,
                    "correct_answer": item.correct_answer,
                    "student_answer": item.student_answer,
                }
                for item in payload.answers
            ]
            prompt = (
                f"{GRADE_PROMPT}\n"
                f"Ответы: {json.dumps(answers_payload, ensure_ascii=False)}\n"
                'Верни JSON строго в формате {"scores":[{"question_id": int, "score": int}, ...]}. '
                "score должен быть 0 или 1."
            )
            data = await self._chat_json_with_validation(
                prompt,
                validator=lambda d: self._validate_grade_open_answers(d, payload.answers),
                max_retries=3,
                request_type="grade-open-answers"
            )
            raw_scores = data.get("scores", []) if isinstance(data, dict) else []
            normalized_scores = self._normalize_score_pairs(raw_scores, payload.answers)
            if not normalized_scores:
                raise RuntimeError(
                    "Не удалось получить корректные оценки ответов."
                )

            return {
                "scores": [{"question_id": q_id, "score": sc} for q_id, sc in normalized_scores],
            }

    @staticmethod
    def _transcript_text(transcript: list[TranscriptItem]) -> str:
        return "\n".join(f"{item.start_ms}: {item.text}" for item in transcript)

    @staticmethod
    def _ensure_text(value: Any, fallback: str) -> str:
        return ensure_text(value, fallback)

    @staticmethod
    def _ensure_list(value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return items
        return fallback

    @staticmethod
    def _coerce_score(value: Any) -> int:
        return coerce_score(value)

    @staticmethod
    def _normalize_score_pairs(
        raw_scores: Any,
        answers: list[GradeAnswerItem],
    ) -> list[tuple[int, int]]:
        if not isinstance(raw_scores, list):
            return []
        normalized: list[tuple[int, int]] = []
        for index, item in enumerate(raw_scores):
            fallback_question_id = answers[index].question_id if index < len(answers) else 0
            question_id = fallback_question_id
            score = 0
            if isinstance(item, dict):
                question_id = item.get("question_id", fallback_question_id)
                score = item.get("score", 0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                question_id = item[0]
                score = item[1]
            else:
                score = item

            try:
                normalized.append((int(question_id), coerce_score(score)))
            except (TypeError, ValueError):
                normalized.append((fallback_question_id, 0))

        return normalized

    @staticmethod
    def _validate_mini_summary(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON с ключом key_points.")
        key_points = data.get("key_points")
        if not isinstance(key_points, list) or not key_points:
            raise ValueError("key_points должен быть непустым списком.")

    @staticmethod
    def _validate_chunk_analysis(data: Any) -> None:
        GroqService._validate_teacher_analysis(data)
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON с результатом анализа чанка.")
        key_points = data.get("key_points")
        if not isinstance(key_points, list) or not key_points:
            raise ValueError("key_points должен быть непустым списком.")

    @staticmethod
    def _validate_teacher_analysis(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON с результатом анализа речи.")

        required_top_level = (
            "chunk_id",
            "start_time",
            "end_time",
            "teacher_questions",
            "student_answers",
            "examples_and_analogies",
            "lesson_events",
            "goals_and_summary",
            "flags",
        )
        for field in required_top_level:
            if field not in data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        if not isinstance(data.get("chunk_id"), int):
            raise ValueError("chunk_id должен быть числом.")
        if not isinstance(data.get("start_time"), str) or not data.get("start_time"):
            raise ValueError("start_time должен быть непустой строкой.")
        if not isinstance(data.get("end_time"), str) or not data.get("end_time"):
            raise ValueError("end_time должен быть непустой строкой.")

        for field in ("teacher_questions", "student_answers", "examples_and_analogies", "lesson_events"):
            if not isinstance(data.get(field), list):
                raise ValueError(f"Поле {field} должно быть списком.")

        GroqService._validate_teacher_question_items(data.get("teacher_questions"), "teacher_questions")
        GroqService._validate_teacher_span_items(data.get("student_answers"), "student_answers")
        GroqService._validate_teacher_example_items(data.get("examples_and_analogies"), "examples_and_analogies")
        GroqService._validate_lesson_events(data.get("lesson_events"))

        GroqService._validate_goals_and_summary(data.get("goals_and_summary"))

        flags = data.get("flags")
        if not isinstance(flags, dict):
            raise ValueError("flags должен быть объектом.")
        for field in ("profanity", "overly_familiar_tone"):
            if not isinstance(flags.get(field), list):
                raise ValueError(f"flags.{field} должен быть списком.")
        GroqService._validate_teacher_flags(flags)

    @staticmethod
    def _extract_json(content: str) -> Any:
        return extract_json(content)


    @staticmethod
    def _validate_aggregate_structure(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON (structure).")
        for field in ("lesson_format", "lesson_structure", "teacher_recommendation"):
            if field not in data:
                raise ValueError(f"Отсутствует поле: {field}")

    @staticmethod
    def _validate_aggregate_engagement(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON (engagement).")
        for field in ("audience_engagement", "material_explanation"):
            if field not in data:
                raise ValueError(f"Отсутствует поле: {field}")

    @staticmethod
    def _validate_aggregate_flags(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON (flags).")
        if "flags" not in data:
            raise ValueError("Отсутствует поле: flags")

    @staticmethod
    def _validate_teacher_analysis_aggregate(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON с итоговым анализом чанков.")

        for field in ("lesson_format", "audience_engagement", "lesson_structure", "material_explanation", "flags"):
            if field not in data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        lesson_format = data.get("lesson_format")
        if not isinstance(lesson_format, dict):
            raise ValueError("lesson_format должен быть объектом.")
        if not isinstance(lesson_format.get("format"), str) or not lesson_format.get("format").strip():
            raise ValueError("lesson_format.format должен быть непустой строкой.")
        if not isinstance(lesson_format.get("comment"), str) or not lesson_format.get("comment").strip():
            raise ValueError("lesson_format.comment должен быть непустой строкой.")

        audience_engagement = data.get("audience_engagement")
        if not isinstance(audience_engagement, dict):
            raise ValueError("audience_engagement должен быть объектом.")
        question_group = audience_engagement.get("questions_to_students")
        if not isinstance(question_group, dict):
            raise ValueError("audience_engagement.questions_to_students должен быть объектом.")
        if "comment" in question_group and not isinstance(question_group.get("comment"), str):
            raise ValueError("audience_engagement.questions_to_students.comment должен быть строкой.")
        GroqService._validate_teacher_question_items(
            question_group.get("fragments"),
            "audience_engagement.questions_to_students.fragments",
        )

        for field in ("student_answers",):
            group = audience_engagement.get(field)
            if not isinstance(group, dict):
                raise ValueError(f"audience_engagement.{field} должен быть объектом.")
            if "comment" in group and not isinstance(group.get("comment"), str):
                raise ValueError(f"audience_engagement.{field}.comment должен быть строкой.")
            GroqService._validate_teacher_span_items(group.get("fragments"), f"audience_engagement.{field}.fragments")

        lesson_structure = data.get("lesson_structure")
        if not isinstance(lesson_structure, dict):
            raise ValueError("lesson_structure должен быть объектом.")
        step_by_step = lesson_structure.get("step_by_step_explanation")
        if not isinstance(step_by_step, dict):
            raise ValueError("lesson_structure.step_by_step_explanation должен быть объектом.")
        GroqService._validate_timeline_items(step_by_step.get("timeline"))
        GroqService._validate_goals_and_summary(lesson_structure.get("goals_and_summary"))

        material_explanation = data.get("material_explanation")
        if not isinstance(material_explanation, dict):
            raise ValueError("material_explanation должен быть объектом.")
        examples = material_explanation.get("examples_and_analogies")
        if not isinstance(examples, dict):
            raise ValueError("material_explanation.examples_and_analogies должен быть объектом.")
        if "comment" in examples and not isinstance(examples.get("comment"), str):
            raise ValueError("material_explanation.examples_and_analogies.comment должен быть строкой.")
        GroqService._validate_teacher_example_items(
            examples.get("fragments"),
            "material_explanation.examples_and_analogies.fragments",
        )

        flags = data.get("flags")
        if not isinstance(flags, dict):
            raise ValueError("flags должен быть объектом.")
        for field in ("profanity", "overly_familiar_tone"):
            group = flags.get(field)
            if not isinstance(group, dict):
                raise ValueError(f"flags.{field} должен быть объектом.")
            if not isinstance(group.get("present"), bool):
                raise ValueError(f"flags.{field}.present должен быть логическим значением.")
            GroqService._validate_teacher_span_items(group.get("fragments"), f"flags.{field}.fragments")

    @staticmethod
    def _validate_teacher_span_items(items: Any, field_name: str) -> None:
        if not isinstance(items, list):
            raise ValueError(f"Поле {field_name} должно быть списком.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Элементы {field_name} должны быть объектами.")
            if not isinstance(item.get("start_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать start_ms как число.")
            if not isinstance(item.get("end_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать end_ms как число.")
            if not isinstance(item.get("text"), str) or not item.get("text").strip():
                raise ValueError(f"Элемент {field_name} должен содержать непустой text.")

    @staticmethod
    def _validate_teacher_question_items(items: Any, field_name: str) -> None:
        if not isinstance(items, list):
            raise ValueError(f"Поле {field_name} должно быть списком.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Элементы {field_name} должны быть объектами.")
            if not isinstance(item.get("start_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать start_ms как число.")
            if not isinstance(item.get("end_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать end_ms как число.")
            if not isinstance(item.get("text"), str) or not item.get("text").strip():
                raise ValueError(f"Элемент {field_name} должен содержать непустой text.")
            if "question_type" not in item:
                raise ValueError(f"Элемент {field_name} должен содержать question_type.")
            if not isinstance(item.get("question_type"), str):
                raise ValueError(f"Элемент {field_name} должен содержать question_type как строку.")

    @staticmethod
    def _validate_teacher_example_items(items: Any, field_name: str) -> None:
        if not isinstance(items, list):
            raise ValueError(f"Поле {field_name} должно быть списком.")
        allowed_types = {"example", "analogy", "metaphor", "storytelling"}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Элементы {field_name} должны быть объектами.")
            if not isinstance(item.get("start_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать start_ms как число.")
            if not isinstance(item.get("end_ms"), int):
                raise ValueError(f"Элемент {field_name} должен содержать end_ms как число.")
            if not isinstance(item.get("text"), str) or not item.get("text").strip():
                raise ValueError(f"Элемент {field_name} должен содержать непустой text.")
            item_type = normalize_example_type(item.get("type"))
            if item_type not in allowed_types:
                raise ValueError(
                    f"Элемент {field_name} должен содержать type со значением example, analogy, metaphor или storytelling."
                )

    @staticmethod
    def _validate_lesson_events(items: Any) -> None:
        if not isinstance(items, list):
            raise ValueError("Поле lesson_events должно быть списком.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Элементы lesson_events должны быть объектами.")
            if not isinstance(item.get("start_ms"), int):
                raise ValueError("Элемент lesson_events должен содержать start_ms как число.")
            if not isinstance(item.get("title"), str) or not item.get("title").strip():
                raise ValueError("Элемент lesson_events должен содержать непустой title.")
            if not isinstance(item.get("description"), str) or not item.get("description").strip():
                raise ValueError("Элемент lesson_events должен содержать непустой description.")

    @staticmethod
    def _validate_timeline_items(items: Any) -> None:
        if not isinstance(items, list):
            raise ValueError("timeline должен быть списком.")
        if len(items) > 10:
            raise ValueError("timeline должен содержать не более 10 элементов.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Элементы timeline должны быть объектами.")
            if not isinstance(item.get("start_ms"), int):
                raise ValueError("Элемент timeline должен содержать start_ms как число.")
            if not isinstance(item.get("title"), str) or not item.get("title").strip():
                raise ValueError("Элемент timeline должен содержать непустой title.")
            if not isinstance(item.get("description"), str) or not item.get("description").strip():
                raise ValueError("Элемент timeline должен содержать непустой description.")

    @staticmethod
    def _validate_goals_and_summary(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("goals_and_summary должен быть объектом.")
        for goal_name in ("intro", "summary"):
            goal = value.get(goal_name)
            if not isinstance(goal, dict):
                raise ValueError(f"{goal_name} должен быть объектом.")
            if "present" not in goal or not isinstance(goal.get("present"), bool):
                raise ValueError(f"{goal_name}.present должен быть логическим значением.")
            if "start_ms" not in goal:
                raise ValueError(f"{goal_name}.start_ms обязателен.")
            if goal.get("present"):
                if not isinstance(goal.get("start_ms"), int):
                    raise ValueError(f"{goal_name}.start_ms должен быть числом, если present=true.")
            elif goal.get("start_ms") is not None and not isinstance(goal.get("start_ms"), int):
                raise ValueError(f"{goal_name}.start_ms должен быть null или числом.")
            if not isinstance(goal.get("comment"), str):
                raise ValueError(f"{goal_name}.comment должен быть строкой.")

    @staticmethod
    def _validate_teacher_flags(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("flags должен быть объектом.")
        for field in ("profanity", "overly_familiar_tone"):
            items = value.get(field)
            if not isinstance(items, list):
                raise ValueError(f"flags.{field} должен быть списком.")
            GroqService._validate_teacher_span_items(items, f"flags.{field}")

    @staticmethod
    def _validate_lesson_summary(data: Any) -> None:
        sections = data if isinstance(data, list) else data.get("summary") if isinstance(data, dict) else None
        if not isinstance(sections, list) or not sections:
            raise ValueError("Ожидался непустой массив summary.")
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("Каждый элемент summary должен быть объектом.")
            if not isinstance(section.get("subtopic"), str) or not section.get("subtopic"):
                raise ValueError("Каждый элемент summary должен содержать непустой subtopic.")
            if not isinstance(section.get("content"), str) or not section.get("content"):
                raise ValueError("Каждый элемент summary должен содержать непустой content.")

    @staticmethod
    def _validate_quiz(data: Any) -> None:
        quiz_items = data if isinstance(data, list) else data.get("quiz") if isinstance(data, dict) else None
        if not isinstance(quiz_items, list) or not quiz_items:
            raise ValueError("Ожидался непустой массив quiz.")
        for item in quiz_items:
            if not isinstance(item, dict):
                raise ValueError("Каждый вопрос должен быть объектом.")
            for field in ("question_id", "question_text", "question_type", "options", "correct_answer", "explanation", "subtopic"):
                if field not in item:
                    raise ValueError(f"Отсутствует обязательное поле: {field}")
            if item.get("question_type") == "multiple_choice" and not isinstance(item.get("options"), list):
                raise ValueError("multiple_choice вопрос должен содержать options (список).")

    @staticmethod
    def _validate_practice_summary(data: Any) -> None:
        sections = data if isinstance(data, list) else data.get("summary") if isinstance(data, dict) else None
        if not isinstance(sections, list) or not sections:
            raise ValueError("Ожидался непустой массив summary.")
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("Каждый элемент summary должен быть объектом.")
            if not isinstance(section.get("subtopic"), str) or not section.get("subtopic"):
                raise ValueError("Каждый элемент summary должен содержать непустой subtopic.")
            if not isinstance(section.get("content"), str) or not section.get("content"):
                raise ValueError("Каждый элемент summary должен содержать непустой content.")

    @staticmethod
    def _extract_span_items(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                span = TeacherSpeechSpanItem(
                    start_ms=int(item.get("start_ms")),
                    end_ms=int(item.get("end_ms")),
                    text=ensure_text(item.get("text"), ""),
                ).model_dump()
                key = (span["start_ms"], span["end_ms"], span["text"])
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(span)
            except (TypeError, ValueError):
                continue
            if len(normalized) >= 10:
                break
        return normalized

    @staticmethod
    def _extract_teacher_question_items(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                question_type = normalize_question_type(item.get("question_type"))
                question_item = TeacherSpeechQuestionItem(
                    start_ms=int(item.get("start_ms")),
                    end_ms=int(item.get("end_ms")),
                    text=ensure_text(item.get("text"), ""),
                    question_type=question_type,
                ).model_dump()
                key = (
                    question_item["start_ms"],
                    question_item["end_ms"],
                    question_item["text"],
                    question_item["question_type"],
                )
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(question_item)
            except (TypeError, ValueError):
                continue
            if len(normalized) >= 10:
                break
        return normalized

    @staticmethod
    def _extract_example_span_items(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        allowed_types = {"example", "analogy", "metaphor", "storytelling"}
        seen: set[tuple[int, int, str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                example_item = TeacherSpeechExampleItem(
                    start_ms=int(item.get("start_ms")),
                    end_ms=int(item.get("end_ms")),
                    text=ensure_text(item.get("text"), ""),
                    type=normalize_example_type(item.get("type")),
                ).model_dump()
                if example_item["type"] not in allowed_types:
                    continue
                key = (example_item["start_ms"], example_item["end_ms"], example_item["text"], example_item["type"])
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(example_item)
            except (TypeError, ValueError):
                continue
            if len(normalized) >= 10:
                break
        return normalized

    @staticmethod
    def _extract_lesson_events(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                normalized.append(
                    {
                        "start_ms": int(item.get("start_ms")),
                        "title": ensure_text(item.get("title"), ""),
                        "description": ensure_text(item.get("description"), ""),
                    }
                )
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _extract_timeline_items(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[int, str, str]] = set()
        for item in sorted(
            (entry for entry in items if isinstance(entry, dict)),
            key=lambda entry: int(entry.get("start_ms", 0)) if isinstance(entry.get("start_ms"), int) else 0,
        ):
            try:
                timeline_item = {
                    "start_ms": int(item.get("start_ms")),
                    "title": ensure_text(item.get("title"), ""),
                    "description": ensure_text(item.get("description"), ""),
                }
                key = (timeline_item["start_ms"], timeline_item["title"], timeline_item["description"])
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(timeline_item)
            except (TypeError, ValueError):
                continue
            if len(normalized) >= 10:
                break
        return normalized

    @staticmethod
    def _normalize_title_fragments_group(
        value: Any,
        *,
        fallback_title: str,
        fallback_comment: str = "",
    ) -> dict[str, Any]:
        title = fallback_title
        comment = fallback_comment
        fragments = []
        if isinstance(value, dict):
            title = ensure_text(value.get("title"), fallback_title)
            comment = ensure_text(value.get("comment"), fallback_comment)
            fragments = GroqService._extract_span_items(value.get("fragments"))
        return {"title": title, "comment": comment, "fragments": fragments}

    @staticmethod
    def _normalize_question_fragments_group(
        value: Any,
        *,
        fallback_title: str,
        fallback_comment: str = "",
    ) -> dict[str, Any]:
        title = fallback_title
        comment = fallback_comment
        fragments = []
        if isinstance(value, dict):
            title = ensure_text(value.get("title"), fallback_title)
            comment = ensure_text(value.get("comment"), fallback_comment)
            fragments = GroqService._extract_teacher_question_items(value.get("fragments"))
        return {"title": title, "comment": comment, "fragments": fragments}

    @staticmethod
    def _normalize_example_fragments_group(
        value: Any,
        *,
        fallback_title: str,
        fallback_comment: str,
    ) -> dict[str, Any]:
        title = fallback_title
        comment = fallback_comment
        fragments = []
        if isinstance(value, dict):
            title = ensure_text(value.get("title"), fallback_title)
            comment = ensure_text(value.get("comment"), fallback_comment)
            fragments = GroqService._extract_example_span_items(value.get("fragments"))
        return {"title": title, "comment": comment, "fragments": fragments}

    @staticmethod
    def _normalize_recommendation_block(
        value: Any,
        *,
        fallback_title: str,
        fallback_comment: str,
    ) -> dict[str, Any]:
        title = fallback_title
        comment = fallback_comment
        if isinstance(value, dict):
            title = ensure_text(value.get("title"), fallback_title)
            comment = ensure_text(value.get("comment"), fallback_comment)
        return {"title": title, "comment": comment}

    @staticmethod
    def _normalize_flag_group(value: Any, *, fallback_present: bool = False) -> dict[str, Any]:
        fragments = []
        present = fallback_present
        if isinstance(value, dict):
            fragments = GroqService._extract_span_items(value.get("fragments"))
            present = bool(value.get("present", False)) or bool(fragments)
        return {"present": present, "fragments": fragments}

    @staticmethod
    def _extract_goals_and_summary(value: Any) -> dict[str, Any]:
        fallback_goal = {"present": False, "start_ms": None, "comment": ""}
        if not isinstance(value, dict):
            return {"intro": fallback_goal.copy(), "summary": fallback_goal.copy()}

        def normalize_goal(goal_value: Any) -> dict[str, Any]:
            if not isinstance(goal_value, dict):
                return fallback_goal.copy()
            present = bool(goal_value.get("present", False))
            start_ms = goal_value.get("start_ms") if present else None
            if present:
                try:
                    start_ms = int(start_ms)
                except (TypeError, ValueError):
                    start_ms = None
            comment = goal_value.get("comment", "")
            comment = comment.strip() if isinstance(comment, str) else ""
            return {"present": present, "start_ms": start_ms, "comment": comment}

        return {"intro": normalize_goal(value.get("intro")), "summary": normalize_goal(value.get("summary"))}

    @staticmethod
    def _extract_flags(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"profanity": [], "overly_familiar_tone": []}
        return {
            "profanity": GroqService._extract_span_items(value.get("profanity")),
            "overly_familiar_tone": GroqService._extract_span_items(value.get("overly_familiar_tone")),
        }

    @staticmethod
    def _validate_grade_open_answers(data: Any, answers: list[GradeAnswerItem]) -> None:
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект JSON с ключом scores.")
        raw_scores = data.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(answers):
            raise ValueError(f"scores должен быть списком длиной {len(answers)}.")
        for item in raw_scores:
            if isinstance(item, dict):
                if "question_id" not in item or "score" not in item:
                    raise ValueError("Каждый элемент scores должен содержать question_id и score.")
                if item.get("score") not in (0, 1):
                    raise ValueError("score должен быть 0 или 1.")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[1] not in (0, 1):
                    raise ValueError("score должен быть 0 или 1.")
            else:
                raise ValueError("Каждый элемент scores должен быть объектом или парой [question_id, score].")
