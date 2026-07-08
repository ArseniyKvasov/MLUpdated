# AdaptiveLearningML

ML-сервис на FastAPI для:

- транскрибации аудио- и видео-чанков
- мини-конспектов
- итогового конспекта
- генерации теста
- проверки открытых ответов

Сервис работает через внутреннюю очередь задач:

- запросы быстро принимаются и получают `job_id`
- обработка идёт в фоне
- результаты забираются отдельным запросом по `job_id`
- транскрибация и остальные задачи разведены по разным пулам, чтобы не убивать один CPU-воркер аудио-чaнками

## Переменные окружения

Создайте `.env` на основе `.env.example`:

```env
TRANSCRIBER_TYPE=groq  # groq или local
LLM_TYPE=gemini        # gemini или local

GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
API_KEY=your_service_api_key_here

# Настройки для локальных моделей (если выбрано local)
LOCAL_WHISPER_URL=http://localhost:8080/v1
LOCAL_WHISPER_MODEL=whisper-large-v3
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=gemma
```

- `TRANSCRIBER_TYPE` и `LLM_TYPE` переключают режим работы: облачный (groq / gemini) или локальный (local).
- `GROQ_API_KEY` используется для транскрибации через API Groq (один ключ, без пулов).
- `GEMINI_API_KEY` используется для генераций через Google GenAI / Gemini API.
- `API_KEY` требуется для авторизации запросов к сервису через заголовок `X-API-Key`.
- Локальные настройки `LOCAL_WHISPER_URL` и `LOCAL_LLM_URL` позволяют интегрироваться с Whisper.cpp ASR или Ollama.
- Очередь задач можно тюнить переменными `TRANSCRIBE_WORKERS` и `ANALYSIS_WORKERS` для регулирования параллелизма.

`GET /health` возвращает статусы доступности: `groq_ok` и `google_ok` (для локальных моделей опрашиваются соответствующие endpoint-ы).

## Хранилище аудио-чанков (Cloudflare R2)

По умолчанию `POST /transcribe-chunk` сохраняет загруженный файл во временный файл на диске текущего инстанса сервиса. Это ломается, если приём запроса и обработка задачи из очереди происходят на разных инстансах/контейнерах (например, при горизонтальном масштабировании за балансировщиком).

Чтобы это исправить, сервис умеет использовать Cloudflare R2 (S3-совместимое объектное хранилище) как промежуточное хранилище чанков:

- если заданы переменные `CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` — файл загружается в R2, а в очередь задач передаётся `storage_key`; воркер, который может выполняться в другом процессе/контейнере, скачивает файл из R2 перед транскрибацией и удаляет его из R2 и локального диска после обработки;
- если переменные не заданы — поведение не меняется (локальный temp-файл), это подходит для локальной разработки и однопроцессного деплоя.

См. `.env.example` для полного списка переменных и как их получить в Cloudflare Dashboard.


## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker Compose

```bash
docker compose up --build
```

Сервис будет доступен на `http://localhost:8000`.

## Авторизация

Передавайте заголовок:

```http
X-API-Key: your_service_api_key_here
```

Эндпоинт `GET /health` доступен без авторизации.

## Эндпоинты

- `GET /health`
- `POST /transcribe-chunk`
- `POST /chunk-analyze`
- `POST /teacher-analysis-aggregate`
- `POST /lesson-summary`
- `POST /practice-summary`
- `POST /quiz`
- `POST /grade-open-answers`
- `GET /tasks/{job_id}`

## Новый API контракт

Все POST-эндпоинты теперь ставят задачу в очередь и возвращают `202 Accepted`.

### `POST /transcribe-chunk`

Request is now `multipart/form-data`.

```json
{
  "chunk_id": 1,
  "start_ms": 0,
  "end_ms": 15000,
  "audio_file": "<binary file>"
}
```

Response:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "transcribe-chunk",
  "status": "queued"
}
```

Client example:

```bash
curl -X POST "http://localhost:8000/transcribe-chunk" \
  -H "X-API-Key: your_service_api_key_here" \
  -F "chunk_id=1" \
  -F "start_ms=0" \
  -F "end_ms=15000" \
  -F "audio_file=@chunk_001.webm"
```

### `POST /chunk-analyze`

Request body:

```json
{
  "chunk_id": 1,
  "start_time": "00:00",
  "end_time": "00:15",
  "transcript": [
    { "start_ms": 0, "text": "..." }
  ]
}
```

The resulting `GET /tasks/{job_id}` payload will contain:

- `key_points`
- `teacher_questions` with `start_ms`, `end_ms`, `text`, and an English `question_type`
- `student_answers`
- `examples_and_analogies` with `start_ms`, `end_ms`, an English `type`, and paraphrased `text`
- `lesson_events`
- `goals_and_summary`
- `flags`

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "chunk-analyze",
  "status": "queued"
}
```

### `POST /teacher-analysis-aggregate`

Request body:

```json
{
  "chunk_analyses": [
    {
      "chunk_id": 1,
      "start_time": "00:00",
      "end_time": "00:15",
      "key_points": ["..."],
      "teacher_questions": [
        {
          "start_ms": 4200,
          "end_ms": 9800,
          "question_type": "checking_understanding",
          "text": "Поняли, почему это работает?"
        }
      ],
      "student_answers": [],
      "examples_and_analogies": [],
      "lesson_events": [],
      "goals_and_summary": {
        "intro": { "present": false, "start_ms": null, "comment": "" },
        "summary": { "present": false, "start_ms": null, "comment": "" }
      },
      "flags": {
        "profanity": [],
        "overly_familiar_tone": []
      }
    }
  ]
}
```

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "teacher-analysis-aggregate",
  "status": "queued"
}
```

The resulting `GET /tasks/{job_id}` payload will contain an aggregate analysis with:

- `lesson_format.comment` describing the basis for the conclusion
- `audience_engagement.questions_to_students.comment` describing which question types dominate
- `audience_engagement.questions_to_students.fragments` preserving `question_type` for each question
- `material_explanation.examples_and_analogies` containing only real examples, analogies, and storytelling fragments, with an English `type` and a paraphrased `text`
- `teacher_recommendation`

Example result shape:

```json
{
  "lesson_format": {
    "format": "лекция",
    "comment": "..."
  },
  "audience_engagement": {
      "questions_to_students": {
        "title": "Преподаватель активно задаёт вопросы студентам",
        "comment": "...",
        "fragments": [
          {
            "start_ms": 4200,
            "end_ms": 9800,
            "question_type": "checking_understanding",
            "text": "Как вы думаете, почему это происходит?"
          }
        ]
      },
    "student_answers": {
      "title": "Преподаватель реагирует на ответы студентов и развивает обсуждение",
      "comment": "...",
      "fragments": [
        {
          "start_ms": 184000,
          "end_ms": 187500,
          "text": "Да, это хороший пример. Давайте разберём его подробнее."
        }
      ]
    }
  },
  "lesson_structure": {
    "step_by_step_explanation": {
      "timeline": []
    },
    "goals_and_summary": {
      "intro": { "present": false, "start_ms": null, "comment": "" },
      "summary": { "present": false, "start_ms": null, "comment": "" }
    }
  },
  "material_explanation": {
    "examples_and_analogies": {
      "title": "Преподаватель активно использует примеры и аналогии",
      "comment": "",
      "fragments": [
        {
          "start_ms": 355000,
          "end_ms": 363000,
          "type": "analogy",
          "text": "Преподаватель сравнивает идею с ситуацией, когда сначала строят фундамент, а уже потом переходят к следующим этапам."
        }
      ]
    }
  },
  "teacher_recommendation": {
    "title": "Рекомендация преподавателю",
    "comment": "..."
  },
  "flags": {
    "profanity": {
      "present": false,
      "fragments": []
    },
    "overly_familiar_tone": {
      "present": false,
      "fragments": []
    }
  }
}
```

### `POST /lesson-summary`

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "lesson-summary",
  "status": "queued"
}
```

### `POST /quiz`

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "quiz",
  "status": "queued"
}
```

### `POST /practice-summary`

Request body:

```json
{
  "weak_subtopics": ["Сумма Риммана и площадь под графиком"],
  "topics": [
    {
      "subtopic": "Сумма Риммана и площадь под графиком",
      "summary_section": {
        "subtopic": "Сумма Риммана и площадь под графиком",
        "content": "..."
      },
      "mini_summaries": [
        {
          "chunk_id": 1,
          "start_time": "00:00",
          "end_time": "08:03",
          "key_points": ["..."],
          "terms": [],
          "examples": []
        }
      ]
    }
  ],
  "questions": [
    {
      "question_id": "3",
      "question_type": "multiple_choice",
      "subtopic": "Сумма Риммана и площадь под графиком",
      "question_text": "...",
      "student_answer": "",
      "correct_answer": "3",
      "is_correct": false,
      "explanation": "..."
    }
  ]
}
```

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "practice-summary",
  "status": "queued"
}
```

Когда задача завершится, `GET /tasks/{job_id}` вернёт объект с `result.summary`, то есть список секций вида:

```json
{
  "summary": [
    {
      "subtopic": "Сумма Риммана и площадь под графиком",
      "content": "..."
    }
  ]
}
```

### `POST /grade-open-answers`

Response format:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "grade-open-answers",
  "status": "queued"
}
```

### `GET /tasks/{job_id}`

Response:

```json
{
  "job_id": "a1b2c3...",
  "task_type": "transcribe-chunk",
  "status": "succeeded",
  "created_at": "2026-05-10T12:00:00Z",
  "updated_at": "2026-05-10T12:00:07Z",
  "started_at": "2026-05-10T12:00:01Z",
  "finished_at": "2026-05-10T12:00:07Z",
  "result": {...},
  "error": null
}
```

Если задача не завершена, `result` будет `null`, а `status` будет `queued` или `running`.

For `transcribe-chunk`, the `result` field will contain the transcript only after the job reaches `succeeded`.


---

# Инструкция по развертыванию (Production)

ML-сервис обеспечивает работу транскрибации и генеративного ИИ. Он использует асинхронную очередь задач для обработки тяжелых запросов.

## Запуск через Docker

Рекомендуется запускать через Docker Compose вместе с основным приложением из корня проекта.

### 1. Настройка .env
Создайте файл ML/.env на основе ML/.env.example. Минимально необходимые настройки:

```env
# Ключи доступа к провайдерам ИИ
GROQ_API_KEY=ваш_ключ_groq
GEMINI_API_KEY=ваш_ключ_gemini

# Ключ авторизации для Web-сервиса (должен совпадать с ML_API_KEY в Web/.env)
API_KEY=ваш_секретный_ключ_сервиса

# Типы провайдеров (для продакшена обычно groq и gemini)
TRANSCRIBER_TYPE=groq
LLM_TYPE=gemini

# Список моделей (рекомендуемый набор для стабильности)
GEMINI_GENERATION_MODELS=gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.0-flash
```

### 2. Запуск
```bash
docker compose up -d --build ml-service
```

## Основные настройки

- **Таймауты**: В системе настроен таймаут 600 секунд на одну генерацию, что позволяет обрабатывать сложные запросы без обрывов.
- **Очередь воркеров**: Вы можете настроить количество параллельных задач через TRANSCRIBE_WORKERS и ANALYSIS_WORKERS.
- **Отказоустойчивость**: Реализован механизм автоматических повторов (retries) с экспоненциальной задержкой и случайным выбором моделей из списка при сбоях API.

## Мониторинг
Эндпоинт GET /health доступен без авторизации и позволяет проверить доступность внешних API (Groq и Gemini).
