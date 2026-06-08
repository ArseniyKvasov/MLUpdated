from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "ml"
    version: str = "1.0.0"
    groq_ok: bool = False
    google_ok: bool = False


class TranscribeChunkRequest(BaseModel):
    file_name: str
    mime_type: str
    audio_base64: str
    chunk_id: int
    start_ms: int
    end_ms: int


class TranscribeChunkJobRequest(BaseModel):
    file_path: str
    file_name: str
    mime_type: str
    chunk_id: int
    start_ms: int
    end_ms: int


class TranscriptItem(BaseModel):
    start_ms: int
    text: str
    speaker: Optional[str] = None


class TranscribeChunkResponse(BaseModel):
    chunk_id: int
    start_ms: int
    end_ms: int
    transcript: list[TranscriptItem]


class MiniSummaryResponse(BaseModel):
    chunk_id: int
    start_time: str
    end_time: str
    key_points: list[str]


class ChunkAnalyzeRequest(BaseModel):
    chunk_id: int
    start_time: str
    end_time: str
    transcript: list[TranscriptItem]


class TeacherSpeechSpanItem(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class TeacherSpeechQuestionItem(TeacherSpeechSpanItem):
    question_type: str


class TeacherSpeechExampleItem(TeacherSpeechSpanItem):
    type: Literal["example", "analogy", "metaphor", "storytelling"]


class LessonEventItem(BaseModel):
    start_ms: int
    title: str
    description: str


class GoalProgressItem(BaseModel):
    present: bool
    start_ms: Optional[int] = None
    comment: str = ""


class GoalsAndSummaryItem(BaseModel):
    intro: GoalProgressItem
    summary: GoalProgressItem


class SpeechFlagsItem(BaseModel):
    profanity: list[TeacherSpeechSpanItem] = Field(default_factory=list)
    overly_familiar_tone: list[TeacherSpeechSpanItem] = Field(default_factory=list)


class ChunkAnalyzeResponse(BaseModel):
    chunk_id: int
    start_time: str
    end_time: str
    key_points: list[str] = Field(default_factory=list)
    teacher_questions: list[TeacherSpeechQuestionItem]
    student_answers: list[TeacherSpeechSpanItem]
    examples_and_analogies: list[TeacherSpeechExampleItem]
    lesson_events: list[LessonEventItem]
    goals_and_summary: GoalsAndSummaryItem
    flags: SpeechFlagsItem


class TeacherAnalysisSpanGroup(BaseModel):
    title: Optional[str] = None
    comment: str = ""
    fragments: list[TeacherSpeechSpanItem] = Field(default_factory=list)


class TeacherAnalysisQuestionGroup(BaseModel):
    title: Optional[str] = None
    comment: str = ""
    fragments: list[TeacherSpeechQuestionItem] = Field(default_factory=list)


class TeacherAnalysisExampleGroup(BaseModel):
    title: Optional[str] = None
    comment: str = ""
    fragments: list[TeacherSpeechExampleItem] = Field(default_factory=list)


class TeacherAnalysisFlagGroup(BaseModel):
    present: bool
    fragments: list[TeacherSpeechSpanItem] = Field(default_factory=list)


class TeacherAnalysisLessonFormatItem(BaseModel):
    format: str
    comment: str


class TeacherAnalysisRecommendationItem(BaseModel):
    title: str
    comment: str


class TeacherAnalysisAudienceEngagementItem(BaseModel):
    questions_to_students: TeacherAnalysisQuestionGroup
    student_answers: TeacherAnalysisSpanGroup


class TeacherAnalysisTimelineItem(BaseModel):
    start_ms: int
    title: str
    description: str


class TeacherAnalysisStepByStepExplanationItem(BaseModel):
    timeline: list[TeacherAnalysisTimelineItem] = Field(default_factory=list)


class TeacherAnalysisLessonStructureItem(BaseModel):
    step_by_step_explanation: TeacherAnalysisStepByStepExplanationItem
    goals_and_summary: GoalsAndSummaryItem


class TeacherAnalysisMaterialExplanationItem(BaseModel):
    examples_and_analogies: TeacherAnalysisExampleGroup


class TeacherAnalysisFlagsItem(BaseModel):
    profanity: TeacherAnalysisFlagGroup
    overly_familiar_tone: TeacherAnalysisFlagGroup


class TeacherAnalysisAggregateRequest(BaseModel):
    chunk_analyses: list[ChunkAnalyzeResponse] = Field(min_length=1)


class TeacherAnalysisAggregateResponse(BaseModel):
    lesson_format: TeacherAnalysisLessonFormatItem
    audience_engagement: TeacherAnalysisAudienceEngagementItem
    lesson_structure: TeacherAnalysisLessonStructureItem
    material_explanation: TeacherAnalysisMaterialExplanationItem
    teacher_recommendation: TeacherAnalysisRecommendationItem
    flags: TeacherAnalysisFlagsItem


class LessonSummaryRequest(BaseModel):
    topic_hint: Optional[str] = None
    key_points: list[str]


class LessonSummarySection(BaseModel):
    subtopic: str
    content: str


class LessonSummaryResponse(BaseModel):
    summary: list[LessonSummarySection]


class TopicSummarySection(BaseModel):
    subtopic: str
    content: str


class MiniSummaryTopicItem(BaseModel):
    chunk_id: int
    start_time: str
    end_time: str
    key_points: list[str] = Field(min_length=1)
    terms: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class PracticeTopicItem(BaseModel):
    subtopic: str
    summary_section: Optional[TopicSummarySection] = None
    mini_summaries: list[MiniSummaryTopicItem] = Field(default_factory=list)


class PracticeQuestionItem(BaseModel):
    question_id: str
    question_type: str
    subtopic: str
    question_text: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    explanation: Optional[str] = None


class PracticeSummaryRequest(BaseModel):
    weak_subtopics: list[str] = Field(min_length=1)
    topics: list[PracticeTopicItem]
    questions: list[PracticeQuestionItem]


class QuizRequest(BaseModel):
    summary: list[LessonSummarySection]
    mcq_count: int = 5
    open_count: int = 2


class QuizQuestion(BaseModel):
    question_id: int
    question_text: str
    question_type: Literal["multiple_choice", "open_ended"]
    options: Optional[list[str]]
    correct_answer: Union[int, str]
    explanation: str
    subtopic: str


class QuizResponse(BaseModel):
    quiz: list[QuizQuestion]


class GradeAnswerItem(BaseModel):
    question_id: int
    question_text: str
    correct_answer: str
    student_answer: str


class GradeOpenAnswersRequest(BaseModel):
    answers: list[GradeAnswerItem]


class GradeScoreItem(BaseModel):
    question_id: int
    score: int


class GradeOpenAnswersResponse(BaseModel):
    scores: list[GradeScoreItem]


TaskStatus = Literal["queued", "running", "succeeded", "failed"]
TaskType = Literal[
    "transcribe-chunk",
    "chunk-analyze",
    "mini-summary",
    "teacher-analysis",
    "teacher-analysis-aggregate",
    "lesson-summary",
    "practice-summary",
    "quiz",
    "grade-open-answers",
]


class TaskAcceptedResponse(BaseModel):
    job_id: str
    task_type: TaskType
    status: Literal["queued"]


class TaskStatusResponse(BaseModel):
    job_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


MiniSummaryRequest = ChunkAnalyzeRequest
TeacherSpeechAnalysisRequest = ChunkAnalyzeRequest
TeacherSpeechAnalysisResponse = ChunkAnalyzeResponse


class TaskQueuePoolStats(BaseModel):
    queued: int
    running: int
    workers: int
    max_queue_size: int


class TaskQueueDebugResponse(BaseModel):
    transcribe: TaskQueuePoolStats
    analysis: TaskQueuePoolStats
