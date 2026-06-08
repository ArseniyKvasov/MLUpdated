from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, cast
from uuid import uuid4

from app.schemas import TaskAcceptedResponse, TaskStatus, TaskStatusResponse, TaskType


@dataclass
class TaskRecord:
    job_id: str
    task_type: TaskType
    payload: dict[str, Any]
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

    def mark_running(self) -> None:
        now = datetime.now(timezone.utc)
        self.status = "running"
        self.started_at = self.started_at or now
        self.updated_at = now

    def mark_succeeded(self, result: Any) -> None:
        now = datetime.now(timezone.utc)
        self.status = "succeeded"
        self.result = result
        self.finished_at = now
        self.updated_at = now

    def mark_failed(self, error: Exception | str) -> None:
        now = datetime.now(timezone.utc)
        self.status = "failed"
        self.error = str(error)
        self.finished_at = now
        self.updated_at = now

    def to_task_accepted(self) -> TaskAcceptedResponse:
        return TaskAcceptedResponse(job_id=self.job_id, task_type=self.task_type, status="queued")

    def to_task_status(self) -> TaskStatusResponse:
        return TaskStatusResponse(
            job_id=self.job_id,
            task_type=self.task_type,
            status=cast(TaskStatus, self.status),
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            result=self.result,
            error=self.error,
        )


class TaskQueueManager:
    def __init__(
        self,
        executor: Callable[[TaskType, dict[str, Any]], Awaitable[Any]],
        *,
        worker_counts: Optional[dict[str, int]] = None,
        queue_sizes: Optional[dict[str, int]] = None,
        completed_retention_seconds: int = 3600,
        cleanup_interval_seconds: int = 60,
    ) -> None:
        self._executor = executor
        self._worker_counts = worker_counts or {
            "transcribe": 1,
            "analysis": 4,
        }
        self._queue_sizes = queue_sizes or {
            "transcribe": 50,
            "analysis": 100,
        }
        self._queues: dict[str, asyncio.Queue[str]] = {
            name: asyncio.Queue(maxsize=self._queue_sizes.get(name, 100))
            for name in self._worker_counts
        }
        self._jobs: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._started = False
        self._completed_retention_seconds = completed_retention_seconds
        self._cleanup_interval_seconds = cleanup_interval_seconds

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for pool_name, worker_count in self._worker_counts.items():
            for worker_index in range(worker_count):
                worker = asyncio.create_task(self._worker_loop(pool_name, worker_index))
                self._workers.append(worker)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if not self._started:
            return
        for worker in self._workers:
            worker.cancel()
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._cleanup_task is not None:
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        self._workers.clear()
        self._cleanup_task = None
        self._started = False

    async def submit(self, task_type: TaskType, payload: dict[str, Any]) -> TaskRecord:
        pool_name = self._pool_name_for_task(task_type)
        queue = self._queues[pool_name]
        if queue.full():
            raise RuntimeError("Очередь задач переполнена. Попробуйте позже.")

        job = TaskRecord(job_id=uuid4().hex, task_type=task_type, payload=payload)
        async with self._lock:
            self._jobs[job.job_id] = job

        try:
            queue.put_nowait(job.job_id)
        except asyncio.QueueFull as exc:
            async with self._lock:
                self._jobs.pop(job.job_id, None)
            raise RuntimeError("Очередь задач переполнена. Попробуйте позже.") from exc
        return job

    async def get(self, job_id: str) -> Optional[TaskRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def purge_expired(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - self._completed_retention_seconds
        async with self._lock:
            expired_job_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at.timestamp() < cutoff
            ]
            for job_id in expired_job_ids:
                self._jobs.pop(job_id, None)

    async def get_queue_debug(self) -> dict[str, dict[str, int]]:
        async with self._lock:
            stats: dict[str, dict[str, int]] = {}
            for pool_name, queue in self._queues.items():
                queued = queue.qsize()
                running = sum(
                    1
                    for job in self._jobs.values()
                    if self._pool_name_for_task(job.task_type) == pool_name and job.status == "running"
                )
                stats[pool_name] = {
                    "queued": queued,
                    "running": running,
                    "workers": self._worker_counts.get(pool_name, 0),
                    "max_queue_size": self._queue_sizes.get(pool_name, 0),
                }
            return stats

    async def _worker_loop(self, pool_name: str, worker_index: int) -> None:
        queue = self._queues[pool_name]
        while True:
            job_id: Optional[str] = None
            try:
                job_id = await queue.get()
            except asyncio.CancelledError:
                # Не вызываем task_done(), т.к. job_id не получен
                raise

            try:
                job = await self.get(job_id)
                if job is None:
                    continue
                job.mark_running()
                try:
                    result = await self._executor(job.task_type, job.payload)
                    job.mark_succeeded(result)
                except Exception as exc:
                    job.mark_failed(exc)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep the worker alive even if a single job blows up in an unexpected way.
                continue
            finally:
                if job_id is not None:
                    queue.task_done()

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval_seconds)
                await self.purge_expired()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue

    @staticmethod
    def _pool_name_for_task(task_type: TaskType) -> str:
        if task_type == "transcribe-chunk":
            return "transcribe"
        return "analysis"
