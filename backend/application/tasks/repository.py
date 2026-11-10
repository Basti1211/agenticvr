"""Repositories for task representations."""

from __future__ import annotations

from uuid import UUID

from application.tasks.models import TaskRepresentation


class InMemoryTaskRepository:
    """Process-local task storage for early development and demos."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, TaskRepresentation] = {}

    def add(self, task: TaskRepresentation) -> TaskRepresentation:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: UUID) -> TaskRepresentation | None:
        return self._tasks.get(task_id)

    def list(self) -> list[TaskRepresentation]:
        return sorted(self._tasks.values(), key=lambda task: task.created_at)


task_repository = InMemoryTaskRepository()

