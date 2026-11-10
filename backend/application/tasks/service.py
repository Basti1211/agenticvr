"""Task application service."""

from __future__ import annotations

from uuid import UUID

from application.system.task_encoder.encoder import VLMTaskEncoder, TaskEncoder
from application.system.task_input_access import (
    MODULE_NAMES,
    TaskInputAccessSettings,
    load_task_input_access_settings,
    task_description_for_module,
)
from application.tasks.models import TaskCreate, TaskRepresentation, TaskStatus
from application.tasks.repository import InMemoryTaskRepository, task_repository
from application.types.retrieval_loop import RetrievalLoopResult


class TaskService:
    """Coordinates task persistence and the task encoder boundary."""

    def __init__(
        self,
        repository: InMemoryTaskRepository,
        encoder: TaskEncoder,
        task_input_access: TaskInputAccessSettings | None = None,
    ) -> None:
        self.repository = repository
        self.encoder = encoder
        self.task_input_access = task_input_access
        self._loop_results: dict[UUID, RetrievalLoopResult] = {}

    async def create_task(self, payload: TaskCreate) -> TaskRepresentation:
        encoded_task = await self.encoder.encode(payload)
        task_input_access = self.task_input_access or load_task_input_access_settings()
        status = (
            TaskStatus.ENCODED
            if encoded_task.status == "encoded"
            else TaskStatus.ENCODE_PENDING
        )
        task = TaskRepresentation(
            status=status,
            task_type=payload.task_type,
            dataset=payload.dataset,
            input=payload.input,
            source_task_name=payload.source_task_name,
            encoded_task=encoded_task,
            task_input_descriptions={
                module_name: description
                for module_name in MODULE_NAMES
                if (
                    description := task_description_for_module(
                        encoded_task,
                        payload,
                        getattr(task_input_access, module_name),
                    )
                )
            },
        )
        return self.repository.add(task)

    def get_task(self, task_id: UUID) -> TaskRepresentation | None:
        return self.repository.get(task_id)

    def list_tasks(self) -> list[TaskRepresentation]:
        return self.repository.list()

    def get_loop_result(self, task_id: UUID) -> RetrievalLoopResult | None:
        return self._loop_results.get(task_id)

    def set_loop_result(
        self,
        task_id: UUID,
        result: RetrievalLoopResult,
    ) -> RetrievalLoopResult:
        self._loop_results[task_id] = result
        return result

    def clear_loop_result(self, task_id: UUID) -> None:
        self._loop_results.pop(task_id, None)


task_service = TaskService(
    repository=task_repository,
    encoder=VLMTaskEncoder(),
)
