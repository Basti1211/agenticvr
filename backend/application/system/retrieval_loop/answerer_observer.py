"""Independent answer guess recorded after each retrieval-loop iteration."""

from __future__ import annotations

import logging

from application.system.answerer import Answerer, full_history_to_answerer_history
from application.system.retrieval_loop.history import (
    add_timing,
    jsonable,
    llm_history_entries,
)
from application.system.retrieval_loop.state import RunContext
from application.system.retrieval_loop.submission import verify_answerer_output
from application.system.task_input_access import (
    DEFAULT_TASK_INPUT_ACCESS,
    ModuleTaskInputAccess,
    encoded_task_for_module,
    task_videos_for_module,
)
from application.system.timing import elapsed_ms, timer_start
from application.types.answerer.models import AnswererRequest
from application.types.retrieval_loop.models import (
    RetrievalHistoryEntry,
    RetrievalLoopIteration,
)


logger = logging.getLogger(__name__)


class AnswererObserver:
    """Run an answerer without allowing it to alter loop control flow."""

    def __init__(
        self,
        answerer: Answerer | None,
        *,
        task_input_access: ModuleTaskInputAccess | None = None,
        task_reference_video_resolver=None,
    ) -> None:
        self.answerer = answerer
        self.task_input_access = task_input_access or DEFAULT_TASK_INPUT_ACCESS.answerer
        self.task_reference_video_resolver = task_reference_video_resolver

    async def run(
        self,
        context: RunContext,
        iteration: RetrievalLoopIteration,
    ) -> None:
        if self.answerer is None:
            return

        request: AnswererRequest | None = None
        answerer_start = timer_start()
        try:
            request = AnswererRequest(
                encoded_task=encoded_task_for_module(
                    context.encoded,
                    context.task,
                    self.task_input_access,
                ),
                history=full_history_to_answerer_history(context.full_history),
                dataset=context.task.dataset,
                task_type=context.task.task_type,
                task_videos=(
                    task_videos_for_module(
                        context.task,
                        self.task_input_access,
                        self.task_reference_video_resolver,
                    )
                    if self.task_reference_video_resolver is not None
                    else []
                ),
            )
            logger.info("retrieval_loop.answerer.start iteration=%s", iteration.index)
            output = await self.answerer.answer(request)
        except Exception as error:
            duration = elapsed_ms(answerer_start)
            message = str(error) or type(error).__name__
            add_timing(context.timings_ms, "answerer", duration)
            add_timing(iteration.timings_ms, "answerer", duration)
            add_timing(iteration.timings_ms, "total", duration)
            iteration.answerer_error = message
            context.full_history.append(
                RetrievalHistoryEntry(
                    stage="answerer",
                    kind="module",
                    request=jsonable(request),
                    error=message,
                    metadata={"duration_ms": duration, "iteration": iteration.index},
                )
            )
            logger.exception(
                "retrieval_loop.answerer.failed iteration=%s duration_ms=%.3f",
                iteration.index,
                duration,
            )
            return

        duration = elapsed_ms(answerer_start)
        add_timing(context.timings_ms, "answerer", duration)
        add_timing(iteration.timings_ms, "answerer", duration)
        add_timing(iteration.timings_ms, "total", duration)
        iteration.answerer_output = output
        context.full_history.append(
            RetrievalHistoryEntry(
                stage="answerer",
                kind="module",
                request=jsonable(request),
                response=jsonable(output),
                metadata={"duration_ms": duration, "iteration": iteration.index},
            )
        )
        context.full_history.extend(
            llm_history_entries("answerer", output.metadata, iteration=iteration.index)
        )

        validation_start = timer_start()
        try:
            validation = verify_answerer_output(task=context.task, output=output)
        except Exception as error:
            validation_duration = elapsed_ms(validation_start)
            message = str(error) or type(error).__name__
            iteration.answerer_error = f"Answer validation failed: {message}"
            context.full_history.append(
                RetrievalHistoryEntry(
                    stage="answerer.validator",
                    kind="module",
                    request=jsonable(output),
                    error=message,
                    metadata={
                        "duration_ms": validation_duration,
                        "iteration": iteration.index,
                    },
                )
            )
            logger.exception(
                "retrieval_loop.answerer.validator.failed iteration=%s "
                "duration_ms=%.3f",
                iteration.index,
                validation_duration,
            )
        else:
            validation_duration = elapsed_ms(validation_start)
            iteration.answerer_validation = validation
            context.full_history.append(
                RetrievalHistoryEntry(
                    stage="answerer.validator",
                    kind="module",
                    request={
                        "video_id": output.video_id,
                        "timestamp_ms": output.timestamp_ms,
                        "has_ground_truth": context.task.input.answer_video is not None,
                    },
                    response=jsonable(validation),
                    metadata={
                        "duration_ms": validation_duration,
                        "iteration": iteration.index,
                    },
                )
            )

        add_timing(context.timings_ms, "answerer.validator", validation_duration)
        add_timing(iteration.timings_ms, "answerer.validator", validation_duration)
        add_timing(iteration.timings_ms, "total", validation_duration)
        logger.info(
            "retrieval_loop.answerer.done iteration=%s duration_ms=%.3f "
            "video_id=%s timestamp_ms=%s validation=%s",
            iteration.index,
            duration,
            output.video_id,
            output.timestamp_ms,
            (
                iteration.answerer_validation.status
                if iteration.answerer_validation is not None
                else "error"
            ),
        )
