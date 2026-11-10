"""Staged image and video browsing for retrieved candidates."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from application.settings.settings import _browser_settings
from application.system.browser import Browser
from application.system.retrieval_loop.candidates import (
    CandidateImageResolver,
    CandidateVideoResolver,
    browser_scene_window_settings_ms,
    candidate_browser_scene_interval_ms,
    candidate_id,
    candidate_ids,
    candidate_source_video_duration_seconds,
    candidate_to_browser_image_candidate,
    candidate_to_browser_video_candidate,
)
from application.system.retrieval_loop.history import (
    add_timing,
    jsonable,
    llm_history_entries,
    planner_feedback_history_step,
)
from application.system.retrieval_loop.state import IterationContext
from application.system.task_input_access import (
    DEFAULT_TASK_INPUT_ACCESS,
    ModuleTaskInputAccess,
    task_description_for_module,
    task_images_for_module,
    task_videos_for_module,
)
from application.system.timing import elapsed_ms, timer_start
from application.tasks.models import Dataset, EncodedTask, TaskCreate
from application.types.browser.models import (
    BrowserCandidate,
    BrowserClassification,
    BrowserClassificationOutput,
    BrowserDescription,
    BrowserDescriptionOutput,
    BrowserFrameAnalysisOutput,
    BrowserJudgement,
    BrowserReasoning,
    BrowserReasoningOutput,
    BrowserRequest,
    BrowserSceneAnalysisOutput,
    KeyframeDecision,
    SceneVerdict,
)
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import RetrievalHistoryEntry


logger = logging.getLogger(__name__)
DEFAULT_IMAGE_DOWNLOAD_CONCURRENCY = 8
DEFAULT_VIDEO_MATERIALIZATION_CONCURRENCY = 3


def _positive_concurrency_setting(
    settings: dict[str, object],
    name: str,
    default: int,
) -> int:
    value = int(settings.get(name, default))
    if value < 1:
        raise ValueError(f"Browser {name} must be at least 1.")
    return value


@dataclass(frozen=True)
class BrowserCandidates:
    """Materialized evidence and its source-candidate mapping."""

    candidates: list[BrowserCandidate]
    request: BrowserRequest
    by_id: dict[str, Candidate]


@dataclass(frozen=True)
class BrowserInspection:
    """Combined result of image screening and optional video inspection."""

    evidence: BrowserCandidates
    output: BrowserClassificationOutput
    descriptions: BrowserDescriptionOutput
    inspected_video_candidate_ids: tuple[str, ...] = ()
    inspected_video_intervals_ms: tuple[tuple[str, int, int], ...] = ()
    exact_candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class _SceneWindow:
    """One bounded video window shared by overlapping routed keyframes."""

    candidate: Candidate
    member_ids: tuple[str, ...]
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class _BrowserClassificationPass:
    output: BrowserClassificationOutput
    video_candidates: BrowserCandidates | None
    video_candidate_ids: list[str]
    scene_windows: tuple[_SceneWindow, ...]
    scene_descriptions: BrowserDescriptionOutput
    exact_candidates: tuple[Candidate, ...]
    duration_ms: float


class BrowserPipeline:
    """Screen images, inspect promising scenes, and retain useful feedback."""

    def __init__(
        self,
        *,
        browser: Browser,
        image_resolver: CandidateImageResolver | None,
        video_resolver: CandidateVideoResolver | None,
        video_followup: bool,
        task_input_access: ModuleTaskInputAccess | None = None,
        task_reference_video_resolver=None,
    ) -> None:
        self.browser = browser
        self.image_resolver = image_resolver
        self.video_resolver = video_resolver
        self.video_followup = video_followup
        self.task_input_access = task_input_access or DEFAULT_TASK_INPUT_ACCESS.browser
        self.task_reference_video_resolver = task_reference_video_resolver
        settings = _browser_settings()
        self.image_download_concurrency = _positive_concurrency_setting(
            settings,
            "image_download_concurrency",
            DEFAULT_IMAGE_DOWNLOAD_CONCURRENCY,
        )
        self.video_materialization_concurrency = _positive_concurrency_setting(
            settings,
            "video_materialization_concurrency",
            DEFAULT_VIDEO_MATERIALIZATION_CONCURRENCY,
        )

    async def inspect(
        self,
        context: IterationContext,
        candidates: list[Candidate],
    ) -> BrowserInspection:
        browser_start = timer_start()
        logger.info(
            "retrieval_loop.browser.start iteration=%s candidate_count=%s "
            "candidate_ids=%s",
            context.index,
            len(candidates),
            candidate_ids(candidates),
        )
        try:
            image_candidates = await self._resolve_image_candidates(context, candidates)
            inspection = await self._classify(context, image_candidates)
        except Exception:
            logger.exception(
                "retrieval_loop.browser.failed iteration=%s duration_ms=%.3f "
                "candidate_ids=%s",
                context.index,
                elapsed_ms(browser_start),
                candidate_ids(candidates),
            )
            raise

        duration = elapsed_ms(browser_start)
        add_timing(context.run.timings_ms, "browser", duration)
        context.iteration.timings_ms["browser"] = duration
        logger.info(
            "retrieval_loop.browser.done iteration=%s duration_ms=%.3f "
            "candidate_count=%s",
            context.index,
            duration,
            len(candidates),
        )
        return inspection

    async def record_feedback(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
        *,
        include_accepted_evaluations: bool = False,
    ) -> None:
        """Record keyframe- and scene-level browser feedback for the planner."""

        run = context.run
        classifications_by_id = {
            result.candidate_id: result for result in inspection.output.results
        }
        reasoning = BrowserReasoningOutput(
            relevant=[
                BrowserReasoning(
                    candidate_ids=[candidate_id],
                    judgement=classifications_by_id[candidate_id].judgement,
                    reasoning=(
                        _classification_feedback(
                            classifications_by_id[candidate_id]
                        )
                        or inspection.descriptions.description_for(
                            candidate_id,
                            "scene",
                        )
                        or inspection.descriptions.description_for(
                            candidate_id, "frame"
                        )
                        or "The browser classified this keyframe as relevant."
                    ),
                )
                for candidate_id in (
                    inspection.output.relevant_candidate_ids
                    + inspection.output.match_candidate_ids
                )
            ],
            metadata=inspection.descriptions.metadata,
        )
        context.iteration.browser_reasoning_output = reasoning
        run.condensed_history.add_step(
            planner_feedback_history_step(
                action=context.decision.action,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
                candidates_by_id=inspection.evidence.by_id,
                browser_output=inspection.output,
                browser_reasoning=reasoning,
                browser_descriptions=inspection.descriptions,
                inspected_video_candidate_ids=set(
                    inspection.inspected_video_candidate_ids
                ),
                inspected_video_intervals_ms={
                    candidate_id: (start_ms, end_ms)
                    for candidate_id, start_ms, end_ms in (
                        inspection.inspected_video_intervals_ms
                    )
                },
                evaluator_results=context.iteration.evaluator_results,
                submissions=context.iteration.submissions,
                include_accepted_evaluations=include_accepted_evaluations,
            )
        )
        filtered_candidate_ids = candidate_ids(context.iteration.filtered_candidates)
        if filtered_candidate_ids:
            run.condensed_history.include_seen_keyframes(
                filtered_candidate_ids,
                returned_candidate_count=(
                    len(context.iteration.candidates)
                    + len(context.iteration.filtered_candidates)
                ),
            )

    async def _resolve_image_candidates(
        self,
        context: IterationContext,
        candidates: list[Candidate],
    ) -> BrowserCandidates:
        resolution_start = timer_start()
        try:
            self._log_candidate_resolution_start(
                context, "image", candidate_ids(candidates)
            )
            semaphore = asyncio.Semaphore(self.image_download_concurrency)

            async def resolve_candidate(
                candidate: Candidate,
                ordinal: int,
            ) -> BrowserCandidate:
                async with semaphore:
                    return await self._resolve_image_candidate(
                        context,
                        candidate,
                        ordinal,
                        candidates,
                    )

            browser_candidates = await asyncio.gather(
                *(
                    resolve_candidate(candidate, ordinal)
                    for ordinal, candidate in enumerate(candidates, start=1)
                )
            )
        except Exception:
            logger.exception(
                "retrieval_loop.candidate_image_resolution.failed "
                "iteration=%s duration_ms=%.3f candidate_ids=%s",
                context.index,
                elapsed_ms(resolution_start),
                candidate_ids(candidates),
            )
            raise

        duration = elapsed_ms(resolution_start)
        self._record_candidate_resolution(
            context,
            timing_key="candidate_resolution",
            media_kind="image",
            duration=duration,
            candidates=browser_candidates,
        )
        return BrowserCandidates(
            candidates=browser_candidates,
            request=self._image_browser_request(context, browser_candidates),
            by_id=self._browser_candidates_by_id(browser_candidates, candidates),
        )

    async def _resolve_video_candidates(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        scene_windows: tuple[_SceneWindow, ...],
    ) -> BrowserCandidates:
        candidate_ids_to_inspect = [
            candidate_id(window.candidate) for window in scene_windows
        ]
        resolution_start = timer_start()
        try:
            self._log_candidate_resolution_start(
                context,
                "video",
                candidate_ids_to_inspect,
            )
            semaphore = asyncio.Semaphore(self.video_materialization_concurrency)

            async def resolve_candidate(
                scene_window: _SceneWindow,
                ordinal: int,
            ) -> BrowserCandidate:
                async with semaphore:
                    return await self._resolve_video_candidate(
                        context,
                        scene_window,
                        ordinal,
                        len(scene_windows),
                    )

            browser_candidates = await asyncio.gather(
                *(
                    resolve_candidate(scene_window, ordinal)
                    for ordinal, scene_window in enumerate(
                        scene_windows,
                        start=1,
                    )
                )
            )
        except Exception:
            logger.exception(
                "retrieval_loop.candidate_video_resolution.failed "
                "iteration=%s duration_ms=%.3f candidate_ids=%s",
                context.index,
                elapsed_ms(resolution_start),
                candidate_ids_to_inspect,
            )
            raise

        duration = elapsed_ms(resolution_start)
        self._record_candidate_resolution(
            context,
            timing_key="candidate_video_resolution",
            media_kind="video",
            duration=duration,
            candidates=browser_candidates,
        )
        return BrowserCandidates(
            candidates=browser_candidates,
            request=self._video_browser_request(
                context, image_candidates, browser_candidates
            ),
            by_id=image_candidates.by_id,
        )

    async def _resolve_image_candidate(
        self,
        context: IterationContext,
        candidate: Candidate,
        ordinal: int,
        candidates: list[Candidate],
    ) -> BrowserCandidate:
        identifier = candidate_id(candidate)
        candidate_start = timer_start()
        self._log_materialize_start(
            context,
            "image",
            identifier,
            ordinal,
            len(candidates),
        )
        try:
            browser_candidate = await self._materialize_image_candidate(candidate)
        except Exception as exc:
            self._log_materialize_failed(context, "image", identifier, candidate_start)
            raise RuntimeError(
                f"Failed to materialize browser image for candidate {identifier!r}."
            ) from exc
        self._log_materialize_done(
            context, "image", identifier, candidate_start, browser_candidate
        )
        return browser_candidate

    async def _resolve_video_candidate(
        self,
        context: IterationContext,
        scene_window: _SceneWindow,
        ordinal: int,
        candidate_count: int,
    ) -> BrowserCandidate:
        candidate = scene_window.candidate
        candidate_identifier = candidate_id(candidate)
        candidate_start = timer_start()
        self._log_video_materialize_start(
            context,
            candidate,
            candidate_identifier,
            ordinal,
            candidate_count,
        )
        try:
            browser_candidate = await self._materialize_video_candidate(
                candidate,
                context.run.task.dataset,
                (scene_window.start_ms, scene_window.end_ms),
            )
        except Exception as exc:
            self._log_materialize_failed(
                context, "video", candidate_identifier, candidate_start
            )
            raise RuntimeError(
                "Failed to materialize browser video for candidate "
                f"{candidate_identifier!r}: {exc}"
            ) from exc
        self._log_materialize_done(
            context,
            "video",
            candidate_identifier,
            candidate_start,
            browser_candidate,
        )
        return browser_candidate

    def _log_candidate_resolution_start(
        self,
        context: IterationContext,
        media_kind: str,
        candidate_identifiers: list[str],
    ) -> None:
        logger.info(
            "retrieval_loop.candidate_%s_resolution.start "
            "iteration=%s candidate_count=%s concurrency_limit=%s candidate_ids=%s",
            media_kind,
            context.index,
            len(candidate_identifiers),
            (
                self.image_download_concurrency
                if media_kind == "image"
                else self.video_materialization_concurrency
            ),
            candidate_identifiers,
        )

    def _record_candidate_resolution(
        self,
        context: IterationContext,
        *,
        timing_key: str,
        media_kind: str,
        duration: float,
        candidates: list[BrowserCandidate],
    ) -> None:
        add_timing(context.run.timings_ms, timing_key, duration)
        context.iteration.timings_ms[timing_key] = duration
        logger.info(
            "retrieval_loop.candidate_%s_resolution.done "
            "iteration=%s duration_ms=%.3f browser_candidate_ids=%s",
            media_kind,
            context.index,
            duration,
            [candidate.identifier for candidate in candidates],
        )

    def _image_browser_request(
        self,
        context: IterationContext,
        browser_candidates: list[BrowserCandidate],
    ) -> BrowserRequest:
        run = context.run
        return BrowserRequest(
            task_description=task_description_for_module(
                run.encoded,
                run.task,
                self.task_input_access,
            ),
            candidates=browser_candidates,
            task_images=task_images_for_module(run.encoded, self.task_input_access),
            task_videos=(
                task_videos_for_module(
                    run.task,
                    self.task_input_access,
                    self.task_reference_video_resolver,
                )
                if self.task_reference_video_resolver is not None
                else []
            ),
            task_type=run.task.task_type,
            dataset=run.task.dataset,
        )

    def _video_browser_request(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        browser_candidates: list[BrowserCandidate],
    ) -> BrowserRequest:
        return BrowserRequest(
            task_description=image_candidates.request.task_description,
            candidates=browser_candidates,
            task_images=image_candidates.request.task_images,
            task_videos=image_candidates.request.task_videos,
            task_type=context.run.task.task_type,
            dataset=context.run.task.dataset,
        )

    def _browser_candidates_by_id(
        self,
        browser_candidates: list[BrowserCandidate],
        candidates: list[Candidate],
    ) -> dict[str, Candidate]:
        return {
            browser_candidate.identifier: candidate
            for browser_candidate, candidate in zip(
                browser_candidates,
                candidates,
                strict=True,
            )
        }

    def _log_materialize_start(
        self,
        context: IterationContext,
        media_kind: str,
        candidate_identifier: str,
        ordinal: int,
        candidate_count: int,
    ) -> None:
        logger.info(
            "timing.browser.%s_materialize.start iteration=%s "
            "candidate_id=%s ordinal=%s candidate_count=%s",
            media_kind,
            context.index,
            candidate_identifier,
            ordinal,
            candidate_count,
        )

    def _log_video_materialize_start(
        self,
        context: IterationContext,
        candidate: Candidate,
        candidate_identifier: str,
        ordinal: int,
        candidate_count: int,
    ) -> None:
        logger.info(
            "timing.browser.video_materialize.start iteration=%s "
            "candidate_id=%s ordinal=%s candidate_count=%s "
            "scene_start_ms=%s scene_end_ms=%s",
            context.index,
            candidate_identifier,
            ordinal,
            candidate_count,
            candidate.time_scene_start,
            candidate.time_scene_end,
        )

    def _log_materialize_failed(
        self,
        context: IterationContext,
        media_kind: str,
        candidate_identifier: str,
        candidate_start: float,
    ) -> None:
        logger.exception(
            "timing.browser.%s_materialize.failed iteration=%s "
            "candidate_id=%s duration_ms=%.3f",
            media_kind,
            context.index,
            candidate_identifier,
            elapsed_ms(candidate_start),
        )

    def _log_materialize_done(
        self,
        context: IterationContext,
        media_kind: str,
        candidate_identifier: str,
        candidate_start: float,
        candidate: BrowserCandidate,
    ) -> None:
        logger.info(
            "timing.browser.%s_materialize.done iteration=%s "
            "candidate_id=%s duration_ms=%.3f file_bytes=%s",
            media_kind,
            context.index,
            candidate_identifier,
            elapsed_ms(candidate_start),
            _candidate_file_size(candidate),
        )

    async def _materialize_image_candidate(
        self,
        candidate: Candidate,
    ) -> BrowserCandidate:
        return await asyncio.to_thread(
            candidate_to_browser_image_candidate,
            candidate,
            self.image_resolver,
        )

    async def _materialize_video_candidate(
        self,
        candidate: Candidate,
        dataset: Dataset,
        scene_interval_ms: tuple[int, int],
    ) -> BrowserCandidate:
        return await asyncio.to_thread(
            candidate_to_browser_video_candidate,
            candidate,
            dataset,
            self.video_resolver,
            scene_interval_ms=scene_interval_ms,
        )

    async def _classify(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
    ) -> BrowserInspection:
        browser_start = timer_start()
        try:
            frame_analysis = await self._run_frame_analysis(
                context,
                image_candidates,
            )
            frame_descriptions = frame_analysis_descriptions(frame_analysis)
            classification = await self._classification_pass(
                context,
                image_candidates,
                frame_analysis,
                browser_start,
            )
            self._record_classification(
                context,
                image_candidates,
                classification,
            )
            descriptions = merge_browser_descriptions(
                frame_descriptions,
                classification.scene_descriptions,
            )
        except Exception:
            logger.exception(
                "retrieval_loop.browser.classify.failed iteration=%s "
                "duration_ms=%.3f candidate_ids=%s",
                context.index,
                elapsed_ms(browser_start),
                [candidate.identifier for candidate in image_candidates.candidates],
            )
            raise

        context.iteration.browser_description_output = descriptions
        self._log_classification_done(context, image_candidates, classification)
        return BrowserInspection(
            evidence=image_candidates,
            output=classification.output,
            descriptions=descriptions,
            inspected_video_candidate_ids=(
                tuple(
                    member_id
                    for window in classification.scene_windows
                    for member_id in window.member_ids
                )
            ),
            inspected_video_intervals_ms=tuple(
                (member_id, window.start_ms, window.end_ms)
                for window in classification.scene_windows
                for member_id in window.member_ids
            ),
            exact_candidates=classification.exact_candidates,
        )

    async def _classification_pass(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        frame_analysis: BrowserFrameAnalysisOutput,
        browser_start: float,
    ) -> _BrowserClassificationPass:
        scene_windows = (
            await self._scene_windows(image_candidates, frame_analysis)
            if self.video_followup
            else ()
        )
        video_candidate_ids = [candidate_id(window.candidate) for window in scene_windows]
        (
            video_candidates,
            video_output,
            scene_descriptions,
        ) = await self._run_scene_analysis(
            context,
            image_candidates,
            scene_windows,
        )
        browser_output = merge_browser_classifications(
            frame_analysis,
            video_output,
            scene_windows,
        )
        scene_descriptions = expand_scene_descriptions(
            scene_descriptions,
            scene_windows,
        )
        exact_candidates = exact_scene_candidates(video_output, scene_windows)
        classification_duration = elapsed_ms(browser_start)
        add_timing(context.run.timings_ms, "browser.classify", classification_duration)
        context.iteration.timings_ms["browser.classify"] = classification_duration
        context.iteration.browser_output = browser_output
        return _BrowserClassificationPass(
            output=browser_output,
            video_candidates=video_candidates,
            video_candidate_ids=video_candidate_ids,
            scene_windows=scene_windows,
            scene_descriptions=scene_descriptions,
            exact_candidates=exact_candidates,
            duration_ms=classification_duration,
        )

    async def _run_frame_analysis(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
    ) -> BrowserFrameAnalysisOutput:
        logger.info(
            "retrieval_loop.browser.frame_analysis.start "
            "iteration=%s candidate_count=%s candidate_ids=%s",
            context.index,
            len(image_candidates.candidates),
            [candidate.identifier for candidate in image_candidates.candidates],
        )
        image_start = timer_start()
        frame_output = await self.browser.analyze_frames(image_candidates.request)
        image_duration = elapsed_ms(image_start)
        add_timing(context.run.timings_ms, "browser.frame_analysis", image_duration)
        context.iteration.timings_ms["browser.frame_analysis"] = image_duration
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="browser.frame_analysis",
                kind="module",
                request=jsonable(image_candidates.request),
                response=jsonable(frame_output),
                metadata={
                    "duration_ms": image_duration,
                    "iteration": context.index,
                    "candidate_count": len(image_candidates.candidates),
                },
            )
        )
        logger.info(
            "retrieval_loop.browser.frame_analysis.done "
            "iteration=%s duration_ms=%.3f candidate_count=%s",
            context.index,
            image_duration,
            len(frame_output.results),
        )
        return frame_output

    async def _scene_windows(
        self,
        image_candidates: BrowserCandidates,
        frame_output: BrowserFrameAnalysisOutput,
    ) -> tuple[_SceneWindow, ...]:
        routed = [
            image_candidates.by_id[result.candidate_id]
            for result in frame_output.results
            if result.decision == KeyframeDecision.INSPECT_SCENE
            and result.candidate_id in image_candidates.by_id
        ]
        if not routed:
            return ()
        dataset = image_candidates.request.dataset
        if dataset is None:
            raise ValueError("Browser scene inspection requires a dataset.")

        candidates_by_video: dict[str, list[Candidate]] = {}
        for candidate in routed:
            video_id = _candidate_video_identifier(candidate)
            candidates_by_video.setdefault(video_id, []).append(candidate)

        async def source_duration_ms(candidates: list[Candidate]) -> int:
            seconds = await asyncio.to_thread(
                candidate_source_video_duration_seconds,
                candidates[0],
                dataset,
            )
            return round(seconds * 1000)

        durations = await asyncio.gather(
            *(source_duration_ms(candidates) for candidates in candidates_by_video.values())
        )
        source_duration_by_video = dict(zip(candidates_by_video, durations, strict=True))

        windows: list[_SceneWindow] = []
        _, minimum_ms, maximum_ms = browser_scene_window_settings_ms()
        for video_id, candidates in candidates_by_video.items():
            bounded = [
                (
                    candidate,
                    candidate_browser_scene_interval_ms(
                        candidate,
                        dataset,
                        source_duration_ms=source_duration_by_video[video_id],
                    ),
                )
                for candidate in candidates
            ]
            bounded.sort(key=lambda item: (item[1][0], item[1][1]))
            for candidate, (start_ms, end_ms) in bounded:
                identifier = candidate_id(candidate)
                overlaps_previous = (
                    bool(windows)
                    and _candidate_video_identifier(windows[-1].candidate) == video_id
                    and start_ms <= windows[-1].end_ms
                )
                if overlaps_previous:
                    previous = windows[-1]
                    merged_end = max(previous.end_ms, end_ms)
                    if merged_end - previous.start_ms <= maximum_ms:
                        windows[-1] = _SceneWindow(
                            candidate=previous.candidate.model_copy(
                                update={
                                    "time_scene_start": previous.start_ms,
                                    "time_scene_end": merged_end,
                                }
                            ),
                            member_ids=(*previous.member_ids, identifier),
                            start_ms=previous.start_ms,
                            end_ms=merged_end,
                        )
                        continue

                    # The combined coverage is too long for one clip. Partition the
                    # overlap so the same footage is not sent twice while retaining
                    # a useful window on both sides of the boundary.
                    boundary_ms = (start_ms + previous.end_ms) // 2
                    boundary_ms = max(
                        previous.start_ms + minimum_ms,
                        min(boundary_ms, merged_end - minimum_ms),
                    )
                    boundary_ms = max(start_ms, min(previous.end_ms, boundary_ms))
                    if previous.start_ms < boundary_ms < merged_end:
                        windows[-1] = _SceneWindow(
                            candidate=previous.candidate.model_copy(
                                update={"time_scene_end": boundary_ms}
                            ),
                            member_ids=previous.member_ids,
                            start_ms=previous.start_ms,
                            end_ms=boundary_ms,
                        )
                        start_ms = boundary_ms

                windows.append(
                    _SceneWindow(
                        candidate=candidate.model_copy(
                            update={
                                "time_scene_start": start_ms,
                                "time_scene_end": end_ms,
                            }
                        ),
                        member_ids=(identifier,),
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
        return tuple(windows)

    async def _run_scene_analysis(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        scene_windows: tuple[_SceneWindow, ...],
    ) -> tuple[
        BrowserCandidates | None,
        BrowserClassificationOutput | None,
        BrowserDescriptionOutput,
    ]:
        if not self.video_followup:
            logger.info(
                "retrieval_loop.browser.scene_analysis.skipped "
                "iteration=%s reason=image_only_override",
                context.index,
            )
            return None, None, BrowserDescriptionOutput()
        if not scene_windows:
            logger.info(
                "retrieval_loop.browser.scene_analysis.skipped "
                "iteration=%s reason=no_image_matches",
                context.index,
            )
            return None, None, BrowserDescriptionOutput()

        video_candidates = await self._resolve_video_candidates(
            context,
            image_candidates,
            scene_windows,
        )
        logger.info(
            "retrieval_loop.browser.scene_analysis.start "
            "iteration=%s candidate_count=%s candidate_ids=%s",
            context.index,
            len(video_candidates.candidates),
            [candidate.identifier for candidate in video_candidates.candidates],
        )
        video_start = timer_start()
        scene_analysis = await self.browser.analyze_scenes(video_candidates.request)
        video_output = scene_analysis_classifications(scene_analysis)
        scene_descriptions = scene_analysis_descriptions(scene_analysis)
        video_duration = elapsed_ms(video_start)
        add_timing(context.run.timings_ms, "browser.scene_analysis", video_duration)
        context.iteration.timings_ms["browser.scene_analysis"] = video_duration
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="browser.scene_analysis",
                kind="module",
                request=jsonable(video_candidates.request),
                response=jsonable(scene_analysis),
                metadata={
                    "duration_ms": video_duration,
                    "iteration": context.index,
                    "candidate_count": len(video_candidates.candidates),
                },
            )
        )
        logger.info(
            "retrieval_loop.browser.scene_analysis.done "
            "iteration=%s duration_ms=%.3f match_count=%s "
            "relevant_count=%s irrelevant_count=%s",
            context.index,
            video_duration,
            len(video_output.match_candidate_ids),
            len(video_output.relevant_candidate_ids),
            len(video_output.irrelevant_candidate_ids),
        )
        return video_candidates, video_output, scene_descriptions

    def _record_classification(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        classification: _BrowserClassificationPass,
    ) -> None:
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="browser.classify",
                kind="module",
                request={
                    "frame_analysis": jsonable(image_candidates.request),
                    "scene_analysis": (
                        jsonable(classification.video_candidates.request)
                        if classification.video_candidates is not None
                        else None
                    ),
                },
                response=jsonable(classification.output),
                metadata={
                    "duration_ms": classification.duration_ms,
                    "iteration": context.index,
                    "image_candidate_count": len(image_candidates.candidates),
                    "video_candidate_count": (
                        len(classification.video_candidates.candidates)
                        if classification.video_candidates is not None
                        else 0
                    ),
                },
            )
        )
        context.run.full_history.extend(
            llm_history_entries(
                "browser.classify",
                classification.output.metadata,
                iteration=context.index,
            )
        )

    def _log_classification_done(
        self,
        context: IterationContext,
        image_candidates: BrowserCandidates,
        classification: _BrowserClassificationPass,
    ) -> None:
        logger.info(
            "retrieval_loop.browser.classify.done iteration=%s duration_ms=%.3f "
            "image_candidate_count=%s video_candidate_count=%s "
            "exact_count=%s relevant_count=%s irrelevant_count=%s exact_ids=%s",
            context.index,
            classification.duration_ms,
            len(image_candidates.candidates),
            (
                len(classification.video_candidates.candidates)
                if classification.video_candidates is not None
                else 0
            ),
            len(classification.output.match_candidate_ids),
            len(classification.output.relevant_candidate_ids),
            len(classification.output.irrelevant_candidate_ids),
            classification.output.match_candidate_ids,
        )


def _task_description(encoded_task: EncodedTask, task: TaskCreate) -> str:
    return (encoded_task.text or task.input.description or "").strip()


def merge_browser_classifications(
    frame_output: BrowserFrameAnalysisOutput,
    video_output: BrowserClassificationOutput | None,
    scene_windows: tuple[_SceneWindow, ...],
) -> BrowserClassificationOutput:
    """Expand authoritative scene decisions back to every routed keyframe."""

    scene_result_by_member_id: dict[str, BrowserClassification] = {}
    if video_output is not None:
        expected_ids = {candidate_id(window.candidate) for window in scene_windows}
        actual_ids = {result.candidate_id for result in video_output.results}
        if actual_ids != expected_ids or len(video_output.results) != len(scene_windows):
            raise ValueError(
                "Video browser classifications must cover exactly the screened "
                f"scene windows. Expected {sorted(expected_ids)!r}, got "
                f"{[result.candidate_id for result in video_output.results]!r}."
            )
        video_results_by_id = {
            result.candidate_id: result for result in video_output.results
        }
        for window in scene_windows:
            result = video_results_by_id[candidate_id(window.candidate)]
            for member_id in window.member_ids:
                scene_result_by_member_id[member_id] = result.model_copy(
                    update={"candidate_id": member_id}
                )

    image_calls = _calls_with_phase(frame_output.metadata, "frame_analysis")
    video_calls = (
        _calls_with_phase(video_output.metadata, "scene_analysis")
        if video_output is not None
        else []
    )
    return BrowserClassificationOutput(
        results=[
            scene_result_by_member_id.get(
                result.candidate_id,
                BrowserClassification(
                    candidate_id=result.candidate_id,
                    judgement=(
                        BrowserJudgement.RELEVANT
                        if result.decision == KeyframeDecision.INSPECT_SCENE
                        else BrowserJudgement.IRRELEVANT
                    ),
                    reasoning=result.reason,
                    matching_evidence=result.visible_support,
                    contradicting_evidence=result.visible_conflicts,
                ),
            )
            for result in frame_output.results
        ],
        metadata={
            "llm_calls": image_calls + video_calls,
            "frame_analysis": frame_output.metadata,
            "scene_analysis": (
                video_output.metadata if video_output is not None else None
            ),
            "scene_results": (
                video_output.metadata.get("scene_results", {})
                if video_output is not None
                else {}
            ),
            "image_candidate_count": len(frame_output.results),
            "video_candidate_count": (
                len(video_output.results) if video_output is not None else 0
            ),
            "cache_hits": [
                *frame_output.metadata.get("cache_hits", []),
                *(
                    video_output.metadata.get("cache_hits", [])
                    if video_output is not None
                    else []
                ),
            ],
            "deduplicated_candidate_count": int(
                frame_output.metadata.get("deduplicated_candidate_count", 0)
            )
            + (
                int(video_output.metadata.get("deduplicated_candidate_count", 0))
                if video_output is not None
                else 0
            ),
        },
    )


def merge_browser_descriptions(
    frame_output: BrowserDescriptionOutput,
    scene_output: BrowserDescriptionOutput,
) -> BrowserDescriptionOutput:
    """Combine keyframe and optional scene descriptions without grouping ids."""

    return BrowserDescriptionOutput(
        results=[*frame_output.results, *scene_output.results],
        metadata={
            "llm_calls": [
                *_calls_with_phase(frame_output.metadata, "frame"),
                *_calls_with_phase(scene_output.metadata, "scene"),
            ],
            "frame_descriptions": frame_output.metadata,
            "scene_descriptions": scene_output.metadata,
            "cache_hits": [
                *frame_output.metadata.get("cache_hits", []),
                *scene_output.metadata.get("cache_hits", []),
            ],
        },
    )


def frame_analysis_classifications(
    output: BrowserFrameAnalysisOutput,
) -> BrowserClassificationOutput:
    """Derive static-image categories from the authoritative frame analysis."""

    return BrowserClassificationOutput(
        results=[
            BrowserClassification(
                candidate_id=result.candidate_id,
                judgement=(
                    BrowserJudgement.RELEVANT
                    if result.decision == KeyframeDecision.INSPECT_SCENE
                    else BrowserJudgement.IRRELEVANT
                ),
                reasoning=result.reason,
                matching_evidence=result.visible_support,
                contradicting_evidence=result.visible_conflicts,
            )
            for result in output.results
        ],
        metadata=output.metadata,
    )


def frame_analysis_descriptions(
    output: BrowserFrameAnalysisOutput,
) -> BrowserDescriptionOutput:
    return BrowserDescriptionOutput(
        results=[
            BrowserDescription(
                candidate_id=result.candidate_id,
                media_kind="frame",
                description=result.frame_description,
            )
            for result in output.results
        ],
        metadata=output.metadata,
    )


def scene_analysis_classifications(
    output: BrowserSceneAnalysisOutput,
) -> BrowserClassificationOutput:
    """Convert authoritative scene verdicts into retrieval-loop categories."""

    return BrowserClassificationOutput(
        results=[
            BrowserClassification(
                candidate_id=result.candidate_id,
                judgement=(
                    BrowserJudgement.MATCH
                    if result.verdict == SceneVerdict.EXACT
                    else (
                        BrowserJudgement.RELEVANT
                        if result.verdict == SceneVerdict.RELEVANT
                        else BrowserJudgement.IRRELEVANT
                    )
                ),
                reasoning=result.reason,
                matching_evidence=result.matching_evidence,
                missing_evidence=result.missing_evidence,
                contradicting_evidence=result.contradicting_evidence,
            )
            for result in output.results
        ],
        metadata={
            **output.metadata,
            "scene_results": {
                result.candidate_id: result.model_dump(mode="json")
                for result in output.results
            },
        },
    )


def scene_analysis_descriptions(
    output: BrowserSceneAnalysisOutput,
) -> BrowserDescriptionOutput:
    return BrowserDescriptionOutput(
        results=[
            BrowserDescription(
                candidate_id=result.candidate_id,
                media_kind="scene",
                description=result.scene_description,
            )
            for result in output.results
        ],
        metadata=output.metadata,
    )


def expand_scene_descriptions(
    descriptions: BrowserDescriptionOutput,
    scene_windows: tuple[_SceneWindow, ...],
) -> BrowserDescriptionOutput:
    """Attach one merged-window scene description to all of its keyframes."""

    descriptions_by_id = {
        description.candidate_id: description for description in descriptions.results
    }
    expanded: list[BrowserDescription] = []
    for window in scene_windows:
        description = descriptions_by_id.get(candidate_id(window.candidate))
        if description is None:
            continue
        expanded.extend(
            description.model_copy(update={"candidate_id": member_id})
            for member_id in window.member_ids
        )
    return BrowserDescriptionOutput(results=expanded, metadata=descriptions.metadata)


def exact_scene_candidates(
    output: BrowserClassificationOutput | None,
    scene_windows: tuple[_SceneWindow, ...],
) -> tuple[Candidate, ...]:
    """Return verifier candidates narrowed to each browser-reported exact interval."""

    if output is None:
        return ()
    exact_ids = set(output.exact_candidate_ids)
    analyses = output.metadata.get("scene_results", {})
    candidates: list[Candidate] = []
    for window in scene_windows:
        identifier = candidate_id(window.candidate)
        if identifier not in exact_ids:
            continue
        interval = analyses.get(identifier) if isinstance(analyses, dict) else None
        start_offset = (
            interval.get("best_match_start_seconds")
            if isinstance(interval, dict)
            else None
        )
        end_offset = (
            interval.get("best_match_end_seconds")
            if isinstance(interval, dict)
            else None
        )
        if not isinstance(start_offset, (int, float)) or not isinstance(
            end_offset, (int, float)
        ):
            start_ms, end_ms = window.start_ms, window.end_ms
        else:
            start_ms = max(
                window.start_ms,
                min(window.end_ms, window.start_ms + round(start_offset * 1000)),
            )
            end_ms = max(
                window.start_ms,
                min(window.end_ms, window.start_ms + round(end_offset * 1000)),
            )
            if end_ms <= start_ms:
                start_ms, end_ms = window.start_ms, window.end_ms
        candidates.append(
            window.candidate.model_copy(
                update={
                    "time_keyframe": start_ms + (end_ms - start_ms) // 2,
                    "time_scene_start": start_ms,
                    "time_scene_end": end_ms,
                }
            )
        )
    return tuple(candidates)


def _calls_with_phase(
    metadata: dict[str, object],
    phase: str,
) -> list[dict[str, object]]:
    calls = metadata.get("llm_calls", [])
    if not isinstance(calls, list):
        return []
    return [{**call, "phase": phase} for call in calls if isinstance(call, dict)]


def _classification_feedback(result: BrowserClassification) -> str | None:
    """Preserve the scene classifier's concrete match and mismatch evidence."""

    parts = [result.reasoning] if result.reasoning else []
    for label, evidence in (
        ("Matches", result.matching_evidence),
        ("Missing", result.missing_evidence),
        ("Contradictions", result.contradicting_evidence),
    ):
        if evidence:
            parts.append(f"{label}: {'; '.join(evidence)}.")
    return " ".join(parts) or None


def _candidate_video_identifier(candidate: Candidate) -> str:
    return candidate.id[0] if candidate.id else candidate_id(candidate)


def _candidate_file_size(candidate: BrowserCandidate) -> int:
    media = candidate.video or candidate.image
    if media is None and candidate.images:
        return sum(_safe_file_size(image) for image in candidate.images)
    if media is None:
        return 0
    return _safe_file_size(media)


def _safe_file_size(media: object) -> int:
    try:
        return media.file.stat().st_size
    except (AttributeError, OSError):
        return 0
