import { useEffect, useState } from 'react';
import type { ChangeEvent } from 'react';

import { LazyJsonDetails } from './LazyJsonDetails';
import { LoopVisualization } from './retrieval/LoopVisualization';
import {
  candidateImageUrl,
  candidateKey,
  submittedFrameReference,
  submissionIdentityKey,
  submissionFrameUrl,
  submissionFromPlannerAction,
  uniqueSubmissions,
} from '../utils/retrievalLoop';
import type {
  AnswererOutput,
  AnswererValidation,
  CandidateSubmission,
  RetrievalLoopResult,
} from '../types/retrievalLoop';

type LoadStatus = "idle" | "loading" | "success" | "error";

type TaskRunLogSummary = {
  log_id: string;
  run_id: string;
  task_name: string;
  task_type: string;
  dataset: string;
  status: string;
  source_file?: string | null;
  max_iterations: number;
  max_candidates: number;
  hide_avs_submission_ratings_from_planner?: boolean;
  iteration_count: number;
  final_candidate_id?: string | null;
  final_answer?: string | null;
  duration_ms?: number | null;
  error?: string | null;
};

type TaskRunLog = TaskRunLogSummary & {
  task_id?: string;
  payload?: {
    task_type?: string;
    dataset?: string;
    source_task_name?: string | null;
    input?: {
      description?: string | null;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  } | null;
  target_video?: Record<string, unknown> | null;
  task_input_descriptions?: Record<string, string>;
  final_submission?: CandidateSubmission | null;
  final_answerer_output?: AnswererOutput | null;
  final_answerer_validation?: AnswererValidation | null;
  actions?: TaskRunAction[];
  result?: RetrievalLoopResult | null;
  traceback?: string | null;
  [key: string]: unknown;
};

type TaskRunAction = {
  iteration?: number;
  index?: number;
  action?: string;
  endpoint?: string;
  candidate_ids?: string[];
  retrieval_error?: string | null;
  submissions?: CandidateSubmission[];
  answerer_output?: AnswererOutput | null;
  answerer_validation?: AnswererValidation | null;
  answerer_error?: string | null;
  timings_ms?: Record<string, number>;
  [key: string]: unknown;
};

const formatDuration = (durationMs?: number) => {
  if (durationMs == null) return "Unknown";
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  return `${(durationMs / 1000).toFixed(1)} s`;
};

const statusClass = (status: string) => {
  if (status === "found" || status === "found via answerer") {
    return "task-log-status task-log-status--found";
  }
  if (status === "error") return "task-log-status task-log-status--error";
  return "task-log-status";
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isRetrievalLoopResult = (value: unknown): value is RetrievalLoopResult =>
  isRecord(value) &&
  typeof value.status === "string" &&
  Array.isArray(value.iterations);

const asString = (value: unknown, fallback: string) =>
  typeof value === "string" && value.trim() ? value : fallback;

const asNullableString = (value: unknown) =>
  typeof value === "string" && value.trim() ? value : null;

const asNumber = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const asActionList = (value: unknown): TaskRunAction[] =>
  Array.isArray(value) ? value.filter(isRecord) : [];

const fileLogId = (fileName: string) =>
  fileName.replace(/\.json$/i, "").replace(/[^\w.-]+/g, "-") || "uploaded-log";

const actionIterationIndex = (action: TaskRunAction, fallback: number) =>
  asNumber(action.iteration ?? action.index, fallback);

const actionToIteration = (
  action: TaskRunAction,
  fallbackIndex: number,
): RetrievalLoopResult["iterations"][number] => {
  const candidateIds = Array.isArray(action.candidate_ids)
    ? action.candidate_ids
    : [];
  const actionSubmissions = Array.isArray(action.submissions)
    ? action.submissions
    : [];
  const plannerSubmission =
    actionSubmissions.length === 0
      ? submissionFromPlannerAction(action.action)
      : null;

  return {
    index: actionIterationIndex(action, fallbackIndex),
    action: asString(action.action, "Logged action"),
    endpoint: asString(action.endpoint, "loggedAction"),
    candidate_ids: candidateIds,
    candidates: [],
    filtered_candidates: [],
    retrieval_error: action.retrieval_error ?? null,
    submissions: plannerSubmission ? [plannerSubmission] : actionSubmissions,
    answerer_output: isRecord(action.answerer_output)
      ? (action.answerer_output as AnswererOutput)
      : null,
    answerer_validation: isRecord(action.answerer_validation)
      ? (action.answerer_validation as AnswererValidation)
      : null,
    answerer_error: asNullableString(action.answerer_error),
    timings_ms: isRecord(action.timings_ms)
      ? (action.timings_ms as Record<string, number>)
      : undefined,
  };
};

const mergeActionsIntoResult = (
  result: RetrievalLoopResult,
  actions: TaskRunAction[],
): RetrievalLoopResult => {
  if (!actions.length) return result;

  const actionsByIndex = new Map(
    actions.map((action, index) => [actionIterationIndex(action, index + 1), action]),
  );
  const mergedIterations = result.iterations.map((iteration) => {
    const action = actionsByIndex.get(iteration.index);
    if (!action) return iteration;
    const actionSubmissions = Array.isArray(action.submissions)
      ? action.submissions
      : [];
    const plannerSubmission =
      actionSubmissions.length === 0
        ? submissionFromPlannerAction(iteration.action)
        : null;

    return {
      ...iteration,
      submissions:
        iteration.submissions?.length
          ? iteration.submissions
          : plannerSubmission
            ? [plannerSubmission]
            : actionSubmissions,
      answerer_output:
        iteration.answerer_output ??
        (isRecord(action.answerer_output)
          ? (action.answerer_output as AnswererOutput)
          : null),
      answerer_validation:
        iteration.answerer_validation ??
        (isRecord(action.answerer_validation)
          ? (action.answerer_validation as AnswererValidation)
          : null),
      answerer_error:
        iteration.answerer_error ?? asNullableString(action.answerer_error),
    };
  });

  const displayedIndexes = new Set(mergedIterations.map((iteration) => iteration.index));
  const actionOnlyIterations = actions
    .map(actionToIteration)
    .filter((iteration) => !displayedIndexes.has(iteration.index));

  return {
    ...result,
    iterations: [...mergedIterations, ...actionOnlyIterations].sort(
      (left, right) => left.index - right.index,
    ),
  };
};

const resultFromActions = (
  log: Pick<
    TaskRunLog,
    | "status"
    | "actions"
    | "final_candidate_id"
    | "final_answer"
    | "final_submission"
    | "duration_ms"
  >,
): RetrievalLoopResult | null => {
  const actions = log.actions ?? [];
  if (!actions.length) return null;

  return {
    status: log.status,
    final_candidate_id: log.final_candidate_id,
    final_answer: log.final_answer,
    final_submission: log.final_submission,
    timings_ms:
      log.duration_ms == null ? undefined : { total: Number(log.duration_ms) },
    iterations: actions.map(actionToIteration),
  };
};

const normalizeTaskRunLog = (
  value: unknown,
  fallbackFileName = "uploaded-log.json",
): TaskRunLog => {
  if (isRetrievalLoopResult(value)) {
    const logId = fileLogId(fallbackFileName);
    return {
      log_id: logId,
      run_id: logId,
      task_name: logId,
      task_type: "unknown",
      dataset: "unknown",
      status: value.status,
      source_file: fallbackFileName,
      max_iterations: value.iterations.length,
      max_candidates: 0,
      iteration_count: value.iterations.length,
      final_candidate_id: value.final_candidate_id,
      final_answer: value.final_answer,
      result: value,
    };
  }

  if (!isRecord(value)) {
    throw new Error("Selected JSON is not a task execution log.");
  }

  const result = isRetrievalLoopResult(value.result) ? value.result : null;
  const actions = asActionList(value.actions);
  const payload = isRecord(value.payload) ? value.payload : {};
  const logId = asString(value.log_id, fileLogId(fallbackFileName));
  const taskType = asString(value.task_type, asString(payload.task_type, "unknown"));
  const dataset = asString(value.dataset, asString(payload.dataset, "unknown"));

  return {
    ...value,
    log_id: logId,
    run_id: asString(value.run_id, logId),
    task_name: asString(
      value.task_name,
      asString(payload.source_task_name, logId),
    ),
    task_type: taskType,
    dataset,
    status: asString(value.status, result?.status ?? "unknown"),
    source_file: asNullableString(value.source_file) ?? fallbackFileName,
    max_iterations: asNumber(
      value.max_iterations,
      result?.iterations.length ?? actions.length,
    ),
    max_candidates: asNumber(value.max_candidates, 0),
    iteration_count: asNumber(
      value.iteration_count,
      result?.iterations.length ?? actions.length,
    ),
    final_candidate_id:
      asNullableString(value.final_candidate_id) ?? result?.final_candidate_id,
    final_answer: asNullableString(value.final_answer) ?? result?.final_answer,
    duration_ms:
      typeof value.duration_ms === "number" ? value.duration_ms : undefined,
    error: asNullableString(value.error),
    task_input_descriptions: isRecord(value.task_input_descriptions)
      ? Object.fromEntries(
          Object.entries(value.task_input_descriptions).filter(
            (entry): entry is [string, string] => typeof entry[1] === "string",
          ),
        )
      : {},
    final_submission: isRecord(value.final_submission)
      ? (value.final_submission as CandidateSubmission)
      : result?.final_submission ?? null,
    final_answerer_output: isRecord(value.final_answerer_output)
      ? (value.final_answerer_output as AnswererOutput)
      : null,
    final_answerer_validation: isRecord(value.final_answerer_validation)
      ? (value.final_answerer_validation as AnswererValidation)
      : null,
    actions,
    result: result
      ? mergeActionsIntoResult(result, actions)
      : resultFromActions({
          status: asString(value.status, "unknown"),
          actions,
          final_candidate_id: asNullableString(value.final_candidate_id),
          final_answer: asNullableString(value.final_answer),
          final_submission: isRecord(value.final_submission)
            ? (value.final_submission as CandidateSubmission)
            : null,
          duration_ms:
            typeof value.duration_ms === "number" ? value.duration_ms : undefined,
        }),
  };
};

const toSummary = (log: TaskRunLog): TaskRunLogSummary => ({
  log_id: log.log_id,
  run_id: log.run_id,
  task_name: log.task_name,
  task_type: log.task_type,
  dataset: log.dataset,
  status: log.status,
  source_file: log.source_file,
  max_iterations: log.max_iterations,
  max_candidates: log.max_candidates,
  hide_avs_submission_ratings_from_planner:
    log.hide_avs_submission_ratings_from_planner,
  iteration_count: log.iteration_count,
  final_candidate_id: log.final_candidate_id,
  final_answer: log.final_answer,
  duration_ms: log.duration_ms,
  error: log.error,
});

type SubmittedItem = {
  iteration: number | null;
  submission: CandidateSubmission;
  label?: string;
};

type AnswerScene = {
  videoId: string;
  startMs: number;
  endMs: number;
  source: string;
};

const submittedItemFingerprint = (item: SubmittedItem) => {
  return submissionIdentityKey(item.submission);
};

const submittedItems = (log: TaskRunLog): SubmittedItem[] => {
  const iterationSubmissions: SubmittedItem[] = (log.result?.iterations ?? [])
    .flatMap((iteration) => {
      const actionSubmission = submissionFromPlannerAction(iteration.action);
      return uniqueSubmissions([
        ...(iteration.submissions ?? []),
        ...(actionSubmission ? [actionSubmission] : []),
      ]).map((submission) => ({
        iteration: iteration.index,
        submission,
        label: "Planner submission",
      }));
    });

  const finalItems: SubmittedItem[] = [];
  if (log.final_submission) {
    finalItems.push({
      iteration: null,
      submission: log.final_submission,
      label: "Final submission",
    });
  }
  const seen = new Set<string>();
  return [...iterationSubmissions, ...finalItems]
    .sort((left, right) => {
      if (left.iteration == null && right.iteration != null) return 1;
      if (left.iteration != null && right.iteration == null) return -1;
      return 0;
    })
    .filter((item) => {
      const key = submittedItemFingerprint(item);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
};

const submissionStateLabel = (submission: CandidateSubmission) => {
  if (submission.status) return submission.status;
  if (submission.source === "planner") return "submitted";
  return submission.accepted ? "accepted" : "rejected";
};

const numberField = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const answerSceneFromValue = (
  value: unknown,
  source: string,
): AnswerScene | null => {
  if (!isRecord(value)) return null;
  const rawName = value.name ?? value.video_id ?? value.videoId;
  const videoId = typeof rawName === "string" ? rawName.trim() : "";
  const startMs = numberField(value.start_ms ?? value.startMs);
  const endMs = numberField(value.end_ms ?? value.endMs);
  if (!videoId || startMs == null || endMs == null || endMs <= startMs) {
    return null;
  }

  return { videoId, startMs, endMs, source };
};

const answerScene = (log: TaskRunLog): AnswerScene | null =>
  answerSceneFromValue(log.target_video, "target video") ??
  answerSceneFromValue(log.payload?.input?.answer_video, "answer video");

const answerSceneUrl = (scene: AnswerScene, dataset: string) => {
  const params = new URLSearchParams({
    dataset,
    video_id: scene.videoId,
    start_ms: String(scene.startMs),
    end_ms: String(scene.endMs),
  });
  return `/api/tasks/answer-scene?${params.toString()}`;
};

const answerSceneFrameUrl = (scene: AnswerScene, dataset: string) => {
  const params = new URLSearchParams({
    dataset,
    video_id: scene.videoId,
    timestamp_ms: String(Math.round((scene.startMs + scene.endMs) / 2)),
  });
  return `/api/tasks/submission-frame?${params.toString()}`;
};

const formatSceneTime = (timestampMs: number) => {
  const totalSeconds = timestampMs / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  if (minutes === 0) return `${seconds.toFixed(2)} s`;
  return `${minutes}:${seconds.toFixed(2).padStart(5, "0")}`;
};

const scrollToStep = (iteration: number) => {
  const target = document.getElementById(`task-log-step-${iteration}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  target.focus({ preventScroll: true });
};

type FailureItem = {
  title: string;
  message: string;
  details?: string;
};

const failureItems = (log: TaskRunLog): FailureItem[] => {
  const failures: FailureItem[] = [];
  const hasIterations = Boolean(log.result?.iterations?.length);
  if (hasIterations) return failures;

  if (log.error) {
    failures.push({
      title: log.status === "error" ? "Startup failure" : "Run failure",
      message: log.error,
      details: typeof log.traceback === "string" ? log.traceback : undefined,
    });
  } else if (log.status === "error") {
    failures.push({
      title: "Startup failure",
      message: "The task failed before a loop result or detailed error was written.",
    });
  }

  const seen = new Set<string>();
  return failures.filter((failure) => {
    const key = `${failure.title}|${failure.message}|${failure.details ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const responseErrorMessage = async (response: Response, fallback: string) => {
  const body = await response.json().catch(() => null);
  if (!isRecord(body)) return `${fallback} (${response.status})`;
  const detail = body.detail ?? body.error;
  if (typeof detail === "string" && detail.trim()) {
    return `${detail} (${response.status})`;
  }
  return `${fallback} (${response.status})`;
};

const submissionPreviewUrl = (
  log: TaskRunLog,
  item: SubmittedItem,
) => {
  if (submittedFrameReference(item.submission)) {
    return submissionFrameUrl(item.submission, log.dataset);
  }

  if (item.iteration != null) {
    const iteration = log.result?.iterations.find(
      (candidateIteration) => candidateIteration.index === item.iteration,
    );
    const candidate = [
      ...(iteration?.candidates ?? []),
      ...(iteration?.filtered_candidates ?? []),
    ].find((nextCandidate) => candidateKey(nextCandidate) === item.submission.candidate_id);
    if (candidate) return candidateImageUrl(candidate, log.dataset);
  }

  return submissionFrameUrl(item.submission, log.dataset);
};

const LoadableImage = ({
  alt,
  fallback,
  src,
}: {
  alt: string;
  fallback: string;
  src: string;
}) => {
  const [failedSrc, setFailedSrc] = useState("");
  const failed = failedSrc === src;

  if (!src || failed) {
    return <span className="task-run-log-media-fallback">{fallback}</span>;
  }

  return <img alt={alt} onError={() => setFailedSrc(src)} src={src} />;
};

const AnswerScenePreview = ({
  dataset,
  scene,
}: {
  dataset: string;
  scene: AnswerScene;
}) => {
  const [videoSrc, setVideoSrc] = useState("");
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const src = answerSceneUrl(scene, dataset);

  useEffect(() => {
    const abortController = new AbortController();
    let objectUrl = "";

    const loadVideo = async () => {
      setLoadStatus("loading");
      setErrorMessage("");
      setVideoSrc("");

      try {
        const response = await fetch(src, { signal: abortController.signal });
        if (!response.ok) {
          throw new Error(
            await responseErrorMessage(
              response,
              "Could not load the correct answer scene.",
            ),
          );
        }

        objectUrl = URL.createObjectURL(await response.blob());
        setVideoSrc(objectUrl);
        setLoadStatus("success");
      } catch (error) {
        if (abortController.signal.aborted) return;
        setLoadStatus("error");
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "Could not load the correct answer scene.",
        );
      }
    };

    loadVideo();

    return () => {
      abortController.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  if (loadStatus === "success" && videoSrc) {
    return <video controls preload="metadata" src={videoSrc} />;
  }

  return (
    <div className="task-run-log-answer-scene__fallback">
      {loadStatus === "loading" ? (
        <p>Loading correct answer scene...</p>
      ) : (
        <p className="task-run-log-answer-scene__error">{errorMessage}</p>
      )}
      {loadStatus === "error" ? (
        <LoadableImage
          alt={`Correct answer scene frame ${scene.videoId}`}
          fallback="Correct answer frame unavailable"
          src={answerSceneFrameUrl(scene, dataset)}
        />
      ) : null}
    </div>
  );
};

const effectiveTaskDescriptions = (log: TaskRunLog) => {
  const modulesByDescription = new Map<string, string[]>();

  Object.entries(log.task_input_descriptions ?? {}).forEach(([module, value]) => {
    const description = value.trim();
    if (!description) return;
    modulesByDescription.set(description, [
      ...(modulesByDescription.get(description) ?? []),
      module,
    ]);
  });

  const descriptions = Array.from(
    modulesByDescription,
    ([text, modules]) => ({ modules, text }),
  );
  if (descriptions.length) return descriptions;

  const encodedText = log.result?.encoded_task?.text?.trim();
  if (!encodedText) return [];

  const encodedDescription = encodedText
    .split(/\n{2,}(?=[^\n]+:\s*\n)/)
    .filter((section) => !/^Task question:\s*/i.test(section))
    .join("\n\n")
    .trim();

  return encodedDescription
    ? [{ modules: ["encoded task (legacy log)"], text: encodedDescription }]
    : [];
};

export const TaskRunLogs = () => {
  const [summaries, setSummaries] = useState<TaskRunLogSummary[]>([]);
  const [selectedLogId, setSelectedLogId] = useState("");
  const [selectedLog, setSelectedLog] = useState<TaskRunLog | null>(null);
  const [uploadedLogs, setUploadedLogs] = useState<Record<string, TaskRunLog>>({});
  const [status, setStatus] = useState<LoadStatus>("idle");
  const [message, setMessage] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const taskDescriptions = selectedLog
    ? effectiveTaskDescriptions(selectedLog)
    : [];
  const submissions = selectedLog ? submittedItems(selectedLog) : [];
  const failures = selectedLog ? failureItems(selectedLog) : [];
  const correctAnswerScene = selectedLog ? answerScene(selectedLog) : null;

  const loadJsonFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploadedFileName(file.name);
    setStatus("loading");
    setMessage("");

    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const normalizedLog = normalizeTaskRunLog(parsed, file.name);

      setSelectedLog(normalizedLog);
      setSelectedLogId(normalizedLog.log_id);
      setUploadedLogs((current) => ({
        ...current,
        [normalizedLog.log_id]: normalizedLog,
      }));
      setSummaries((current) =>
        current.some((summary) => summary.log_id === normalizedLog.log_id)
          ? current
          : [toSummary(normalizedLog), ...current],
      );
      setStatus("success");
      setMessage(`Loaded ${file.name}`);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not load JSON log.");
    }
  };

  const loadSummaries = async () => {
    setStatus("loading");
    setMessage("");

    try {
      const response = await fetch("/api/tasks/logs");
      if (!response.ok) {
        throw new Error(
          await responseErrorMessage(response, "Could not load task run logs."),
        );
      }

      const nextSummaries = (await response.json()) as TaskRunLogSummary[];
      setSummaries(nextSummaries);
      setSelectedLogId((current) =>
        current && nextSummaries.some((summary) => summary.log_id === current)
          ? current
          : "",
      );
      setStatus("success");
      setMessage(nextSummaries.length ? "" : "No task logs are available yet.");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not load task run logs.");
    }
  };

  useEffect(() => {
    loadSummaries();
  }, []);

  useEffect(() => {
    if (!selectedLogId) {
      setSelectedLog(null);
      return;
    }

    const abortController = new AbortController();

    const loadLog = async () => {
      const uploadedLog = uploadedLogs[selectedLogId];
      if (uploadedLog) {
        setSelectedLog(uploadedLog);
        setStatus("success");
        return;
      }

      setStatus("loading");
      setMessage("");

      try {
        const response = await fetch(`/api/tasks/logs/${selectedLogId}`, {
          signal: abortController.signal,
        });
        if (!response.ok) {
          throw new Error(
            await responseErrorMessage(
              response,
              "Could not load the selected task run log.",
            ),
          );
        }

        setSelectedLog(
          normalizeTaskRunLog(await response.json(), `${selectedLogId}.json`),
        );
        setStatus("success");
      } catch (error) {
        if (!abortController.signal.aborted) {
          setStatus("error");
          setMessage(
            error instanceof Error
              ? error.message
              : "Could not load the selected task run log.",
          );
        }
      }
    };

    loadLog();

    return () => abortController.abort();
  }, [selectedLogId, uploadedLogs]);

  return (
    <section className="debug-panel task-run-log-panel" aria-label="Imported task run logs">
      <div className="task-run-log-panel__header">
        <div>
          <h2>Imported task logs</h2>
          <p>Load one task execution to inspect loop actions, submissions, and errors.</p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={status === "loading"}
          onClick={loadSummaries}
        >
          {status === "loading" ? "Loading..." : "Refresh logs"}
        </button>
      </div>

      <div className="task-run-log-controls">
        <label className="form-field">
          <span>Task log</span>
          <select
            value={selectedLogId}
            onChange={(event) => setSelectedLogId(event.target.value)}
          >
            <option value="">Select a task log...</option>
            {summaries.map((summary) => (
              <option key={summary.log_id} value={summary.log_id}>
                {summary.task_name} · {summary.status} · {summary.run_id}
              </option>
            ))}
          </select>
        </label>

        <div className="form-field">
          <span>JSON file</span>
          <label className="task-log-upload">
            <input
              className="sr-only"
              type="file"
              accept="application/json,.json"
              onChange={loadJsonFile}
            />
            <span className="task-log-upload__button">Choose JSON file</span>
            <span className="task-log-upload__name">
              {uploadedFileName || "No file selected"}
            </span>
          </label>
        </div>

        {selectedLog ? (
          <dl className="stored-representation__meta">
            <div>
              <dt>Status</dt>
              <dd>{selectedLog.status}</dd>
            </div>
            <div>
              <dt>Iterations</dt>
              <dd>{selectedLog.iteration_count}</dd>
            </div>
            <div>
              <dt>Run</dt>
              <dd>{selectedLog.run_id}</dd>
            </div>
          </dl>
        ) : null}
      </div>

      {message ? (
        <p className={`status-message status-message--${status}`}>{message}</p>
      ) : null}

      {selectedLog ? (
        <article className="task-run-log-card task-run-log-card--summary">
          <header>
            <div>
              <span className="summary-kicker">
                {selectedLog.task_type.toUpperCase()} · {selectedLog.dataset.toUpperCase()}
              </span>
              <h3>{selectedLog.task_name}</h3>
            </div>
            <span className={statusClass(selectedLog.status)}>{selectedLog.status}</span>
          </header>

          <dl className="stored-representation__meta">
            <div>
              <dt>Iterations</dt>
              <dd>{selectedLog.iteration_count ?? 0}</dd>
            </div>
            <div>
              <dt>Final candidate</dt>
              <dd>{selectedLog.final_candidate_id ?? "None"}</dd>
            </div>
            {selectedLog.final_answer ? (
              <div>
                <dt>Final answer</dt>
                <dd>{selectedLog.final_answer}</dd>
              </div>
            ) : null}
            {selectedLog.final_answerer_output ? (
              <div>
                <dt>Answerer fallback</dt>
                <dd>
                  {selectedLog.final_answerer_output.answer ??
                    `${selectedLog.final_answerer_output.video_id}, ${selectedLog.final_answerer_output.timestamp_ms}`}
                </dd>
              </div>
            ) : null}
            <div>
              <dt>Duration</dt>
              <dd>{formatDuration(selectedLog.duration_ms ?? undefined)}</dd>
            </div>
            <div>
              <dt>Log id</dt>
              <dd>{selectedLog.log_id}</dd>
            </div>
          </dl>

          {failures.length ? (
            <section className="task-run-log-failures">
              <h4>Failure details</h4>
              <ol>
                {failures.map((failure, index) => (
                  <li key={`${failure.title}-${index}`}>
                    <strong>{failure.title}</strong>
                    <p>{failure.message}</p>
                    {failure.details ? (
                      <details>
                        <summary>Show exception details</summary>
                        <pre>{failure.details}</pre>
                      </details>
                    ) : null}
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          {correctAnswerScene ? (
            <section className="task-run-log-answer-scene">
              <div>
                <h4>Correct answer scene</h4>
                <p>
                  {correctAnswerScene.videoId} ·{" "}
                  {formatSceneTime(correctAnswerScene.startMs)} -{" "}
                  {formatSceneTime(correctAnswerScene.endMs)} ·{" "}
                  {correctAnswerScene.source}
                </p>
              </div>
              <AnswerScenePreview
                dataset={selectedLog.dataset}
                scene={correctAnswerScene}
              />
            </section>
          ) : null}

          {submissions.length ? (
            <section className="task-run-log-submissions">
              <h4>Submitted answers</h4>
              <ol>
                {submissions.map((item, index) => {
                  const { iteration, submission } = item;
                  const frameUrl = submissionPreviewUrl(selectedLog, item);
                  const sourceLabel = item.label ?? submission.source;
                  const content = (
                    <>
                      {frameUrl ? (
                        <LoadableImage
                          alt={`Submitted answer ${submission.candidate_id}`}
                          fallback="Frame unavailable"
                          src={frameUrl}
                        />
                      ) : null}
                      <strong>
                        {submission.answer?.trim() ||
                          submission.candidate_id ||
                          `${submission.video_id ?? "video"}, ${
                            submission.timestamp_ms ?? "time"
                          }`}
                      </strong>
                      <span>
                        {iteration == null ? sourceLabel : `Step ${iteration}`} ·{" "}
                        {submissionStateLabel(submission)}
                        {submission.source ? ` · ${submission.source}` : ""}
                      </span>
                      {submission.reason ? (
                        <small className="task-run-log-submission-reason">
                          {submission.reason}
                        </small>
                      ) : null}
                    </>
                  );

                  return (
                    <li
                      className={
                        frameUrl ? "task-run-log-submission--with-image" : undefined
                      }
                      key={`${iteration ?? "final"}-${submission.candidate_id}-${submission.answer ?? index}`}
                    >
                      {iteration == null ? (
                        <div className="task-run-log-submission-content">
                          {content}
                        </div>
                      ) : (
                        <button
                          className="task-run-log-submission-content task-run-log-submission-button"
                          onClick={() => scrollToStep(iteration)}
                          type="button"
                        >
                          {content}
                        </button>
                      )}
                    </li>
                  );
                })}
              </ol>
            </section>
          ) : null}

          {taskDescriptions.length ? (
            <div className="stored-task-descriptions">
              <h4>VLM task/search description</h4>
              {taskDescriptions.map((description) => (
                <section
                  className="stored-task-description"
                  key={`${description.modules.join("-")}-${description.text}`}
                >
                  <h4>{description.modules.join(", ")}</h4>
                  <p>{description.text}</p>
                </section>
              ))}
            </div>
          ) : null}
        </article>
      ) : null}

      {selectedLog?.result ? (
        <LoopVisualization
          allowContinue={false}
          dataset={selectedLog.dataset}
          isContinuing={false}
          onContinue={() => undefined}
          result={selectedLog.result}
        />
      ) : selectedLog ? (
        <p className="input-panel__empty">
          {selectedLog.status === "error"
            ? "This task failed before a loop result could be written. See failure details above."
            : "This task log does not contain a completed loop result yet."}
        </p>
      ) : null}

      {selectedLog ? (
        <LazyJsonDetails summary="Show full run log JSON" value={selectedLog} />
      ) : null}
    </section>
  );
};
