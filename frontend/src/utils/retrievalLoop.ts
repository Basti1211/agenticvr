import type {
  Candidate,
  CandidateFrame,
  CandidateSubmission,
  HistoryEntry,
  RetrievalLoopIteration,
  RetrievalLoopResult,
} from "../types/retrievalLoop";

const ACHERON_IMAGE_BASE = "http://acheron.ms.mff.cuni.cz:42032/images";
const ACTION_CODES = new Set(["nl", "lnl", "t", "im", "b", "lvs", "iv", "lm"]);

export const candidateKey = (candidate: Candidate) => candidate.id.join("_");

export const candidateFrames = (candidate: Candidate): CandidateFrame[] =>
  candidate.frames?.length ? candidate.frames : [candidate];

const datasetPath = (datasetValue: string) =>
  datasetValue.toLowerCase() === "lhe" ? "VBSLHE" : datasetValue.toUpperCase();

const normalizeMediaUriPath = (uri: string) => {
  const cleanUri = uri.replace(/^\/+/, "");
  const [firstPart, ...rest] = cleanUri.split("/");
  if (firstPart?.toLowerCase() === "lhe") {
    return ["VBSLHE", ...rest].join("/");
  }
  return cleanUri;
};

export type SubmittedFrameReference = {
  videoId: string;
  timestampMs: number;
};

const directAnswerFrameReference = (
  candidate: CandidateFrame,
): SubmittedFrameReference | null => {
  const videoId = candidate.id[0]?.trim();
  const frameId = candidate.id[1]?.trim();
  const timestampMs = Number(frameId);
  if (!videoId || !Number.isFinite(timestampMs) || timestampMs < 10_000) {
    return null;
  }

  const looksLikeDirectAnswer =
    candidate.uri.includes("/direct-answer-frames/") ||
    candidate.uri.includes("\\direct-answer-frames\\") ||
    candidate.time_keyframe === timestampMs;
  if (!looksLikeDirectAnswer) return null;

  return { videoId, timestampMs };
};

const submittedFrameUrl = (
  frame: SubmittedFrameReference,
  datasetValue: string,
) => {
  const params = new URLSearchParams({
    dataset: datasetValue,
    video_id: frame.videoId,
    timestamp_ms: String(frame.timestampMs),
  });
  return `/api/tasks/submission-frame?${params.toString()}`;
};

export const candidateImageUrl = (
  candidate: CandidateFrame,
  datasetValue: string,
) => {
  const directAnswerFrame = directAnswerFrameReference(candidate);
  if (directAnswerFrame) return submittedFrameUrl(directAnswerFrame, datasetValue);

  if (/^https?:\/\//i.test(candidate.uri)) return candidate.uri;
  if (candidate.uri.startsWith("/")) return candidate.uri;

  const cleanUri = normalizeMediaUriPath(candidate.uri.replace(/^file:\/\//, ""));
  if (/\.(jpe?g|png|webp)$/i.test(cleanUri) && !cleanUri.startsWith("tmp/")) {
    const firstPart = cleanUri.split("/")[0]?.toLowerCase();
    const datasetPart = datasetPath(datasetValue).toLowerCase();
    const path =
      firstPart === datasetPart
        ? cleanUri
        : `${datasetPath(datasetValue)}/${cleanUri}`;
    return `${ACHERON_IMAGE_BASE}/${path}`;
  }

  if (candidate.id.length >= 2) {
    const frameId = candidate.id[1];
    const frameFile = /\.(jpe?g|png|webp)$/i.test(frameId)
      ? frameId
      : `${frameId}.jpg`;
    return `${ACHERON_IMAGE_BASE}/${datasetPath(datasetValue)}/${candidate.id[0]}/${frameFile}`;
  }

  return "";
};

export const candidateImageUrls = (candidate: Candidate, datasetValue: string) =>
  candidateFrames(candidate)
    .map((frame) => candidateImageUrl(frame, datasetValue))
    .filter(Boolean);

const parseSubmittedFrameReference = (
  value?: string | null,
): SubmittedFrameReference | null => {
  const trimmed = value?.trim();
  if (!trimmed) return null;

  const match = trimmed.match(/^(?:AN\s+)?([^,\s]+)\s*(?:,|\s)\s*(\d+)$/i);
  if (!match) return null;

  return {
    videoId: match[1],
    timestampMs: Number(match[2]),
  };
};

export const submissionFromPlannerAction = (
  action?: string | null,
): CandidateSubmission | null => {
  const frame = parseSubmittedFrameReference(action);
  if (!frame) return null;

  return {
    candidate_id: `${frame.videoId}_${frame.timestampMs}`,
    answer: action?.trim() ?? `AN ${frame.videoId} ${frame.timestampMs}`,
    accepted: false,
    status: null,
    reason: null,
    source: "planner",
    video_id: frame.videoId,
    timestamp_ms: frame.timestampMs,
  };
};

const submissionVideoId = (submission: CandidateSubmission) => {
  if (submission.video_id?.trim()) return submission.video_id.trim();
  const parsedAnswer = parseSubmittedFrameReference(submission.answer);
  if (parsedAnswer) return parsedAnswer.videoId;
  if (!submission.candidate_id) return "";
  const timestamp = submission.timestamp_ms;
  const suffix = timestamp == null ? "" : `_${timestamp}`;
  if (suffix && submission.candidate_id.endsWith(suffix)) {
    return submission.candidate_id.slice(0, -suffix.length);
  }

  const separatorIndex = submission.candidate_id.lastIndexOf("_");
  return separatorIndex > 0
    ? submission.candidate_id.slice(0, separatorIndex)
    : submission.candidate_id;
};

const directFrameFromCandidateId = (
  candidateId?: string | null,
): SubmittedFrameReference | null => {
  const match = candidateId?.trim().match(/^([^_\s]+)_(\d+)$/);
  if (!match) return null;

  const timestampMs = Number(match[2]);
  if (!Number.isFinite(timestampMs) || timestampMs < 10_000) return null;

  return {
    videoId: match[1],
    timestampMs,
  };
};

export const submittedFrameReference = (
  submission: CandidateSubmission,
): SubmittedFrameReference | null => {
  const parsedAnswer = parseSubmittedFrameReference(submission.answer);
  const candidateIdFrame = directFrameFromCandidateId(submission.candidate_id);
  const timestampMs =
    typeof submission.timestamp_ms === "number"
      ? submission.timestamp_ms
      : parsedAnswer?.timestampMs ?? candidateIdFrame?.timestampMs;
  if (typeof timestampMs !== "number") return null;

  const videoId =
    parsedAnswer?.videoId ?? candidateIdFrame?.videoId ?? submissionVideoId(submission);
  if (!videoId) return null;

  return { videoId, timestampMs };
};

export const submissionIdentityKey = (submission: CandidateSubmission) => {
  const frame = submittedFrameReference(submission);
  if (frame) return `frame|${frame.videoId}|${frame.timestampMs}`;

  return [
    "submission",
    submission.candidate_id,
    submission.answer,
    submission.video_id,
    submission.timestamp_ms,
  ]
    .map((part) => String(part ?? ""))
    .join("|");
};

const firstText = (...values: Array<string | null | undefined>) =>
  values.find((value) => typeof value === "string" && value.trim()) ?? null;

const mergeSubmissions = (
  base: CandidateSubmission,
  next: CandidateSubmission,
): CandidateSubmission => ({
  ...base,
  candidate_id: firstText(base.candidate_id, next.candidate_id) ?? base.candidate_id,
  answer: firstText(base.answer, next.answer),
  accepted: base.accepted || next.accepted,
  status: base.status ?? next.status ?? null,
  reason: firstText(base.reason, next.reason),
  source: firstText(base.source, next.source) ?? undefined,
  video_id: firstText(base.video_id, next.video_id),
  timestamp_ms: base.timestamp_ms ?? next.timestamp_ms ?? null,
  scene_start_ms: base.scene_start_ms ?? next.scene_start_ms ?? null,
  scene_end_ms: base.scene_end_ms ?? next.scene_end_ms ?? null,
});

export const uniqueSubmissions = (submissions: CandidateSubmission[]) => {
  const submissionsByKey = new Map<string, CandidateSubmission>();
  const keys: string[] = [];

  submissions.forEach((submission) => {
    const key = submissionIdentityKey(submission);
    const existingSubmission = submissionsByKey.get(key);
    if (existingSubmission) {
      submissionsByKey.set(key, mergeSubmissions(existingSubmission, submission));
      return;
    }

    keys.push(key);
    submissionsByKey.set(key, submission);
  });

  return keys.flatMap((key) => {
    const submission = submissionsByKey.get(key);
    return submission ? [submission] : [];
  });
};

export const submissionFrameUrl = (
  submission: CandidateSubmission,
  datasetValue: string,
) => {
  const frame = submittedFrameReference(submission);
  if (!frame) return "";

  return submittedFrameUrl(frame, datasetValue);
};

export const displayActionQuery = (action: string) => {
  const trimmed = action.trim();
  const actionMatch = trimmed.match(/^([a-z][a-z0-9_-]*)(?:\s*:\s*|\s+)(.+)$/i);

  if (actionMatch && ACTION_CODES.has(actionMatch[1].toLowerCase())) {
    return actionMatch[2].trim();
  }

  return trimmed;
};

export const displayPolicyRule = (policyRule: string) =>
  policyRule.replaceAll("_", " ");

export const formatMilliseconds = (value?: number) => {
  if (typeof value !== "number") return "n/a";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value >= 10 ? 0 : 1)} ms`;
};

export const formatTimestamp = (value: number) => {
  if (!Number.isFinite(value)) return "n/a";
  return `${(value / 1000).toFixed(2)} s`;
};

export const judgementLabel = (judgement?: string) => {
  if (judgement === "match") return "Match";
  if (judgement === "relevant") return "Relevant";
  if (judgement === "irrelevant") return "Irrelevant";
  return "Unclassified";
};

export type SubmissionDisplayStatus = "accepted" | "rejected" | "unavailable";

export const submissionStatus = (
  submission: CandidateSubmission,
): SubmissionDisplayStatus => {
  if (submission.status === "accepted") return "accepted";
  if (submission.status === "rejected") return "rejected";
  if (submission.status === "unavailable") return "unavailable";
  return submission.accepted ? "accepted" : "rejected";
};

export const submissionStatusLabel = (submission: CandidateSubmission) => {
  const status = submissionStatus(submission);
  if (status === "accepted") return "Submitted accepted";
  if (status === "rejected") return "Submitted rejected";
  return "Submitted unavailable";
};

export const candidateReason = (
  candidateId: string,
  iteration: RetrievalLoopIteration,
) => {
  const submission = iteration.submissions?.find(
    (result) => result.candidate_id === candidateId,
  );

  const evaluator = latestCandidateEvaluation(candidateId, iteration);
  if (evaluator?.reasoning) return evaluator.reasoning;
  if (submission?.reason) return submission.reason;
  if (submission && submissionStatus(submission) === "unavailable") {
    return "Submitted candidate was not classifiable from available competition judgements.";
  }
  if (submission?.accepted) return "Accepted as submitted answer.";
  if (evaluator?.matches_task) return "Accepted after candidate inspection.";

  const descriptions = iteration.browser_description_output?.results?.filter(
    (description) => description.candidate_id === candidateId,
  );
  if (descriptions?.length) {
    return descriptions
      .map((description) =>
        `${description.media_kind === "scene" ? "Scene" : "Frame"}: ${description.description}`,
      )
      .join(" ");
  }

  const relevant = iteration.browser_reasoning_output?.relevant?.find((reasoning) =>
    reasoning.candidate_ids.includes(candidateId),
  );
  if (relevant?.reasoning) return relevant.reasoning;

  const irrelevant = iteration.browser_reasoning_output?.irrelevant;
  if (irrelevant?.candidate_ids.includes(candidateId)) return irrelevant.reasoning;

  const classification = iteration.browser_output?.results?.find(
    (result) => result.candidate_id === candidateId,
  );
  if (classification?.judgement === "relevant") {
    return "Classified as relevant feedback for refining the search.";
  }
  if (classification?.judgement === "irrelevant") {
    return "Classified as irrelevant feedback for this task.";
  }

  return null;
};

export const latestCandidateEvaluation = (
  candidateId: string,
  iteration?: RetrievalLoopIteration,
) => {
  const evaluations = iteration?.evaluator_results ?? [];
  for (let index = evaluations.length - 1; index >= 0; index -= 1) {
    if (evaluations[index].candidate_id === candidateId) {
      return evaluations[index];
    }
  }
  return undefined;
};

export type IterationDiagnostics = {
  entries: HistoryEntry[];
  moduleEntries: HistoryEntry[];
  vlmEntries: HistoryEntry[];
  plannerInputHistory: unknown;
};

export const diagnosticsByIteration = (result: RetrievalLoopResult) => {
  const entriesByIteration = new Map<number, HistoryEntry[]>();
  result.full_history?.forEach((entry) => {
    const iteration = entryIteration(entry);
    if (iteration == null) return;
    const entries = entriesByIteration.get(iteration);
    if (entries) {
      entries.push(entry);
    } else {
      entriesByIteration.set(iteration, [entry]);
    }
  });

  return new Map(
    result.iterations.map((iteration) => [
      iteration.index,
      diagnosticsFromEntries(entriesByIteration.get(iteration.index) ?? []),
    ]),
  );
};

const diagnosticsFromEntries = (entries: HistoryEntry[]): IterationDiagnostics => {
  const moduleEntries = entries.filter((entry) => entry.kind !== "llm");
  const vlmEntries = entries.filter((entry) => entry.kind === "llm");
  const plannerEntry = moduleEntries.find((entry) => entry.stage === "planner");

  return {
    entries,
    moduleEntries,
    vlmEntries,
    plannerInputHistory: plannerInputHistory(plannerEntry),
  };
};

export const compactJson = (value: unknown) =>
  JSON.stringify(value, null, 2) ?? String(value);

export const readableJson = (value: unknown) => {
  const serialized = compactJson(value);
  let readable = "";

  for (let index = 0; index < serialized.length; ) {
    if (serialized[index] !== "\\") {
      readable += serialized[index];
      index += 1;
      continue;
    }

    const slashStart = index;
    while (serialized[index] === "\\") index += 1;

    const slashCount = index - slashStart;
    const escapeCharacter = serialized[index];
    const isLineBreak =
      slashCount % 2 === 1 &&
      (escapeCharacter === "n" || escapeCharacter === "r");

    if (!isLineBreak) {
      readable += serialized.slice(slashStart, index);
      continue;
    }

    readable += "\\".repeat(slashCount - 1);
    readable += "\n";
    index += 1;

    if (
      escapeCharacter === "r" &&
      serialized[index] === "\\" &&
      serialized[index + 1] === "n"
    ) {
      index += 2;
    }
  }

  return readable;
};

export const previewText = (value: unknown, maxLength = 260) => {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : compactJson(value);
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 1).trim()}...`;
};

export const plannerInputHistory = (entry?: HistoryEntry) => {
  const request = entry?.request;
  if (!isRecord(request)) return null;
  return request.history ?? null;
};

const entryIteration = (entry: HistoryEntry) => {
  const iteration = entry.metadata?.iteration;
  return typeof iteration === "number" ? iteration : null;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
