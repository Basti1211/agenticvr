import { useState } from "react";

import type {
  Candidate,
  RetrievalLoopIteration,
} from "../../types/retrievalLoop";
import {
  candidateImageUrl,
  candidateKey,
  candidateReason,
  displayActionQuery,
  displayPolicyRule,
  type IterationDiagnostics,
  previewText,
  readableJson,
  submittedFrameReference,
  submissionFrameUrl,
  submissionFromPlannerAction,
  submissionStatus,
  submissionStatusLabel,
  uniqueSubmissions,
} from "../../utils/retrievalLoop";
import { CandidateCard } from "./CandidateCard";
import { TimingSummary } from "./TimingSummary";

type IterationStepProps = {
  iteration: RetrievalLoopIteration;
  dataset: string;
  diagnostics: IterationDiagnostics;
  onOpenCandidate: (candidate: Candidate) => void;
};

export const IterationStep = ({
  iteration,
  dataset,
  diagnostics,
  onOpenCandidate,
}: IterationStepProps) => {
  const [failedSubmissionImages, setFailedSubmissionImages] = useState<Set<string>>(
    () => new Set(),
  );
  const candidates = iteration.candidates ?? [];
  const filteredCandidates = iteration.filtered_candidates ?? [];
  const displayedCandidates = rankedCandidates(candidates, filteredCandidates);
  const loggedCandidateIds =
    displayedCandidates.length === 0
      ? (iteration.candidate_ids ?? iteration.filtered_candidate_ids ?? [])
      : [];
  const plannerActionSubmission = submissionFromPlannerAction(iteration.action);
  const visibleSubmissions = uniqueSubmissions([
    ...(iteration.submissions ?? []),
    ...(plannerActionSubmission ? [plannerActionSubmission] : []),
  ]);
  const candidatesById = new Map(
    candidates.map((candidate) => [candidateKey(candidate), candidate]),
  );
  const classifications = new Map(
    iteration.browser_output?.results?.map((result) => [result.candidate_id, result]) ?? [],
  );
  const evaluations = new Map(
    iteration.evaluator_results?.map((result) => [result.candidate_id, result]) ?? [],
  );
  const submissions = new Map(
    visibleSubmissions.map((result) => [result.candidate_id, result]),
  );
  const matchCount =
    iteration.browser_output?.results?.filter((result) => result.judgement === "match")
      .length ?? 0;
  const relevantCount =
    iteration.browser_output?.results?.filter((result) => result.judgement === "relevant")
      .length ?? 0;
  const irrelevantCount =
    iteration.browser_output?.results?.filter(
      (result) => result.judgement === "irrelevant",
    ).length ?? 0;
  const submittedAcceptedCount =
    visibleSubmissions.filter((result) => submissionStatus(result) === "accepted")
      .length ?? 0;
  const submittedRejectedCount =
    visibleSubmissions.filter((result) => submissionStatus(result) === "rejected")
      .length ?? 0;
  const submittedUnavailableCount =
    visibleSubmissions.filter((result) => submissionStatus(result) === "unavailable")
      .length ?? 0;
  const submittedPlannerActionCount = visibleSubmissions.filter(
    (submission) => submission.source === "planner" && submission.status == null,
  ).length;
  const vqaResults =
    iteration.evaluator_results
      ?.map((result) => ({
        candidateId: result.candidate_id,
        vqa: result.metadata?.vqa as VqaMetadata | undefined,
        validation: result.metadata?.vqa_answer_validation as
          | Record<string, unknown>
          | undefined,
      }))
      .filter((item) => item.vqa) ?? [];

  return (
    <article
      className="iteration-card"
      id={`task-log-step-${iteration.index}`}
      tabIndex={-1}
    >
      <header className="iteration-card__header">
        <div>
          <span className="step-index">Step {iteration.index}</span>
          <h3 title={iteration.action}>{displayActionQuery(iteration.action)}</h3>
          <p>{iteration.endpoint}</p>
        </div>
        <TimingSummary timings={iteration.timings_ms} />
      </header>

      {iteration.planner_purpose || iteration.planner_policy_rule ? (
        <section className="planner-rationale" aria-label="Planner decision rationale">
          {iteration.planner_policy_rule ? (
            <span>{displayPolicyRule(iteration.planner_policy_rule)}</span>
          ) : null}
          {iteration.planner_purpose ? <p>{iteration.planner_purpose}</p> : null}
        </section>
      ) : null}

      {iteration.retrieval_error ? (
        <p className="status-message status-message--error">{iteration.retrieval_error}</p>
      ) : null}

      <div className="iteration-summary" aria-label={`Step ${iteration.index} result`}>
        <span>{candidates.length} new candidates</span>
        {filteredCandidates.length ? (
          <span>{filteredCandidates.length} previously seen</span>
        ) : null}
        <span>{matchCount} exact scenes</span>
        <span>{relevantCount} relevant</span>
        <span>{irrelevantCount} irrelevant</span>
        {visibleSubmissions.length ? (
          <>
            {submittedPlannerActionCount ? (
              <span>{submittedPlannerActionCount} planner answer submitted</span>
            ) : null}
            {submittedAcceptedCount ? (
              <span>{submittedAcceptedCount} submitted accepted</span>
            ) : null}
            {submittedRejectedCount ? (
              <span>{submittedRejectedCount} submitted rejected</span>
            ) : null}
            {submittedUnavailableCount ? (
              <span>{submittedUnavailableCount} submitted unavailable</span>
            ) : null}
          </>
        ) : null}
      </div>

      {iteration.answerer_output || iteration.answerer_error ? (
        <section
          className={`answerer-strip answerer-strip--${
            iteration.answerer_validation?.status ?? "error"
          }`}
          aria-label="Independent answerer submission"
        >
          <div>
            <h4>Answerer guess</h4>
            {iteration.answerer_validation ? (
              <span className="answerer-status">
                {iteration.answerer_validation.status.replaceAll("_", " ")}
              </span>
            ) : null}
          </div>
          {iteration.answerer_output ? (
            <code>
              {iteration.answerer_output.answer ??
                `${iteration.answerer_output.video_id}, ${iteration.answerer_output.timestamp_ms}`}
            </code>
          ) : null}
          {iteration.answerer_validation?.reason ? (
            <p>{iteration.answerer_validation.reason}</p>
          ) : null}
          {iteration.answerer_error ? (
            <p className="answerer-strip__error">{iteration.answerer_error}</p>
          ) : null}
        </section>
      ) : null}

      {vqaResults.length ? (
        <section className="vqa-strip" aria-label="Video question answering">
          <h4>VQA answers</h4>
          <div className="vqa-list">
            {vqaResults.map(({ candidateId, vqa, validation }) => (
              <details key={candidateId} open={Boolean(vqa?.answer)}>
                <summary>
                  <span>{candidateId}</span>
                  <strong>{vqa?.answerable ? "answered" : "failed"}</strong>
                  {vqa?.answer ? <em>{vqa.answer}</em> : null}
                </summary>
                {vqa?.reason ? <p>{vqa.reason}</p> : null}
                {validation ? (
                  <p className="vqa-validation">
                    Text check: {String(validation.status ?? "unknown")}
                    {validation.reason ? ` - ${String(validation.reason)}` : ""}
                  </p>
                ) : null}
                {vqa?.observations?.length ? (
                  <ol className="vqa-observation-list">
                    {vqa.observations.map((observation) => (
                      <li key={observation.index}>
                        <span>
                          Segment {observation.index} ·{" "}
                          {formatSeconds(observation.start_seconds)} -{" "}
                          {formatSeconds(observation.end_seconds)}
                        </span>
                        {observation.answer ? <strong>{observation.answer}</strong> : null}
                        {observation.verification_reason ? (
                          <small>{observation.verification_reason}</small>
                        ) : null}
                      </li>
                    ))}
                  </ol>
                ) : null}
              </details>
            ))}
          </div>
        </section>
      ) : null}

      {visibleSubmissions.length ? (
        <section className="submission-strip" aria-label="Submitted candidates">
          <h4>
            {visibleSubmissions.some((submission) => submission.answer)
              ? "Submitted answers"
              : "Submitted candidates"}
          </h4>
          <div className="submission-list">
            {visibleSubmissions.map((submission) => {
              const submittedCandidate = candidatesById.get(submission.candidate_id);
              const imageUrl =
                submittedFrameReference(submission)
                  ? submissionFrameUrl(submission, dataset)
                  : submittedCandidate
                    ? candidateImageUrl(submittedCandidate, dataset)
                    : "";
              const isPlannerActionSubmission =
                submission.source === "planner" && submission.status == null;
              const status = isPlannerActionSubmission
                ? "unavailable"
                : submissionStatus(submission);
              const displayAnswer = submission.answer?.trim();
              const imageKey = `${submission.candidate_id}-${imageUrl}`;
              const imageFailed = failedSubmissionImages.has(imageKey);
              return (
                <button
                  className={`submission-preview submission-preview--${status}`}
                  disabled={!submittedCandidate}
                  key={`${submission.candidate_id}-${displayAnswer ?? submission.source ?? "submission"}`}
                  onClick={() => {
                    if (submittedCandidate) onOpenCandidate(submittedCandidate);
                  }}
                  title={submission.reason ?? undefined}
                  type="button"
                >
                  {imageUrl && !imageFailed ? (
                    <img
                      alt={`Submitted candidate ${submission.candidate_id}`}
                      onError={() =>
                        setFailedSubmissionImages((current) =>
                          new Set(current).add(imageKey),
                        )
                      }
                      src={imageUrl}
                    />
                  ) : displayAnswer ? (
                    <code className="submission-answer-text">{displayAnswer}</code>
                  ) : (
                    <span>No preview</span>
                  )}
                  <strong>
                    {isPlannerActionSubmission
                      ? "Submitted"
                      : submissionStatusLabel(submission).replace("Submitted ", "")}
                  </strong>
                  <small>{displayAnswer ? submission.source : submission.candidate_id}</small>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      <div className="candidate-grid">
        {displayedCandidates.map(({ candidate, isFiltered }) => {
          const id = candidateKey(candidate);
          return (
            <CandidateCard
              candidate={candidate}
              classification={isFiltered ? undefined : classifications.get(id)}
              dataset={dataset}
              evaluation={isFiltered ? undefined : evaluations.get(id)}
              isFiltered={isFiltered}
              key={isFiltered ? `filtered-${id}` : id}
              onOpen={onOpenCandidate}
              reason={isFiltered ? null : candidateReason(id, iteration)}
              submission={isFiltered ? undefined : submissions.get(id)}
            />
          );
        })}
      </div>

      {loggedCandidateIds.length ? (
        <section className="logged-candidate-ids" aria-label="Logged candidate ids">
          <h4>Returned candidate ids</h4>
          <ol>
            {loggedCandidateIds.map((candidateId, index) => (
              <li key={`${candidateId}-${index}`}>{candidateId}</li>
            ))}
          </ol>
        </section>
      ) : null}

      {candidates.length === 0 &&
      filteredCandidates.length === 0 &&
      loggedCandidateIds.length === 0 &&
      !iteration.retrieval_error ? (
        <p className="empty-state">No candidates were returned for this action.</p>
      ) : null}

      <IterationDiagnosticsPanel diagnostics={diagnostics} />
    </article>
  );
};

type VqaObservation = {
  index: number;
  start_seconds: number;
  end_seconds?: number | null;
  answer?: string | null;
  verified?: boolean | null;
  verification_reason?: string | null;
};

type VqaMetadata = {
  answerable?: boolean;
  answer?: string | null;
  reason?: string | null;
  observations?: VqaObservation[];
};

const formatSeconds = (value?: number | null) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "end";
  return `${value.toFixed(2)} s`;
};

const rankedCandidates = (
  candidates: Candidate[],
  filteredCandidates: Candidate[],
) =>
  [
    ...candidates.map((candidate, index) => ({
      candidate,
      index,
      isFiltered: false,
    })),
    ...filteredCandidates.map((candidate, index) => ({
      candidate,
      index: candidates.length + index,
      isFiltered: true,
    })),
  ].sort((left, right) => {
    const rankDifference = candidateRank(left.candidate) - candidateRank(right.candidate);
    return rankDifference || left.index - right.index;
  });

const candidateRank = (candidate: Candidate) => {
  const rank = Number.parseFloat(candidate.rank);
  return Number.isFinite(rank) ? rank : Number.POSITIVE_INFINITY;
};

const IterationDiagnosticsPanel = ({
  diagnostics,
}: {
  diagnostics: IterationDiagnostics;
}) => {
  if (
    diagnostics.entries.length === 0 &&
    !diagnostics.plannerInputHistory
  ) {
    return null;
  }

  return (
    <div className="iteration-diagnostics">
      {diagnostics.plannerInputHistory ? (
        <details>
          <summary>Planner input history</summary>
          <pre>{readableJson(diagnostics.plannerInputHistory)}</pre>
        </details>
      ) : null}

      {diagnostics.vlmEntries.length ? (
        <details>
          <summary>VLM calls ({diagnostics.vlmEntries.length})</summary>
          <ol className="vlm-call-list">
            {diagnostics.vlmEntries.map((entry, index) => (
              <li key={`${entry.stage}-${index}`}>
                <details>
                  <summary>
                    <span>{entry.stage}</span>
                    <span>{entryLabel(entry)}</span>
                    {entry.response ? <em>{previewText(entry.response)}</em> : null}
                  </summary>
                  <div className="vlm-call-body">
                    {entry.request ? (
                      <section>
                        <h4>Input</h4>
                        <pre>{readableJson(entry.request)}</pre>
                      </section>
                    ) : null}
                    {entry.response ? (
                      <section>
                        <h4>Output</h4>
                        <pre>{readableJson(entry.response)}</pre>
                      </section>
                    ) : null}
                    {entry.error ? (
                      <section>
                        <h4>Error</h4>
                        <pre>{entry.error}</pre>
                      </section>
                    ) : null}
                  </div>
                </details>
              </li>
            ))}
          </ol>
        </details>
      ) : null}

      {diagnostics.moduleEntries.length ? (
        <details>
          <summary>Round log ({diagnostics.moduleEntries.length})</summary>
          <ol className="module-log-list">
            {diagnostics.moduleEntries.map((entry, index) => (
              <li key={`${entry.stage}-${index}`}>
                <details>
                  <summary>
                    <span>{entry.stage}</span>
                    {entry.error ? <em>{entry.error}</em> : null}
                  </summary>
                  <pre>{readableJson(entry)}</pre>
                </details>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </div>
  );
};

const entryLabel = (entry: { metadata?: Record<string, unknown> }) => {
  const metadata = entry.metadata ?? {};
  const parts = [
    metadata.model,
    metadata.phase,
    metadata.judgement,
    metadata.candidate_id,
    Array.isArray(metadata.candidate_ids) ? metadata.candidate_ids.join(", ") : null,
  ].filter((part): part is string => typeof part === "string" && part.length > 0);

  return parts.join(" · ");
};
