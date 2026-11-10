import type {
  BrowserClassification,
  Candidate,
  CandidateSubmission,
  CandidateEvaluationResult,
} from "../../types/retrievalLoop";
import {
  candidateImageUrls,
  candidateKey,
  formatTimestamp,
  judgementLabel,
  submissionStatus,
  submissionStatusLabel,
} from "../../utils/retrievalLoop";

type CandidateCardProps = {
  candidate: Candidate;
  classification?: BrowserClassification;
  evaluation?: CandidateEvaluationResult;
  submission?: CandidateSubmission;
  reason: string | null;
  dataset: string;
  isFiltered?: boolean;
  onOpen: (candidate: Candidate) => void;
};

export const CandidateCard = ({
  candidate,
  classification,
  evaluation,
  submission,
  reason,
  dataset,
  isFiltered = false,
  onOpen,
}: CandidateCardProps) => {
  const id = candidateKey(candidate);
  const imageUrls = candidateImageUrls(candidate, dataset);
  const isGrouped = imageUrls.length > 1;
  const judgement = classification?.judgement;
  const evaluatorLabel = evaluation
    ? evaluation.matches_task
      ? "Exact scene selected"
      : "Downstream check rejected"
    : null;
  const submissionLabel = submission ? submissionStatusLabel(submission) : null;
  const displaySubmissionStatus = submission ? submissionStatus(submission) : null;
  const submissionClass = submission
    ? `candidate-card--submitted-${displaySubmissionStatus}`
    : "";

  return (
    <article
      className={[
        "candidate-card",
        `candidate-card--${judgement ?? "pending"}`,
        submissionClass,
        isFiltered ? "candidate-card--filtered" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {isFiltered ? (
        <div className="candidate-card__seen-banner">Previously seen</div>
      ) : null}
      <button
        className="candidate-media candidate-media--button"
        onClick={() => onOpen(candidate)}
        type="button"
      >
        {imageUrls.length ? (
          <span
            className={
              isGrouped ? "candidate-frame-strip" : "candidate-frame-strip--single"
            }
          >
            {imageUrls.map((imageUrl, index) => (
              <img
                alt={`Candidate ${id} frame ${index + 1}`}
                key={`${imageUrl}-${index}`}
                loading="lazy"
                src={imageUrl}
              />
            ))}
          </span>
        ) : (
          <span>No preview</span>
        )}
      </button>

      <div className="candidate-body">
        <div className="candidate-title-row">
          <h4 title={id}>{id}</h4>
          <div className="candidate-badges">
            {isFiltered ? (
              <span className="judgement-badge judgement-badge--filtered">
                Previously seen
              </span>
            ) : null}
            <span className={`judgement-badge judgement-badge--${judgement ?? "pending"}`}>
              {judgementLabel(judgement)}
            </span>
          </div>
        </div>

        <dl className="candidate-meta">
          <div>
            <dt>Rank</dt>
            <dd>{candidate.rank}</dd>
          </div>
          <div>
            <dt>Score</dt>
            <dd>{candidate.score}</dd>
          </div>
          <div>
            <dt>{candidate.frames?.length ? "Frames" : "Keyframe"}</dt>
            <dd>{formatTimestamp(candidate.time_keyframe)}</dd>
          </div>
        </dl>

        {evaluatorLabel ? <p className="evaluator-label">{evaluatorLabel}</p> : null}
        {submissionLabel ? (
          <p
            className={`submission-label submission-label--${displaySubmissionStatus}`}
          >
            {submissionLabel}
          </p>
        ) : null}
        {reason ? <p className="candidate-reason">{reason}</p> : null}
      </div>
    </article>
  );
};
