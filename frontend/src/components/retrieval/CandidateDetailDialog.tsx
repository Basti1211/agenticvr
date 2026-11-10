import type {
  BrowserClassification,
  Candidate,
  CandidateSubmission,
  CandidateEvaluationResult,
} from "../../types/retrievalLoop";
import {
  candidateFrames,
  candidateImageUrls,
  candidateKey,
  formatTimestamp,
  judgementLabel,
  submissionStatus,
  submissionStatusLabel,
} from "../../utils/retrievalLoop";

type CandidateDetailDialogProps = {
  candidate: Candidate | null;
  classification?: BrowserClassification;
  evaluation?: CandidateEvaluationResult;
  submission?: CandidateSubmission;
  reason: string | null;
  dataset: string;
  onClose: () => void;
};

export const CandidateDetailDialog = ({
  candidate,
  classification,
  evaluation,
  submission,
  reason,
  dataset,
  onClose,
}: CandidateDetailDialogProps) => {
  if (!candidate) return null;

  const id = candidateKey(candidate);
  const frames = candidateFrames(candidate);
  const imageUrls = candidateImageUrls(candidate, dataset);
  const isGrouped = imageUrls.length > 1;
  const displaySubmissionStatus = submission ? submissionStatus(submission) : null;

  return (
    <div
      aria-label={`Candidate ${id} details`}
      aria-modal="true"
      className="candidate-dialog"
      role="dialog"
    >
      <button
        aria-label="Close candidate details"
        className="candidate-dialog__backdrop"
        onClick={onClose}
        type="button"
      />
      <section className="candidate-dialog__panel">
        <header className="candidate-dialog__header">
          <div>
            <p className="eyebrow">Candidate detail</p>
            <h2>{id}</h2>
          </div>
          <button className="icon-button" onClick={onClose} type="button">
            X
          </button>
        </header>

        <div className="candidate-dialog__content">
          <div className="candidate-dialog__image">
            {imageUrls.length ? (
              <div
                className={
                  isGrouped
                    ? "candidate-dialog__frame-grid"
                    : "candidate-dialog__frame-grid candidate-dialog__frame-grid--single"
                }
              >
                {imageUrls.map((imageUrl, index) => (
                  <img
                    alt={`Candidate ${id} frame ${index + 1}`}
                    key={`${imageUrl}-${index}`}
                    src={imageUrl}
                  />
                ))}
              </div>
            ) : (
              "No preview"
            )}
          </div>

          <aside className="candidate-dialog__info">
            <span
              className={`judgement-badge judgement-badge--${
                classification?.judgement ?? "pending"
              }`}
            >
              {judgementLabel(classification?.judgement)}
            </span>

            <dl className="detail-list">
              <div>
                <dt>Rank</dt>
                <dd>{candidate.rank}</dd>
              </div>
              <div>
                <dt>Score</dt>
                <dd>{candidate.score}</dd>
              </div>
              <div>
                <dt>{frames.length > 1 ? "First frame" : "Keyframe"}</dt>
                <dd>{formatTimestamp(candidate.time_keyframe)}</dd>
              </div>
              <div>
                <dt>Scene</dt>
                <dd>
                  {formatTimestamp(candidate.time_scene_start)} -{" "}
                  {formatTimestamp(candidate.time_scene_end)}
                </dd>
              </div>
              <div>
                <dt>Source URI</dt>
                <dd>{candidate.uri}</dd>
              </div>
            </dl>

            {frames.length > 1 ? (
              <ol className="candidate-frame-list">
                {frames.map((frame, index) => (
                  <li key={`${frame.id.join("_")}-${index}`}>
                    <strong>{frame.id.join("_")}</strong>
                    <span>{formatTimestamp(frame.time_keyframe)}</span>
                    <small>{frame.uri}</small>
                  </li>
                ))}
              </ol>
            ) : null}

            {evaluation ? (
              <p className="evaluator-label">
                {evaluation.matches_task
                  ? "Exact scene selected"
                  : "Downstream check rejected"}
              </p>
            ) : null}
            {submission ? (
              <p
                className={`submission-label submission-label--${displaySubmissionStatus}`}
              >
                {submissionStatusLabel(submission)}
              </p>
            ) : null}
            {reason ? <p className="candidate-reason">{reason}</p> : null}
          </aside>
        </div>
      </section>
    </div>
  );
};
