import type {
  CondensedPlannerHistory,
  CondensedPlannerHistoryStep,
  PlannerActionHistory,
  PlannerKeyframeEvidence,
  PlannerInvalidAttempt,
  PlannerWronglySubmittedCandidate,
} from "../../types/retrievalLoop";
import {
  displayActionQuery,
  displayPolicyRule,
  formatTimestamp,
} from "../../utils/retrievalLoop";

type CondensedHistoryProps = {
  history?: CondensedPlannerHistory | CondensedPlannerHistoryStep[];
};

export const CondensedHistory = ({ history }: CondensedHistoryProps) => {
  const normalized = normalizeHistory(history);
  const usedActions = normalized.used_actions ?? [];
  const latestAction = normalized.latest_valid_action ?? null;
  const invalidAttempts = normalized.invalid_attempts_after_latest_action ?? [];
  const keyframeEvidence = normalized.keyframe_evidence ?? [];
  const wronglySubmitted = normalized.wrongly_submitted_candidates ?? [];
  const validActionCount = usedActions.length + (latestAction ? 1 : 0);

  if (
    !latestAction &&
    !usedActions.length &&
    !invalidAttempts.length &&
    !keyframeEvidence.length &&
    !wronglySubmitted.length
  ) {
    return null;
  }

  return (
    <section className="history-panel" aria-label="Condensed planner history">
      <header className="history-panel__header">
        <div>
          <p className="eyebrow">Planner history</p>
          <h3>{validActionCount} valid actions</h3>
        </div>
      </header>

      {latestAction ? (
        <section
          className="history-feedback-group"
          aria-label="Latest valid action"
        >
          <h5>Latest valid action</h5>
          <ActionSummary action={latestAction} />
        </section>
      ) : null}

      {usedActions.length ? (
        <section className="history-feedback-group" aria-label="Used actions">
          <h5>Used actions</h5>
          <ol className="history-list">
            {usedActions.map((action, index) => (
              <li className="history-step" key={`${index}-${action.action}`}>
                <ActionSummary action={action} stepIndex={index + 1} />
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {invalidAttempts.length ? (
        <section
          className="history-feedback-group"
          aria-label="Invalid attempts"
        >
          <h5>Invalid attempts after latest action</h5>
          <ul className="history-feedback-list history-feedback-list--rejected">
            {invalidAttempts.map((attempt, index) => (
              <InvalidAttempt
                attempt={attempt}
                key={`${index}-${attempt.action}`}
              />
            ))}
          </ul>
        </section>
      ) : null}

      {wronglySubmitted.length ? (
        <section
          className="history-feedback-group"
          aria-label="Wrongly submitted candidates"
        >
          <h5>Wrongly Submitted Candidates</h5>
          <ul className="history-feedback-list history-feedback-list--rejected">
            {wronglySubmitted.map((candidate) => (
              <WronglySubmittedCandidate
                candidate={candidate}
                key={`${candidate.video_id}-${candidate.timestamp_ms}`}
              />
            ))}
          </ul>
        </section>
      ) : null}

      {keyframeEvidence.length ? (
        <section
          className="history-feedback-group"
          aria-label="Keyframe evidence"
        >
          <h5>Relevant keyframes</h5>
          <ul className="history-feedback-list history-feedback-list--relevant">
            {keyframeEvidence.map((evidence) => (
              <EvidenceItem evidence={evidence} key={evidence.keyframe_id} />
            ))}
          </ul>
        </section>
      ) : null}
    </section>
  );
};

const ActionSummary = ({
  action,
  stepIndex,
}: {
  action: PlannerActionHistory;
  stepIndex?: number;
}) => {
  const returnedKeyframes = [
    ...(action.result_groups?.relevant ?? []),
    ...(action.result_groups?.irrelevant ?? []),
    ...(action.result_groups?.highly_relevant ?? []),
  ].filter((keyframe, index, keyframes) => keyframes.indexOf(keyframe) === index);

  return (
    <>
      <div className="history-step__header">
        {stepIndex ? <span className="step-index">Step {stepIndex}</span> : null}
        <h4 title={action.action}>{displayActionQuery(action.action)}</h4>
      </div>

      {action.purpose || action.policy_rule ? (
        <div className="planner-rationale planner-rationale--history">
          {action.policy_rule ? (
            <span>{displayPolicyRule(action.policy_rule)}</span>
          ) : null}
          {action.purpose ? <p>{action.purpose}</p> : null}
        </div>
      ) : null}

      {returnedKeyframes.length ? (
        <p className="history-step__summary">
          Returned keyframes: {returnedKeyframes.join(", ")}
        </p>
      ) : null}

      {action.result_summary ? (
        <p className="history-step__summary">{action.result_summary}</p>
      ) : null}

      {action.evaluator_feedback ? (
        <p className="history-feedback-note">{action.evaluator_feedback}</p>
      ) : null}
    </>
  );
};

const InvalidAttempt = ({ attempt }: { attempt: PlannerInvalidAttempt }) => (
  <li>
    <strong>{attempt.action}</strong>
    {attempt.policy_rule ? (
      <span>{displayPolicyRule(attempt.policy_rule)}</span>
    ) : null}
    {attempt.purpose ? <span>{attempt.purpose}</span> : null}
    <span>
      {attempt.result_summary ??
        attempt.evaluator_feedback ??
        "Invalid planner attempt."}
    </span>
  </li>
);

const EvidenceItem = ({ evidence }: { evidence: PlannerKeyframeEvidence }) => (
  <li>
    <strong>
      {evidence.keyframe_id} ·{" "}
      {evidence.category === "highly_relevant" ? "highly relevant" : "relevant"}
    </strong>
    <span>
      Frame: {evidence.frame_description} {evidence.reason}
      {evidence.scene_description ? (
        <> Scene: {evidence.scene_description}</>
      ) : null}
    </span>
  </li>
);

const WronglySubmittedCandidate = ({
  candidate,
}: {
  candidate: PlannerWronglySubmittedCandidate;
}) => (
  <li>
    <strong>{candidate.video_id}</strong>
    <span>Timestamp: {formatTimestamp(candidate.timestamp_ms)}</span>
    {candidate.scene_start_ms != null && candidate.scene_end_ms != null ? (
      <span>
        Rejected scene: {formatTimestamp(candidate.scene_start_ms)}–
        {formatTimestamp(candidate.scene_end_ms)}
      </span>
    ) : null}
  </li>
);

const normalizeHistory = (
  history?: CondensedPlannerHistory | CondensedPlannerHistoryStep[],
): CondensedPlannerHistory => {
  if (!history) return {};
  if (!Array.isArray(history)) return history;

  const used_actions = history.map((step) => ({
    action: step.action,
    purpose: step.purpose,
    policy_rule: step.policy_rule,
    result_groups: {
      relevant:
        step.evaluator_result?.relevant_images?.map(
          (image) => image.image_id,
        ) ?? [],
      highly_relevant:
        step.evaluator_result?.highly_relevant_images?.map(
          (image) => image.image_id,
        ) ?? [],
      irrelevant:
        step.evaluator_result?.irrelevant_groups?.flatMap(
          (group) => group.image_ids,
        ) ?? [],
    },
    result_summary: step.result_summary,
    evaluator_feedback: step.evaluator_result?.evaluator_feedback,
  }));
  const keyframe_evidence = history.flatMap((step) => [
    ...(step.evaluator_result?.relevant_images ?? []).map((image) => ({
      category: "relevant" as const,
      keyframe_id: image.image_id,
      video_id: image.video_id,
      timestamp_ms: image.timestamp_ms,
      frame_description: image.description,
      scene_description: image.scene_description,
      reason: image.reason,
    })),
    ...(step.evaluator_result?.highly_relevant_images ?? []).map((image) => ({
      category: "highly_relevant" as const,
      keyframe_id: image.image_id,
      video_id: image.video_id,
      timestamp_ms: image.timestamp_ms,
      frame_description: image.description,
      scene_description: image.scene_description,
      reason: image.reason,
    })),
  ]);
  const wrongly_submitted_candidates = history
    .flatMap((step) => step.wrongly_submitted_candidates ?? [])
    .filter(
      (candidate, index, candidates) =>
        candidates.findIndex(
          (item) =>
            item.video_id === candidate.video_id &&
            item.timestamp_ms === candidate.timestamp_ms,
        ) === index,
    );

  return {
    latest_valid_action: used_actions.at(-1) ?? null,
    used_actions: used_actions.slice(0, -1),
    invalid_attempts_after_latest_action: [],
    keyframe_evidence,
    wrongly_submitted_candidates,
  };
};
