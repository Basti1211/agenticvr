import { useMemo, useState } from "react";

import type { Candidate, RetrievalLoopResult } from "../../types/retrievalLoop";
import {
  candidateKey,
  candidateReason,
  diagnosticsByIteration,
  latestCandidateEvaluation,
} from "../../utils/retrievalLoop";
import { LazyJsonDetails } from "../LazyJsonDetails";
import { CandidateDetailDialog } from "./CandidateDetailDialog";
import { IterationStep } from "./IterationStep";
import { TimingSummary } from "./TimingSummary";
import "./retrieval.css";

type LoopVisualizationProps = {
  result: RetrievalLoopResult;
  dataset: string;
  isContinuing: boolean;
  onContinue: (stepCount: number) => void;
  allowContinue?: boolean;
  allowContinueAfterFound?: boolean;
};

export const LoopVisualization = ({
  result,
  dataset,
  isContinuing,
  onContinue,
  allowContinue = true,
  allowContinueAfterFound = false,
}: LoopVisualizationProps) => {
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const canContinue =
    allowContinue &&
    (result.status === "not_found" ||
      (allowContinueAfterFound && result.status === "found"));
  const nextStepIndex = result.iterations.length + 1;
  const diagnostics = useMemo(() => diagnosticsByIteration(result), [result]);

  const selectedContext = useMemo(() => {
    if (!selectedCandidate) return null;

    const id = candidateKey(selectedCandidate);
    const activeIteration = [...result.iterations].reverse().find((item) =>
      item.candidates?.some((candidate) => candidateKey(candidate) === id),
    );
    const iteration =
      activeIteration ??
      result.iterations.find((item) =>
        item.filtered_candidates?.some((candidate) => candidateKey(candidate) === id),
      );
    const classification = iteration?.browser_output?.results?.find(
      (item) => item.candidate_id === id,
    );
    const evaluation = latestCandidateEvaluation(id, iteration);
    const submission = iteration?.submissions?.find(
      (item) => item.candidate_id === id,
    );

    return {
      classification,
      evaluation,
      reason: activeIteration ? candidateReason(id, activeIteration) : null,
      submission,
    };
  }, [result.iterations, selectedCandidate]);

  return (
    <section className="loop-panel" aria-label="Retrieval loop visualization">
      <header className="loop-panel__header">
        <div>
          <p className="eyebrow">Loop result</p>
          <h2>{result.status.replaceAll("_", " ")}</h2>
        </div>
        <div className="loop-panel__actions">
          {result.final_candidate_id ? (
            <span className="final-badge">
              Final: {result.final_candidate_id}
              {result.final_submission?.source === "target_verifier"
                ? " · target checked"
                : ""}
            </span>
          ) : null}
          {result.final_answer ? (
            <span className="final-badge">Answer: {result.final_answer}</span>
          ) : null}
          {canContinue ? (
            <>
              <button
                className="secondary-button"
                disabled={isContinuing}
                onClick={() => onContinue(1)}
                type="button"
              >
                {isContinuing ? "Continuing..." : `Make step ${nextStepIndex}`}
              </button>
              <button
                className="secondary-button"
                disabled={isContinuing}
                onClick={() => onContinue(10)}
                type="button"
              >
                {isContinuing ? "Continuing..." : "Make 10 steps"}
              </button>
            </>
          ) : null}
        </div>
      </header>

      <TimingSummary timings={result.timings_ms} />

      <div className="iteration-list">
        {result.iterations.map((iteration) => (
          <IterationStep
            diagnostics={diagnostics.get(iteration.index)!}
            dataset={dataset}
            iteration={iteration}
            key={`${iteration.index}-${iteration.action}`}
            onOpenCandidate={setSelectedCandidate}
          />
        ))}
      </div>

      <LazyJsonDetails
        className="raw-result"
        summary="Raw loop payload"
        value={result}
      />

      <CandidateDetailDialog
        candidate={selectedCandidate}
        classification={selectedContext?.classification}
        dataset={dataset}
        evaluation={selectedContext?.evaluation}
        onClose={() => setSelectedCandidate(null)}
        reason={selectedContext?.reason ?? null}
        submission={selectedContext?.submission}
      />
    </section>
  );
};
