export type CandidateFrame = {
  uri: string;
  id: string[];
  time_keyframe: number;
  time_scene_start: number;
  time_scene_end: number;
};

export type Candidate = CandidateFrame & {
  rank: string;
  score: string;
  frames?: CandidateFrame[];
};

export type BrowserClassification = {
  candidate_id: string;
  judgement: "match" | "relevant" | "irrelevant" | string;
  reasoning?: string | null;
  matching_evidence?: string[];
  missing_evidence?: string[];
  contradicting_evidence?: string[];
};

export type BrowserReasoning = {
  candidate_ids: string[];
  judgement: string;
  reasoning: string;
};

export type BrowserDescription = {
  candidate_id: string;
  media_kind: "frame" | "scene";
  description: string;
};

export type VlmPromptTrace = {
  system_prompt?: string;
  context_prompt?: string;
  context_images?: Array<Record<string, unknown>>;
  context_videos?: Array<Record<string, unknown>>;
  [key: string]: unknown;
};

export type VlmCall = {
  prompt?: VlmPromptTrace | Record<string, unknown>;
  response?: string;
  model?: string;
  phase?: string;
  judgement?: string;
  candidate_id?: string;
  candidate_ids?: string[];
  error?: string;
  [key: string]: unknown;
};

export type CandidateEvaluationResult = {
  candidate_id: string;
  matches_task: boolean;
  reasoning?: string | null;
  metadata?: {
    llm_calls?: VlmCall[];
    [key: string]: unknown;
  };
};

export type CandidateSubmission = {
  candidate_id: string;
  answer?: string | null;
  accepted: boolean;
  status?: "accepted" | "rejected" | "unavailable" | null;
  reason?: string | null;
  source?: string;
  video_id?: string | null;
  timestamp_ms?: number | null;
  scene_start_ms?: number | null;
  scene_end_ms?: number | null;
};

export type AnswererOutput = {
  video_id: string;
  timestamp_ms: number;
  answer?: string;
  metadata?: Record<string, unknown>;
};

export type AnswererValidation = {
  status: "accepted" | "rejected" | "unavailable";
  reason: string;
  source?: string | null;
};

export type PlannerEvaluatedImage = {
  image_id: string;
  description: string;
  reason: string;
  scene_description?: string | null;
  video_id?: string | null;
  timestamp_ms?: number | null;
};

export type PlannerIrrelevantImageGroup = {
  image_ids: string[];
};

export type PlannerEvaluatorResult = {
  relevant_images?: PlannerEvaluatedImage[];
  highly_relevant_images?: PlannerEvaluatedImage[];
  irrelevant_groups?: PlannerIrrelevantImageGroup[];
  evaluator_feedback?: string | null;
};

export type RetrievalLoopIteration = {
  index: number;
  action: string;
  planner_purpose?: string | null;
  planner_policy_rule?: string | null;
  endpoint: string;
  candidates?: Candidate[];
  filtered_candidates?: Candidate[];
  candidate_ids?: string[];
  filtered_candidate_ids?: string[];
  timings_ms?: Record<string, number>;
  retrieval_error?: string | null;
  browser_output?: {
    results?: BrowserClassification[];
    metadata?: {
      llm_calls?: VlmCall[];
      [key: string]: unknown;
    };
  } | null;
  browser_reasoning_output?: {
    relevant?: BrowserReasoning[];
    irrelevant?: BrowserReasoning | null;
    metadata?: {
      llm_calls?: VlmCall[];
      [key: string]: unknown;
    };
  } | null;
  browser_description_output?: {
    results?: BrowserDescription[];
    metadata?: {
      llm_calls?: VlmCall[];
      [key: string]: unknown;
    };
  } | null;
  evaluator_results?: Array<
    CandidateEvaluationResult & {
      metadata?: {
        llm_calls?: VlmCall[];
        [key: string]: unknown;
      };
    }
  >;
  submissions?: CandidateSubmission[];
  answerer_output?: AnswererOutput | null;
  answerer_validation?: AnswererValidation | null;
  answerer_error?: string | null;
};

export type HistoryEntry = {
  stage: string;
  kind: string;
  request?: unknown;
  response?: unknown;
  metadata?: Record<string, unknown>;
  error?: string | null;
};

export type CondensedPlannerHistoryStep = {
  action: string;
  purpose?: string | null;
  policy_rule?: string | null;
  result_candidate_ids?: string[];
  result_summary?: string | null;
  evaluator_result?: PlannerEvaluatorResult;
  wrongly_submitted_candidates?: PlannerWronglySubmittedCandidate[];
};

export type PlannerWronglySubmittedCandidate = {
  video_id: string;
  timestamp_ms: number;
  scene_start_ms?: number | null;
  scene_end_ms?: number | null;
};

export type PlannerResultGroups = {
  relevant?: string[];
  irrelevant?: string[];
  highly_relevant?: string[];
};

export type PlannerActionHistory = {
  action: string;
  purpose?: string | null;
  policy_rule?: string | null;
  result_groups?: PlannerResultGroups;
  result_summary?: string | null;
  evaluator_feedback?: string | null;
  irrelevant_reason?: string | null;
};

export type PlannerInvalidAttempt = {
  action: string;
  purpose?: string | null;
  policy_rule?: string | null;
  result_summary?: string | null;
  evaluator_feedback?: string | null;
};

export type PlannerCanonicalEvidence = {
  category: "relevant" | "highly_relevant" | string;
  keyframe_ids?: string[];
  video_id?: string | null;
  start_ms?: number | null;
  end_ms?: number | null;
  description: string;
  reason: string;
};

export type PlannerKeyframeEvidence = {
  category: "relevant" | "highly_relevant" | string;
  keyframe_id: string;
  video_id?: string | null;
  timestamp_ms?: number | null;
  frame_description: string;
  scene_description?: string | null;
  reason: string;
  inspected_segment?: {
    candidate_id: string;
    start_ms: number;
    end_ms: number;
  } | null;
};

export type CondensedPlannerHistory = {
  latest_valid_action?: PlannerActionHistory | null;
  used_actions?: PlannerActionHistory[];
  invalid_attempts_after_latest_action?: PlannerInvalidAttempt[];
  keyframe_evidence?: PlannerKeyframeEvidence[];
  canonical_evidence?: PlannerCanonicalEvidence[];
  vqa_searched_video_ids?: string[];
  rejected_vqa_answers?: string[];
  wrongly_submitted_candidates?: PlannerWronglySubmittedCandidate[];
};

export type RetrievalLoopResult = {
  status: string;
  encoded_task?: {
    status: string;
    text?: string | null;
  };
  final_candidate_id?: string | null;
  final_answer?: string | null;
  final_submission?: CandidateSubmission | null;
  timings_ms?: Record<string, number>;
  iterations: RetrievalLoopIteration[];
  full_history?: HistoryEntry[];
  condensed_planner_history?: CondensedPlannerHistory;
  [key: string]: unknown;
};
