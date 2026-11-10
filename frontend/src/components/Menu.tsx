import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

import { LoopVisualization } from "./retrieval/LoopVisualization";
import { TaskRunLogs } from "./TaskRunLogs";
import type { RetrievalLoopResult } from "../types/retrievalLoop";

type InputType = "text" | "media";
type MediaType = "image" | "video";
type SubmitStatus = "idle" | "submitting" | "success" | "error";
type RunStatus = "idle" | "running" | "success" | "error";
type WorkspaceMode = "logs" | "tasks" | "manual";
type AnswerInputType = "text" | "video";

type TaskConfig = {
  label: string;
  description: string;
  requiredInputs: InputType[];
};

type SelectOption = {
  value: string;
  label: string;
  description?: string;
};

type InputFieldConfig = {
  label: string;
  description: string;
};

type TaskOptionsResponse = {
  task_types: {
    value: string;
    label: string;
    required_inputs: InputType[];
    description: string;
  }[];
  datasets: {
    value: string;
    label: string;
    description: string;
  }[];
};

type VideoSegmentState = {
  name: string;
  startMs: string;
  endMs: string;
};

type MediaPayload = {
  name: string;
  media_type: MediaType;
  mime_type?: string | null;
  size_bytes?: number | null;
};

type TaskPayload = {
  task_type: string;
  dataset: string;
  source_task_name?: string | null;
  input: {
    description?: string | null;
    media?: MediaPayload | null;
    task_video?: ReturnType<typeof toVideoSegment>;
    answer_text?: string | null;
    answer_video?: ReturnType<typeof toVideoSegment>;
  };
};

type StoredTask = {
  id: string;
  status?: string;
  task_input_descriptions?: Record<string, string>;
  encoded_task?: {
    status?: string;
    text?: string | null;
    media_refs?: string[];
    notes?: string[];
    metadata?: {
      llm_calls?: Array<{
        response?: string;
        error?: string;
        [key: string]: unknown;
      }>;
      selected_video_frames?: EncodedVideoFrame[];
      [key: string]: unknown;
    };
  } | null;
  [key: string]: unknown;
};

type EncodedVideoFrame = {
  id: string;
  source?: string;
  timestamp_ms?: number;
};

type ImportedTaskMedia = {
  media_type: MediaType;
  file: string;
  url?: string | null;
  start_ms?: number | null;
  end_ms?: number | null;
};

type ImportedTask = {
  name: string;
  task_type: string;
  dataset: string;
  input_text?: string | null;
  input_media?: ImportedTaskMedia | null;
  target_video?: {
    name: string;
    start_ms?: number | null;
    end_ms?: number | null;
  } | null;
};

const FALLBACK_TASKS: Record<string, TaskConfig> = {
  "v-kis": {
    label: "Visual KIS",
    description:
      "Find the exact scene specified by visual media or a video segment.",
    requiredInputs: ["media"],
  },
  "t-kis": {
    label: "Textual KIS",
    description: "Find the exact scene specified by a textual description.",
    requiredInputs: ["text"],
  },
  qa: {
    label: "Question Answering",
    description:
      "Answer a question, optionally grounded in supplied visual context.",
    requiredInputs: ["text", "media"],
  },
  avs: {
    label: "Ad-hoc Video Search",
    description:
      "Find as many scenes as possible matching the task description.",
    requiredInputs: ["text"],
  },
};

const INPUT_FIELDS: Record<InputType, InputFieldConfig> = {
  text: {
    label: "Text query",
    description: "Describe what you want to search for or ask about.",
  },
  media: {
    label: "Image or video",
    description:
      "Upload a visual file when the selected task needs media input.",
  },
};

const DATASETS: SelectOption[] = [
  { value: "mvk", label: "MVK" },
  { value: "v3c", label: "V3C" },
  { value: "lhe", label: "LHE" },
];

const FALLBACK_TASK_OPTIONS: SelectOption[] = Object.entries(
  FALLBACK_TASKS,
).map(([value, task]) => ({
  value,
  label: task.label,
  description: task.description,
}));

const DEFAULT_TASK_TYPE = "v-kis";
const DEFAULT_DATASET = "mvk";
const LOOP_RUN_STEPS = [1, 10, 20] as const;
const NO_IMPORTED_TASK = "";
const TASK_INPUT_MODULE_LABELS: Record<string, string> = {
  planner: "Planner",
  browser: "Browser",
  vqa: "QA",
};
const MODE_OPTIONS: { value: WorkspaceMode; label: string }[] = [
  { value: "logs", label: "Task logs" },
  { value: "tasks", label: "Search tasks" },
  { value: "manual", label: "Manual task" },
];

const INPUT_IDS: Record<InputType, string> = {
  text: "text-query",
  media: "media-upload",
};

const emptyVideoSegment = (): VideoSegmentState => ({
  name: "",
  startMs: "",
  endMs: "",
});

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return "0 B";

  const units = ["B", "KB", "MB", "GB"];
  const unitIndex = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const size = bytes / 1024 ** unitIndex;

  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

const encodedTaskErrors = (task: StoredTask) =>
  task.encoded_task?.metadata?.llm_calls
    ?.map((call) => call.error)
    .filter(
      (error): error is string =>
        typeof error === "string" && error.trim().length > 0,
    ) ?? [];

const encodedTaskText = (task: StoredTask) => {
  const text = task.encoded_task?.text?.trim();
  if (text) return text;

  const firstResponse = task.encoded_task?.metadata?.llm_calls?.find(
    (call) =>
      typeof call.response === "string" && call.response.trim().length > 0,
  )?.response;
  return firstResponse?.trim() ?? "";
};

type TaskDescriptionGroup = {
  modules: string[];
  text: string;
};

const displayedTaskDescriptions = (
  task: StoredTask,
): TaskDescriptionGroup[] => {
  if (task.task_input_descriptions !== undefined) {
    const modulesByDescription = new Map<string, string[]>();
    Object.entries(task.task_input_descriptions).forEach(([module, value]) => {
      const description = value.trim();
      if (!description) return;
      const modules = modulesByDescription.get(description) ?? [];
      modules.push(TASK_INPUT_MODULE_LABELS[module] ?? module);
      modulesByDescription.set(description, modules);
    });

    return Array.from(modulesByDescription, ([text, modules]) => ({
      modules,
      text,
    }));
  }

  const fallbackText = encodedTaskText(task);
  return fallbackText ? [{ modules: [], text: fallbackText }] : [];
};

const encodedTaskFrames = (task: StoredTask) =>
  task.encoded_task?.metadata?.selected_video_frames?.filter(
    (frame): frame is EncodedVideoFrame =>
      typeof frame.id === "string" && frame.id.trim().length > 0,
  ) ?? [];

const formatTimestamp = (timestampMs?: number) => {
  if (timestampMs == null || !Number.isFinite(timestampMs)) return "";

  const totalSeconds = timestampMs / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
};

const toVideoSegment = (segment: VideoSegmentState) => {
  const name = segment.name.trim();
  if (!name) return null;

  return {
    name,
    start_ms: segment.startMs ? Number(segment.startMs) : null,
    end_ms: segment.endMs ? Number(segment.endMs) : null,
  };
};

const getMediaType = (file: File): MediaType => {
  if (file.type.startsWith("video/")) return "video";
  return "image";
};

const createMediaPayload = (file: File | null): MediaPayload | null => {
  if (!file) return null;

  return {
    name: file.name,
    media_type: getMediaType(file),
    mime_type: file.type || null,
    size_bytes: file.size,
  };
};

const toSegmentState = (
  segment?: ImportedTask["target_video"] | ImportedTaskMedia | null,
): VideoSegmentState => {
  if (!segment) {
    return { name: "", startMs: "", endMs: "" };
  }

  return {
    name: "name" in segment ? segment.name : segment.file,
    startMs: segment.start_ms == null ? "" : String(segment.start_ms),
    endMs: segment.end_ms == null ? "" : String(segment.end_ms),
  };
};

const taskTypeShortLabel = (taskType: string) => {
  const labels: Record<string, string> = {
    "v-kis": "KIS-V",
    "t-kis": "KIS-T",
    qa: "QA",
    avs: "AVS",
  };

  return labels[taskType] ?? taskType.toUpperCase();
};

const importedTaskLabel = (task: ImportedTask) => {
  return `${task.name} (${taskTypeShortLabel(task.task_type)}, ${task.dataset.toUpperCase()})`;
};

const mediaTypeLabel = (mediaType: MediaType) =>
  mediaType === "video" ? "Video input" : "Image input";

const targetAnswerInputs = (selectedTaskType: string): AnswerInputType[] => {
  if (selectedTaskType === "qa") return ["text", "video"];
  if (selectedTaskType === "v-kis" || selectedTaskType === "t-kis")
    return ["video"];
  return [];
};

type SelectFieldProps = {
  label: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
};

const SelectField = ({ label, value, options, onChange }: SelectFieldProps) => (
  <label className="form-field">
    <span>{label}</span>
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  </label>
);

type StatusMessageProps = {
  message: string;
  runStatus: RunStatus;
  submitStatus: SubmitStatus;
};

const StatusMessage = ({
  message,
  runStatus,
  submitStatus,
}: StatusMessageProps) => {
  if (!message) return null;

  const tone =
    runStatus === "error"
      ? "error"
      : runStatus === "success"
        ? "success"
        : submitStatus;

  return <p className={`status-message status-message--${tone}`}>{message}</p>;
};

type LoopRunButtonProps = {
  disabled: boolean;
  isRunning: boolean;
  steps: number;
  onRun: (steps: number) => void;
};

const LoopRunButton = ({
  disabled,
  isRunning,
  steps,
  onRun,
}: LoopRunButtonProps) => (
  <button
    className="secondary-button"
    type="button"
    disabled={disabled}
    onClick={() => onRun(steps)}
  >
    {isRunning
      ? "Running loop..."
      : `Run ${steps} ${steps === 1 ? "step" : "steps"}`}
  </button>
);

type ImportedMediaPreviewProps = {
  media?: ImportedTaskMedia | null;
};

const ImportedMediaPreview = ({ media }: ImportedMediaPreviewProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  if (!media?.url) {
    return null;
  }

  const startSeconds = (media.start_ms ?? 0) / 1000;

  if (media.media_type === "image") {
    return (
      <figure className="imported-media-preview">
        <img src={media.url} alt={`Imported task input ${media.file}`} />
      </figure>
    );
  }

  return (
    <figure className="imported-media-preview">
      <video
        ref={videoRef}
        controls
        muted
        preload="metadata"
        src={`${media.url}#t=${startSeconds}`}
        onLoadedMetadata={() => {
          if (videoRef.current) {
            videoRef.current.currentTime = startSeconds;
          }
        }}
      />
    </figure>
  );
};

type EncodedFrameSelectionProps = {
  taskId: string;
  frames: EncodedVideoFrame[];
};

const EncodedFrameSelection = ({
  taskId,
  frames,
}: EncodedFrameSelectionProps) => {
  if (!frames.length) return null;

  return (
    <section
      className="encoded-frame-selection"
      aria-label="Encoder selected video frames"
    >
      <h3>Selected frames</h3>
      <div className="encoded-frame-grid">
        {frames.map((frame) => {
          const timestamp = formatTimestamp(frame.timestamp_ms);

          return (
            <figure className="encoded-frame-card" key={frame.id}>
              <img
                src={`/api/tasks/${taskId}/frames/${encodeURIComponent(frame.id)}`}
                alt={
                  timestamp
                    ? `Selected encoder frame at ${timestamp}`
                    : "Selected encoder frame"
                }
                loading="lazy"
              />
              <figcaption>
                {timestamp ? <strong>{timestamp}</strong> : null}
                {frame.source ? <span>{frame.source}</span> : null}
              </figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
};

type MenuProps = {
  onTaskTypeChange: (taskType: string) => void;
  onDatasetChange: (dataset: string) => void;
};

export const Menu = ({ onTaskTypeChange, onDatasetChange }: MenuProps) => {
  const [activeMode, setActiveMode] = useState<WorkspaceMode>("logs");
  const [taskType, setTaskType] = useState(DEFAULT_TASK_TYPE);
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const [taskConfigs, setTaskConfigs] = useState(FALLBACK_TASKS);
  const [taskOptions, setTaskOptions] = useState(FALLBACK_TASK_OPTIONS);
  const [datasetOptions, setDatasetOptions] = useState(DATASETS);
  const [importedTasks, setImportedTasks] = useState<ImportedTask[]>([]);
  const [selectedImportedTaskName, setSelectedImportedTaskName] =
    useState(NO_IMPORTED_TASK);
  const [importedTaskSearch, setImportedTaskSearch] = useState("");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const filePreviewUrlRef = useRef<string | null>(null);
  const [taskVideo, setTaskVideo] =
    useState<VideoSegmentState>(emptyVideoSegment);
  const [answerText, setAnswerText] = useState("");
  const [answerVideo, setAnswerVideo] =
    useState<VideoSegmentState>(emptyVideoSegment);
  const [submitStatus, setSubmitStatus] = useState<SubmitStatus>("idle");
  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [createdTask, setCreatedTask] = useState<StoredTask | null>(null);
  const [loopResult, setLoopResult] = useState<RetrievalLoopResult | null>(
    null,
  );
  const [showTargetAnswerFields, setShowTargetAnswerFields] = useState(false);

  const selectedTask = taskConfigs[taskType];
  const selectedInputs = selectedTask?.requiredInputs ?? [];
  const availableTargetAnswerInputs = targetAnswerInputs(taskType);
  const selectedImportedTask = useMemo(
    () =>
      importedTasks.find((task) => task.name === selectedImportedTaskName) ??
      null,
    [importedTasks, selectedImportedTaskName],
  );
  const activePreviewMedia = useMemo<ImportedTaskMedia | null>(() => {
    if (selectedImportedTask?.input_media)
      return selectedImportedTask.input_media;
    if (!file || !filePreviewUrl) return null;

    return {
      media_type: getMediaType(file),
      file: file.name,
      url: filePreviewUrl,
    };
  }, [file, filePreviewUrl, selectedImportedTask]);
  const filteredImportedTasks = useMemo(() => {
    const search = importedTaskSearch.trim().toLowerCase();
    if (!search) return importedTasks;

    return importedTasks.filter((task) =>
      [
        task.name,
        task.task_type,
        task.dataset,
        task.input_text ?? "",
        task.input_media?.file ?? "",
      ]
        .join(" ")
        .toLowerCase()
        .includes(search),
    );
  }, [importedTaskSearch, importedTasks]);
  const importedTaskOptions = useMemo(() => {
    const visibleTasks = selectedImportedTask
      ? [
          selectedImportedTask,
          ...filteredImportedTasks.filter(
            (task) => task.name !== selectedImportedTask.name,
          ),
        ]
      : filteredImportedTasks;

    return [
      { value: NO_IMPORTED_TASK, label: "Select a downloaded task" },
      ...visibleTasks.map((task) => ({
        value: task.name,
        label: importedTaskLabel(task),
      })),
    ];
  }, [filteredImportedTasks, selectedImportedTask]);
  const storedTaskDescriptions = createdTask
    ? displayedTaskDescriptions(createdTask)
    : [];
  const storedRepresentationErrors = createdTask
    ? encodedTaskErrors(createdTask)
    : [];
  const storedRepresentationFrames = createdTask
    ? encodedTaskFrames(createdTask)
    : [];
  const canSubmitTask =
    activeMode === "manual" || selectedImportedTask !== null;

  useEffect(() => {
    const abortController = new AbortController();

    const loadOptions = async () => {
      try {
        const response = await fetch("/api/tasks/options", {
          signal: abortController.signal,
        });
        if (!response.ok) return;

        const options = (await response.json()) as TaskOptionsResponse;
        const nextTaskConfigs = options.task_types.reduce<
          Record<string, TaskConfig>
        >(
          (configs, task) => ({
            ...configs,
            [task.value]: {
              label: task.label,
              description: task.description,
              requiredInputs: task.required_inputs,
            },
          }),
          {},
        );

        setTaskConfigs(nextTaskConfigs);
        setTaskOptions(
          options.task_types.map((task) => ({
            value: task.value,
            label: task.label,
            description: task.description,
          })),
        );
        setDatasetOptions(
          options.datasets.map((option) => ({
            value: option.value,
            label: option.label,
            description: option.description,
          })),
        );
      } catch {
        if (!abortController.signal.aborted) {
          setStatusMessage(
            "Using local task options until the backend is reachable.",
          );
        }
      }
    };

    loadOptions();

    return () => abortController.abort();
  }, []);

  useEffect(() => {
    const abortController = new AbortController();

    const loadImportedTasks = async () => {
      try {
        const response = await fetch("/api/tasks/imported", {
          signal: abortController.signal,
        });
        if (!response.ok) return;

        setImportedTasks((await response.json()) as ImportedTask[]);
      } catch {
        if (!abortController.signal.aborted) {
          setStatusMessage(
            "Downloaded task options are unavailable until the backend is reachable.",
          );
        }
      }
    };

    loadImportedTasks();

    return () => abortController.abort();
  }, []);

  useEffect(
    () => () => {
      if (filePreviewUrlRef.current) {
        URL.revokeObjectURL(filePreviewUrlRef.current);
      }
    },
    [],
  );

  const clearLoopState = () => {
    setCreatedTask(null);
    setLoopResult(null);
    setRunStatus("idle");
  };

  const replaceSelectedFile = (nextFile: File | null) => {
    if (filePreviewUrlRef.current) {
      URL.revokeObjectURL(filePreviewUrlRef.current);
      filePreviewUrlRef.current = null;
    }

    if (nextFile) {
      const objectUrl = URL.createObjectURL(nextFile);
      filePreviewUrlRef.current = objectUrl;
      setFilePreviewUrl(objectUrl);
    } else {
      setFilePreviewUrl(null);
    }

    setFile(nextFile);
  };

  const handleModeChange = (nextMode: WorkspaceMode) => {
    setActiveMode(nextMode);
    setStatusMessage("");
    setShowTargetAnswerFields(false);

    if (nextMode === "manual") {
      clearLoopState();
      setSelectedImportedTaskName(NO_IMPORTED_TASK);
      setAnswerText("");
      setAnswerVideo(emptyVideoSegment());
    }
  };

  const handleTaskTypeChange = (selectedTaskType: string) => {
    clearLoopState();
    setSelectedImportedTaskName(NO_IMPORTED_TASK);
    setTaskType(selectedTaskType);
    setShowTargetAnswerFields(false);
    setAnswerText("");
    setAnswerVideo(emptyVideoSegment());
    onTaskTypeChange(selectedTaskType);
  };

  const handleDatasetChange = (selectedDataset: string) => {
    clearLoopState();
    setSelectedImportedTaskName(NO_IMPORTED_TASK);
    setDataset(selectedDataset);
    onDatasetChange(selectedDataset);
  };

  const handleImportedTaskChange = (taskName: string) => {
    clearLoopState();
    setSelectedImportedTaskName(taskName);
    setShowTargetAnswerFields(false);

    const importedTask = importedTasks.find((task) => task.name === taskName);
    if (!importedTask) {
      setQuery("");
      setTaskVideo(emptyVideoSegment());
      setAnswerVideo(emptyVideoSegment());
      setAnswerText("");
      return;
    }

    setTaskType(importedTask.task_type);
    setDataset(importedTask.dataset);
    setQuery(importedTask.input_text ?? "");
    setTaskVideo(
      importedTask.input_media?.media_type === "video"
        ? toSegmentState(importedTask.input_media)
        : emptyVideoSegment(),
    );
    setAnswerVideo(toSegmentState(importedTask.target_video));
    setAnswerText("");
    replaceSelectedFile(null);
    onTaskTypeChange(importedTask.task_type);
    onDatasetChange(importedTask.dataset);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    clearLoopState();
    replaceSelectedFile(event.target.files?.[0] ?? null);
  };

  const handleVideoSegmentChange = (
    segmentName: "taskVideo" | "answerVideo",
    key: keyof VideoSegmentState,
    value: string,
  ) => {
    clearLoopState();
    const setter = segmentName === "taskVideo" ? setTaskVideo : setAnswerVideo;
    setter((current) => ({ ...current, [key]: value }));
  };

  const applyTargetAnswerPayload = (
    inputPayload: TaskPayload["input"],
    payloadTaskType: string,
  ) => {
    const answers = targetAnswerInputs(payloadTaskType);

    if (answers.includes("text")) {
      const text = answerText.trim();
      if (text) inputPayload.answer_text = text;
    }

    if (answers.includes("video")) {
      const segment = toVideoSegment(answerVideo);
      if (segment) inputPayload.answer_video = segment;
    }
  };

  const buildImportedTaskPayload = (
    importedTask: ImportedTask,
  ): TaskPayload => {
    const inputPayload: TaskPayload["input"] = {};
    const description = importedTask.input_text?.trim();

    if (description) {
      inputPayload.description = description;
    }

    if (importedTask.input_media?.media_type === "video") {
      inputPayload.task_video = toVideoSegment(
        toSegmentState(importedTask.input_media),
      );
    }

    if (
      importedTask.input_media?.media_type === "image" &&
      importedTask.input_media.url
    ) {
      inputPayload.media = {
        name: importedTask.input_media.url,
        media_type: "image",
        mime_type: null,
        size_bytes: null,
      };
    }

    applyTargetAnswerPayload(inputPayload, importedTask.task_type);

    return {
      task_type: importedTask.task_type,
      dataset: importedTask.dataset,
      source_task_name: importedTask.name,
      input: inputPayload,
    };
  };

  const buildPayload = (): TaskPayload => {
    if (activeMode === "tasks" && selectedImportedTask) {
      return buildImportedTaskPayload(selectedImportedTask);
    }

    const inputPayload: TaskPayload["input"] = {};

    if (selectedInputs.includes("text")) {
      inputPayload.description = query.trim() || null;
    }

    if (selectedInputs.includes("media")) {
      inputPayload.media = createMediaPayload(file);
    }

    if (taskType === "v-kis") {
      const segment = toVideoSegment(taskVideo);
      if (segment) inputPayload.task_video = segment;
    }

    applyTargetAnswerPayload(inputPayload, taskType);

    return {
      task_type: taskType,
      dataset,
      input: inputPayload,
    };
  };

  const readErrorMessage = async (response: Response, fallback: string) => {
    const errorBody = await response.json().catch(() => null);
    const detail = errorBody?.detail;
    if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
    if (typeof detail === "string") return detail;
    return fallback;
  };

  const postTaskPayload = (payload: TaskPayload) =>
    fetch("/api/tasks", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

  const uploadTaskPayload = (payload: TaskPayload, mediaFile: File) => {
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    formData.append("media", mediaFile);

    return fetch("/api/tasks/upload", {
      method: "POST",
      body: formData,
    });
  };

  const storeTask = async (payload: TaskPayload): Promise<StoredTask> => {
    const shouldUploadMedia =
      activeMode === "manual" &&
      file !== null &&
      payload.input.media?.name === file.name;
    const response = shouldUploadMedia
      ? await uploadTaskPayload(payload, file)
      : await postTaskPayload(payload);

    if (!response.ok) {
      throw new Error(await readErrorMessage(response, "Task upload failed."));
    }

    return (await response.json()) as StoredTask;
  };

  const getOrStoreTask = async (payload: TaskPayload): Promise<StoredTask> => {
    if (createdTask) return createdTask;

    const storedTask = await storeTask(payload);
    setCreatedTask(storedTask);
    return storedTask;
  };

  const runLoop = async (
    payload: TaskPayload,
    maxIterations: number,
  ): Promise<RetrievalLoopResult> => {
    const storedTask = await getOrStoreTask(payload);
    const response = await fetch(
      `/api/tasks/${storedTask.id}/run?max_iterations=${maxIterations}`,
      {
        method: "POST",
      },
    );

    if (!response.ok) {
      throw new Error(
        await readErrorMessage(response, "Retrieval loop failed."),
      );
    }

    return (await response.json()) as RetrievalLoopResult;
  };

  const continueLoop = async (
    maxIterations: number,
  ): Promise<RetrievalLoopResult> => {
    if (!loopResult) {
      throw new Error("Run the retrieval loop before continuing.");
    }

    const storedTask = await getOrStoreTask(buildPayload());
    const response = await fetch(
      `/api/tasks/${storedTask.id}/continue?max_iterations=${maxIterations}`,
      {
        method: "POST",
      },
    );

    if (!response.ok) {
      throw new Error(
        await readErrorMessage(response, "Loop continuation failed."),
      );
    }

    return (await response.json()) as RetrievalLoopResult;
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmitTask) {
      setSubmitStatus("error");
      setStatusMessage(
        "Select a task or switch to manual setup before storing.",
      );
      return;
    }

    setSubmitStatus("submitting");
    setStatusMessage("");
    setCreatedTask(null);
    setLoopResult(null);

    try {
      const storedTask = await storeTask(buildPayload());
      setCreatedTask(storedTask);
      setSubmitStatus("success");
      setStatusMessage(`Stored task ${storedTask.id}`);
    } catch (error) {
      setSubmitStatus("error");
      setStatusMessage(
        error instanceof Error ? error.message : "Task upload failed.",
      );
    }
  };

  const handleRunLoop = async (maxIterations: number) => {
    if (!canSubmitTask) {
      setRunStatus("error");
      setStatusMessage(
        "Select a task or switch to manual setup before running.",
      );
      return;
    }

    setRunStatus("running");
    setSubmitStatus("idle");
    setStatusMessage("");
    setLoopResult(null);

    try {
      const result = await runLoop(buildPayload(), maxIterations);
      setLoopResult(result);
      setRunStatus("success");
      setStatusMessage(
        result.status === "found"
          ? `Found ${result.final_answer ?? result.final_candidate_id ?? "a matching candidate"}`
          : `Loop ran ${result.iterations.length} steps with status ${result.status}`,
      );
    } catch (error) {
      setRunStatus("error");
      setStatusMessage(
        error instanceof Error ? error.message : "Retrieval loop failed.",
      );
    }
  };

  const handleContinueLoop = async (maxIterations: number) => {
    setRunStatus("running");
    setSubmitStatus("idle");
    setStatusMessage(
      `Continuing loop for up to ${maxIterations} more steps...`,
    );

    try {
      const result = await continueLoop(maxIterations);
      setLoopResult(result);
      setRunStatus("success");
      setStatusMessage(
        result.status === "found"
          ? `Found ${result.final_answer ?? result.final_candidate_id ?? "a matching candidate"}`
          : `Loop continued to ${result.iterations.length} steps with status ${result.status}`,
      );
    } catch (error) {
      setRunStatus("error");
      setStatusMessage(
        error instanceof Error ? error.message : "Loop continuation failed.",
      );
    }
  };

  const renderInputField = (inputType: InputType) => {
    const field = INPUT_FIELDS[inputType];
    const inputId = INPUT_IDS[inputType];

    return (
      <article className="input-card" key={inputType}>
        <div className="input-card__header">
          <label className="field-label" htmlFor={inputId}>
            {field.label}
          </label>
          <p>{field.description}</p>
        </div>

        {inputType === "text" ? (
          <textarea
            id={inputId}
            className="text-input"
            value={query}
            onChange={(event) => {
              clearLoopState();
              setQuery(event.target.value);
            }}
            placeholder="Enter your query..."
            rows={5}
          />
        ) : (
          <>
            <label className="file-dropzone" htmlFor={inputId}>
              <input
                id={inputId}
                className="sr-only"
                type="file"
                accept="image/*,video/*"
                onChange={handleFileChange}
              />
              <span className="file-dropzone__icon" aria-hidden="true">
                +
              </span>
              <span>
                <strong>{file ? file.name : "Choose an image or video"}</strong>
                <small>
                  {file
                    ? formatFileSize(file.size)
                    : "PNG, JPG, GIF, MP4, MOV, or WebM"}
                </small>
              </span>
            </label>
          </>
        )}
      </article>
    );
  };

  const renderVideoSegmentField = (
    segmentName: "taskVideo" | "answerVideo",
    segment: VideoSegmentState,
    legend: string,
  ) => (
    <fieldset className="segment-fieldset">
      <legend>{legend}</legend>
      <input
        type="text"
        value={segment.name}
        onChange={(event) =>
          handleVideoSegmentChange(segmentName, "name", event.target.value)
        }
        placeholder="Video name or id"
      />
      <input
        type="number"
        min="0"
        value={segment.startMs}
        onChange={(event) =>
          handleVideoSegmentChange(segmentName, "startMs", event.target.value)
        }
        placeholder="Start ms"
      />
      <input
        type="number"
        min="0"
        value={segment.endMs}
        onChange={(event) =>
          handleVideoSegmentChange(segmentName, "endMs", event.target.value)
        }
        placeholder="End ms"
      />
    </fieldset>
  );

  const renderTargetAnswerFields = () => {
    if (!availableTargetAnswerInputs.length) return null;

    return (
      <div className="input-panel" aria-label="Optional target answer">
        <div className="input-panel__header">
          <h2>Target answer</h2>
          <p>
            Optional ground truth used by the loop to validate final
            submissions.
          </p>
        </div>

        {!showTargetAnswerFields ? (
          <button
            className="secondary-button secondary-button--fit"
            type="button"
            onClick={() => setShowTargetAnswerFields(true)}
          >
            Add target answer
          </button>
        ) : (
          <div className="input-stack">
            {availableTargetAnswerInputs.includes("video")
              ? renderVideoSegmentField(
                  "answerVideo",
                  answerVideo,
                  "Target answer video",
                )
              : null}

            {availableTargetAnswerInputs.includes("text") ? (
              <label className="form-field">
                <span>Answer text</span>
                <input
                  type="text"
                  value={answerText}
                  onChange={(event) => {
                    clearLoopState();
                    setAnswerText(event.target.value);
                  }}
                  placeholder="Expected answer"
                />
              </label>
            ) : null}
          </div>
        )}
      </div>
    );
  };

  const renderSubmitRow = () => {
    const actionDisabled =
      runStatus === "running" ||
      submitStatus === "submitting" ||
      !canSubmitTask;

    return (
      <div className="submit-row">
        <button
          className="initial-button"
          type="submit"
          disabled={submitStatus === "submitting" || !canSubmitTask}
        >
          {submitStatus === "submitting" ? "Storing task..." : "Store task"}
        </button>
        {LOOP_RUN_STEPS.map((steps) => (
          <LoopRunButton
            key={steps}
            disabled={actionDisabled}
            isRunning={runStatus === "running"}
            steps={steps}
            onRun={handleRunLoop}
          />
        ))}
        <StatusMessage
          message={statusMessage}
          runStatus={runStatus}
          submitStatus={submitStatus}
        />
      </div>
    );
  };

  return (
    <section className="task-selector" aria-labelledby="task-selector-title">
      <header className="task-selector__header">
        <p className="eyebrow">Task setup</p>
        <h1 id="task-selector-title">Automated PraK</h1>
        <p>
          {activeMode === "logs"
            ? "Inspect previous task runs."
            : activeMode === "manual"
              ? selectedTask?.description
              : "Search downloaded tasks or switch to manual setup."}
        </p>
      </header>

      <nav className="mode-switcher" aria-label="Task workspace">
        {MODE_OPTIONS.map((mode) => (
          <button
            key={mode.value}
            className={`mode-switcher__button${
              activeMode === mode.value ? " mode-switcher__button--active" : ""
            }`}
            type="button"
            aria-pressed={activeMode === mode.value}
            onClick={() => handleModeChange(mode.value)}
          >
            {mode.label}
          </button>
        ))}
      </nav>

      {activeMode === "logs" ? (
        <TaskRunLogs />
      ) : (
        <form className="task-form" onSubmit={handleSubmit}>
          {activeMode === "tasks" ? (
            <>
              <div className="selector-card">
                <div className="task-search-controls">
                  <label className="form-field">
                    <span>Search downloaded tasks</span>
                    <input
                      type="search"
                      value={importedTaskSearch}
                      onChange={(event) =>
                        setImportedTaskSearch(event.target.value)
                      }
                      placeholder="Task name, type, dataset, or text"
                    />
                  </label>

                  <SelectField
                    label="Downloaded task"
                    value={selectedImportedTaskName}
                    options={importedTaskOptions}
                    onChange={handleImportedTaskChange}
                  />
                </div>

                {selectedImportedTask ? (
                  <aside
                    className={`imported-task-summary imported-task-summary--readonly${
                      selectedImportedTask.input_media
                        ? ""
                        : " imported-task-summary--text-only"
                    }`}
                    aria-label="Selected task preview"
                  >
                    <div>
                      <span className="summary-kicker">
                        {taskTypeShortLabel(selectedImportedTask.task_type)} ·{" "}
                        {selectedImportedTask.dataset.toUpperCase()}
                      </span>
                      <strong>{selectedImportedTask.name}</strong>
                      {selectedImportedTask.input_text ? (
                        <p>{selectedImportedTask.input_text}</p>
                      ) : selectedImportedTask.input_media ? (
                        <p>
                          {mediaTypeLabel(
                            selectedImportedTask.input_media.media_type,
                          )}{" "}
                          · {selectedImportedTask.input_media.file}
                        </p>
                      ) : (
                        <p>No input preview is available for this task.</p>
                      )}
                    </div>
                    <ImportedMediaPreview
                      media={selectedImportedTask.input_media}
                    />
                  </aside>
                ) : (
                  <p className="input-panel__empty">
                    Select a downloaded task to preview its text or media.
                  </p>
                )}
              </div>

              {selectedImportedTask ? renderTargetAnswerFields() : null}
              {renderSubmitRow()}
            </>
          ) : (
            <>
              <div className="selector-card">
                <div className="selector-form">
                  <SelectField
                    label="Task type"
                    value={taskType}
                    options={taskOptions}
                    onChange={handleTaskTypeChange}
                  />

                  <SelectField
                    label="Dataset"
                    value={dataset}
                    options={datasetOptions}
                    onChange={handleDatasetChange}
                  />
                </div>
              </div>

              <div
                className="input-panel"
                aria-label={`${selectedTask?.label ?? "Selected task"} inputs`}
              >
                <div className="input-panel__header">
                  <h2>Inputs</h2>
                  <p>Only the inputs needed for the selected task are shown.</p>
                </div>

                <div className="input-stack">
                  {selectedInputs.map(renderInputField)}
                  {taskType === "v-kis"
                    ? renderVideoSegmentField(
                        "taskVideo",
                        taskVideo,
                        "Existing task video",
                      )
                    : null}
                </div>
              </div>

              {renderTargetAnswerFields()}
              {renderSubmitRow()}
            </>
          )}
        </form>
      )}

      {activeMode !== "logs" && createdTask ? (
        <section
          className="debug-panel"
          aria-label="Stored task representation"
        >
          <h2>Stored representation</h2>
          <div className="stored-representation">
            <dl className="stored-representation__meta">
              <div>
                <dt>Task status</dt>
                <dd>{createdTask.status ?? "stored"}</dd>
              </div>
              <div>
                <dt>Encoder status</dt>
                <dd>{createdTask.encoded_task?.status ?? "not encoded"}</dd>
              </div>
            </dl>

            {storedTaskDescriptions.length ||
            storedRepresentationFrames.length ? (
              <section
                className={
                  storedRepresentationFrames.length || activePreviewMedia
                    ? "encoded-preview-layout"
                    : undefined
                }
              >
                {storedRepresentationFrames.length ? (
                  <EncodedFrameSelection
                    taskId={createdTask.id}
                    frames={storedRepresentationFrames}
                  />
                ) : activePreviewMedia ? (
                  <ImportedMediaPreview media={activePreviewMedia} />
                ) : null}
                {storedTaskDescriptions.length ? (
                  <div className="stored-task-descriptions">
                    <h3>
                      {storedTaskDescriptions.length === 1
                        ? "Task description"
                        : "Task descriptions"}
                    </h3>
                    {storedTaskDescriptions.map((description) => (
                      <div
                        className="stored-task-description"
                        key={`${description.modules.join("-")}-${description.text}`}
                      >
                        {storedTaskDescriptions.length > 1 ? (
                          <h4>{description.modules.join(", ")}</h4>
                        ) : null}
                        <p>{description.text}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="stored-representation__empty">
                    No configured task description is available.
                  </p>
                )}
              </section>
            ) : (
              <p className="stored-representation__empty">
                No configured task description is available.
              </p>
            )}

            {storedRepresentationErrors.length ? (
              <section className="stored-representation__errors">
                <h3>Errors</h3>
                {storedRepresentationErrors.map((error, index) => (
                  <p key={`${index}-${error}`}>{error}</p>
                ))}
              </section>
            ) : null}

            {createdTask.encoded_task?.notes?.length ? (
              <details>
                <summary>Notes</summary>
                <ul>
                  {createdTask.encoded_task.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </details>
            ) : null}

            <details>
              <summary>Show full stored task</summary>
              <pre>{JSON.stringify(createdTask, null, 2)}</pre>
            </details>
          </div>
        </section>
      ) : null}

      {activeMode !== "logs" && loopResult ? (
        <LoopVisualization
          allowContinueAfterFound={taskType === "avs"}
          dataset={dataset}
          isContinuing={runStatus === "running"}
          onContinue={handleContinueLoop}
          result={loopResult}
        />
      ) : null}
    </section>
  );
};
