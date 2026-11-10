# Can Agents Win The Video Browser Showdown?

This repository contains the publication code associated with our submission,
*Can Agents Win The Video Browser Showdown?*, to the International Conference
on Multimedia Modelling.

## Abstract

Searching large video collections is typically an interactive process in which
users play two roles. First, they hold the search intent: the underlying goal
that determines what content they seek and why. Second, users must
operationalize this intent through an iterative search loop. Users translate
their intent into queries, browse the retrieved candidates, and refine their
queries based on the results. In this paper, we investigate the capabilities of
modern Vision Language Models (VLM) and agentic approaches to reach search
goals interactively and fully autonomously. Specifically, we study whether a
provided initial specification of a search goal might be sufficient to solve
traditionally interactive search tasks with an agentic system. Provided that
the involved VLMs are not aware of the whole large video dataset in advance,
the key challenge lies in the effective combination of an existing interactive
video search system and a smart VLM agent controlling the system. While the
search system provides indexing and efficient querying, the VLM-based agents
analyze top-ranked items and make decisions about next actions. Our results
show that modern agents can autonomously operate interactive video retrieval
systems to solve many search tasks from an initial intent description,
achieving performance competitive with strong historical expert-operated
systems in several settings.

The application consists of a FastAPI backend and a React frontend and is run
with Docker Compose.

## Run the application

The only prerequisite is a working Docker installation with Docker Compose.
From the repository root, build and start the application with:

```bash
docker compose up --build
```

Once both containers are running, open the frontend at
[http://localhost:5173](http://localhost:5173). The backend is available on port
`5001`.

To start the containers in the background, use:

```bash
docker compose up --build -d
```

To stop the application, run:

```bash
docker compose down
```

## Annotations

Human annotations can be found in [`backend/annotations`](backend/annotations).

## Inspect task logs

Experiment runs write one JSON log per task to
[`backend/task_logs`](backend/task_logs). Subdirectories group the logs by
experiment setting.

To inspect a log:

1. Start the application and open [http://localhost:5173](http://localhost:5173).
2. Select **Logs** in the frontend.
3. Choose a run from the **Task log** list. The frontend automatically lists the
   logs stored under `backend/task_logs`, including its subdirectories.

The log view shows the task configuration, retrieval iterations, submissions,
errors, timing information, and the full run JSON. A standalone task-log JSON
file can also be opened with **Choose JSON file**.

## Configure a vLLM endpoint and run experiments

Running experiments requires access to your own reachable, OpenAI-compatible
vLLM endpoint. Providing this endpoint is the only project-specific setup needed
before running an experiment; the URL committed to the repository is a
placeholder.

Each experiment has a self-contained configuration directory under
[`backend/application/settings/experiment_settings`](backend/application/settings/experiment_settings).
Before running an experiment, open its `vlm_api_settings.json` and update the
`VLLM` entry:

```json
{
  "default_endpoint": "VLLM",
  "endpoints": {
    "VLLM": {
      "API_url": "https://your-vllm-host.example.com/v1",
      "model": "your-model-name",
      "served_model": "your-served-model-name",
      "parallel_requests": 2
    }
  }
}
```

Set `API_url` to the endpoint reachable from the backend container. If your
server uses different model identifiers, also update `model` and `served_model`
to match it.

With the application running, execute an included experiment inside the backend
container:

```bash
docker compose exec backend python scripts/run_imported_tasks.py <experiment-name>
```

For example:

```bash
docker compose exec backend python scripts/run_imported_tasks.py qa_description_100_run1
```

Use `--limit 1` for a quick test run:

```bash
docker compose exec backend python scripts/run_imported_tasks.py qa_description_100_run1 --limit 1
```

The default configuration used for tasks started through the frontend is
[`backend/application/settings/vlm_api_settings.json`](backend/application/settings/vlm_api_settings.json).
Configure its `VLLM` entry in the same way when using the frontend to start new
tasks.

## Paper experiment results

The experiment results reported in the paper are available in [`supplementary_results`](supplementary_results).
The individual hyperparameters used for each experiment can be found in
[`backend/application/settings/experiment_settings`](backend/application/settings/experiment_settings).
