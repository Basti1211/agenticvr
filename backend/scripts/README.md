# Scripts

## Run imported tasks

Run the batch script from the `backend` directory and pass the name of a directory
under `application/settings/experiment_settings`:

```bash
cd backend
python scripts/run_imported_tasks.py qa_description_100_run1
```

By default, the runner loads every `scripts/vbsYYYY_tasks.json` file. For a quick
smoke test or a specific task file, use:

```bash
python scripts/run_imported_tasks.py qa_description_100_run1 --limit 1
python scripts/run_imported_tasks.py qa_description_100_run1 \
  --tasks-file scripts/vbs2024_tasks.json \
  --task-name vbs24-qa1
```

Use `python scripts/run_imported_tasks.py --help` to see all filters and overrides.
For example, `--max-iterations` overrides `experiments.json`, while
`--max-candidates` overrides `number_of_candidates` for that run.

## Change experiment settings

Each directory in
[`application/settings/experiment_settings`](../application/settings/experiment_settings)
is a self-contained experiment configuration. To create a variant, copy an existing
directory, give it a new name, edit its files, and pass that name to the runner:

```bash
cd backend
cp -R application/settings/experiment_settings/qa_description_100_run1 \
  application/settings/experiment_settings/my_experiment
python scripts/run_imported_tasks.py my_experiment --limit 1
```

The main files are:

- `experiments.json`: task type, dataset, maximum iterations, T-KIS hint selection,
  and AVS feedback visibility.
- `data_layer_settings.json`: data-layer URL, media paths, and candidate count.
- `vlm_api_settings.json`: default VLM endpoint and model. The runner uses this
  default; it has no separate endpoint command-line option.
- `model_settings.json`: model generation and reasoning settings.
- `planner.json`, `browser.json`, `task_encoder.json`, `qa.json`, and
  `answerer.json`: module-specific behavior.
- `task_input_access.json`: which task descriptions and media each module receives.

> **VLLM endpoint required:** The VLLM URL committed in `vlm_api_settings.json` is
> only a dummy placeholder and will not work. Before running an experiment, replace
> `API_url` in the selected experiment directory with your own reachable,
> OpenAI-compatible VLLM endpoint. Also update `model` and `served_model` if their
> values differ on your server.

Example `experiments.json`:

```json
{
  "task_type": "qa",
  "dataset": "all",
  "hint_index": 2,
  "max_iterations": 20,
  "hide_avs_submission_ratings_from_planner": false
}
```

Valid task types are `v-kis`, `t-kis`, `avs`, `qa`, and `all`; valid datasets are
`mvk`, `v3c`, `lhe`, and `all`. For T-KIS, `hint_index` can be `0`, `1`, `2`, or
`"iterative"`.

Logs are written to `backend/task_logs/<setting_name>/`. Completed tasks found in
that directory are skipped on later runs. Use a new experiment name or set
`TASK_RUN_LOG_DIR` to a fresh directory when you want a completely new batch.
