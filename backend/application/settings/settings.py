import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


SETTINGS_DIR = Path(__file__).parent
EXPERIMENT_SETTINGS_DIR = SETTINGS_DIR / "experiment_settings"
MODEL_SETTINGS_FILE = SETTINGS_DIR / "model_settings.json"
VLM_API_SETTINGS_FILE = SETTINGS_DIR / "vlm_api_settings.json"
DATA_LAYER_SETTINGS_FILE = SETTINGS_DIR / "data_layer_settings.json"
TASK_ENCODER_SETTINGS_FILE = SETTINGS_DIR / "task_encoder.json"
PLANNER_SETTINGS_FILE = SETTINGS_DIR / "planner.json"
ANSWERER_SETTINGS_FILE = SETTINGS_DIR / "answerer.json"
BROWSER_SETTINGS_FILE = SETTINGS_DIR / "browser.json"
QA_SETTINGS_FILE = SETTINGS_DIR / "qa.json"
TASK_INPUT_ACCESS_SETTINGS_FILE = SETTINGS_DIR / "task_input_access.json"
LOGGING_SETTINGS_FILE = SETTINGS_DIR / "logging.json"
_SETTINGS_DIR_OVERRIDE: ContextVar[Path | None] = ContextVar(
    "settings_dir_override",
    default=None,
)
_VLM_ENDPOINT: ContextVar[str | None] = ContextVar(
    "vlm_endpoint",
    default=None,
)
_VLM_API_URL_OVERRIDE: ContextVar[tuple[str, str] | None] = ContextVar(
    "vlm_api_url_override",
    default=None,
)


def _model_settings(model: str | None = None) -> dict[str, Any]:
    model_name = model or _vlm_api_settings()["model"]
    settings_by_model = _read_json_settings(_settings_file("model_settings.json"))
    try:
        return settings_by_model[model_name]
    except KeyError as error:
        raise ValueError(f"No settings configured for model {model_name!r}.") from error


def _vlm_endpoint() -> str:
    selected_endpoint = _VLM_ENDPOINT.get()
    if selected_endpoint is not None:
        return selected_endpoint

    settings = _read_json_settings(_settings_file("vlm_api_settings.json"))
    default_endpoint = settings.get("default_endpoint")
    if not isinstance(default_endpoint, str):
        raise ValueError("VLM API settings must define a default_endpoint.")
    return default_endpoint.upper()


def _vlm_api_settings(endpoint: str | None = None) -> dict[str, Any]:
    settings = _read_json_settings(_settings_file("vlm_api_settings.json"))
    endpoint_name = (endpoint or _vlm_endpoint()).upper()
    try:
        endpoint_settings = settings["endpoints"][endpoint_name]
    except KeyError as error:
        raise ValueError(f"Unknown VLM endpoint {endpoint_name!r}.") from error

    api_url_override = _VLM_API_URL_OVERRIDE.get()
    if api_url_override is None or api_url_override[0] != endpoint_name:
        return endpoint_settings
    return {**endpoint_settings, "API_url": api_url_override[1]}


@contextmanager
def use_vlm_endpoint(
    endpoint: str | None,
    *,
    api_url: str | None = None,
) -> Iterator[None]:
    """Select a configured VLM endpoint and optional URL for this async context."""

    if endpoint is None:
        if api_url is not None:
            raise ValueError("A VLM API URL override requires an endpoint.")
        yield
        return

    endpoint_name = endpoint.upper()
    _vlm_api_settings(endpoint_name)
    endpoint_token = _VLM_ENDPOINT.set(endpoint_name)
    api_url_token = (
        _VLM_API_URL_OVERRIDE.set((endpoint_name, api_url))
        if api_url is not None
        else None
    )
    try:
        yield
    finally:
        if api_url_token is not None:
            _VLM_API_URL_OVERRIDE.reset(api_url_token)
        _VLM_ENDPOINT.reset(endpoint_token)


def experiment_settings_dir(setting_name: str) -> Path:
    """Return a validated experiment settings directory."""

    if (
        not setting_name
        or Path(setting_name).name != setting_name
        or setting_name in {".", ".."}
    ):
        raise ValueError(
            "Setting name must be a directory name such as 'experiment_1'."
        )

    settings_dir = EXPERIMENT_SETTINGS_DIR / setting_name
    if not settings_dir.is_dir():
        raise ValueError(f"Unknown experiment setting {setting_name!r}.")
    return settings_dir


@contextmanager
def use_experiment_settings(setting_name: str) -> Iterator[Path]:
    """Use one named experiment settings directory in the current context."""

    settings_dir = experiment_settings_dir(setting_name)
    token = _SETTINGS_DIR_OVERRIDE.set(settings_dir)
    try:
        yield settings_dir
    finally:
        _SETTINGS_DIR_OVERRIDE.reset(token)


def _data_layer_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("data_layer_settings.json"))


def retrieval_candidate_count(
    settings: Mapping[str, Any] | None = None,
) -> int:
    """Return the configured number of candidates in one retrieval page."""

    configured = _data_layer_settings() if settings is None else settings
    try:
        candidate_count = int(configured["number_of_candidates"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Data-layer settings must define a positive integer number_of_candidates."
        ) from error
    if candidate_count < 1:
        raise ValueError(
            "Data-layer settings must define a positive integer number_of_candidates."
        )
    return candidate_count


def _task_encoder_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("task_encoder.json"))


def _planner_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("planner.json"))


def _answerer_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("answerer.json"))


def _browser_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("browser.json"))


def _qa_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("qa.json"))


def _task_input_access_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("task_input_access.json"))


def _logging_settings() -> dict[str, Any]:
    return _read_json_settings(_settings_file("logging.json"))


def _settings_file(file_name: str) -> Path:
    return (_SETTINGS_DIR_OVERRIDE.get() or SETTINGS_DIR) / file_name


def _read_json_settings(settings_file: Path) -> dict[str, Any]:
    with settings_file.open(encoding="utf-8") as file:
        return json.load(file)
