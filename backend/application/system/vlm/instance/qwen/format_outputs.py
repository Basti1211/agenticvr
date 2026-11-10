from typing import Any


def ollama_assistant_content(response: dict[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Qwen response did not contain a message: {response}")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Qwen response did not contain text content: {response}")
    return content


def openai_assistant_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Qwen response did not contain choices: {response}")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"Qwen response did not contain a valid choice: {response}")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Qwen response did not contain a message: {response}")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Qwen response did not contain text content: {response}")
    return content
