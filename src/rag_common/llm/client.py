"""Thin adapters for invoking OpenAI-compatible clients."""

from __future__ import annotations

import base64
from typing import Any

from rag_common.exceptions import ResponseFormatError
from rag_common.llm.json_utils import parse_json_dict
from rag_common.types import JSONDict


class OpenAIClientAdapter:
    """
    Adapts either OpenAI Responses API or Chat Completions API.

    The passed client can be an official OpenAI client or a compatible wrapper.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> JSONDict:
        text = self.complete_text(system_prompt=system_prompt, user_prompt=user_prompt)
        return parse_json_dict(text)

    def complete_json_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_mime_type: str = "image/png",
    ) -> JSONDict:
        text = self.complete_text_with_image(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
        return parse_json_dict(text)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None

        for method in (self._complete_via_responses, self._complete_via_chat):
            try:
                text = method(system_prompt=system_prompt, user_prompt=user_prompt)
                if text.strip():
                    return text
            except Exception as exc:  # pragma: no cover - runtime client differences
                last_error = exc

        raise ResponseFormatError(
            "Unable to obtain textual output from the provided OpenAI client."
        ) from last_error

    def complete_text_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_mime_type: str = "image/png",
    ) -> str:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty.")

        image_data_url = _to_data_url(image_bytes=image_bytes, mime_type=image_mime_type)
        last_error: Exception | None = None

        for method in (self._complete_via_responses_with_image, self._complete_via_chat_with_image):
            try:
                text = method(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_data_url=image_data_url,
                )
                if text.strip():
                    return text
            except Exception as exc:  # pragma: no cover - runtime client differences
                last_error = exc

        raise ResponseFormatError(
            "Unable to obtain textual output from the provided OpenAI client with image input."
        ) from last_error

    def _complete_via_responses(self, system_prompt: str, user_prompt: str) -> str:
        responses_api = getattr(self.client, "responses")
        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        response = _create_with_optional_parameters(
            create_callable=responses_api.create,
            kwargs=kwargs,
            optional_keys=("temperature", "max_output_tokens"),
        )
        return _extract_text(response)

    def _complete_via_chat(self, system_prompt: str, user_prompt: str) -> str:
        chat_api = getattr(getattr(self.client, "chat"), "completions")
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        response = _create_with_optional_parameters(
            create_callable=chat_api.create,
            kwargs=kwargs,
            optional_keys=("response_format",),
        )
        return _extract_text(response)

    def _complete_via_responses_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
    ) -> str:
        responses_api = getattr(self.client, "responses")
        input_variants: list[list[dict[str, Any]]] = [
            [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
        ]

        last_error: Exception | None = None
        for payload in input_variants:
            kwargs = {
                "model": self.model,
                "input": payload,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
            try:
                response = _create_with_optional_parameters(
                    create_callable=responses_api.create,
                    kwargs=kwargs,
                    optional_keys=("temperature", "max_output_tokens"),
                )
                text = _extract_text(response)
                if text.strip():
                    return text
            except Exception as exc:  # pragma: no cover - runtime client differences
                last_error = exc

        raise ResponseFormatError(
            "Responses API image extraction failed for the provided client."
        ) from last_error

    def _complete_via_chat_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
    ) -> str:
        chat_api = getattr(getattr(self.client, "chat"), "completions")
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        response = _create_with_optional_parameters(
            create_callable=chat_api.create,
            kwargs=kwargs,
            optional_keys=("response_format",),
        )
        return _extract_text(response)


def _extract_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, dict):
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        if isinstance(response.get("choices"), list):
            return _extract_from_choices(response["choices"])
        if isinstance(response.get("output"), list):
            return _extract_from_output_items(response["output"])

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        text = _extract_from_output_items(output)
        if text:
            return text

    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        text = _extract_from_choices(choices)
        if text:
            return text

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):  # pragma: no cover - runtime client differences
        dumped = model_dump()
        if isinstance(dumped, dict):
            return _extract_text(dumped)

    return ""


def _extract_from_output_items(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for piece in content:
            if isinstance(piece, dict):
                text = piece.get("text")
            else:
                text = getattr(piece, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_from_choices(choices: list[Any]) -> str:
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    if message is None:
        return ""

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _create_with_optional_parameters(
    *,
    create_callable: Any,
    kwargs: dict[str, Any],
    optional_keys: tuple[str, ...],
) -> Any:
    try:
        return create_callable(**kwargs)
    except Exception as first_error:  # pragma: no cover - runtime client differences
        reduced = dict(kwargs)
        for key in optional_keys:
            reduced.pop(key, None)
        if reduced == kwargs:
            raise
        try:
            return create_callable(**reduced)
        except Exception:
            raise first_error


def _to_data_url(*, image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
