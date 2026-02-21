"""Thin adapters for invoking OpenAI-compatible clients."""

from __future__ import annotations

import base64
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from rag_common.exceptions import ResponseFormatError
from rag_common.llm.json_utils import parse_json_dict
from rag_common.types import JSONDict

LOGGER = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_CLASS_MARKERS = (
    "RateLimit",
    "Timeout",
    "Connection",
    "ServiceUnavailable",
    "InternalServer",
    "APIError",
)
_RETRYABLE_MESSAGE_MARKERS = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "temporar",
    "connection",
    "service unavailable",
    "try again",
    "overloaded",
    "502",
    "503",
    "504",
)


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
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.8,
        retry_max_delay_seconds: float = 8.0,
        request_timeout_seconds: float | None = 60.0,
        min_request_interval_seconds: float = 0.0,
        event_handler: Callable[[JSONDict], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max(max_retries, 0)
        self.retry_base_delay_seconds = max(retry_base_delay_seconds, 0.0)
        self.retry_max_delay_seconds = max(retry_max_delay_seconds, self.retry_base_delay_seconds)
        self.request_timeout_seconds = request_timeout_seconds
        self.min_request_interval_seconds = max(min_request_interval_seconds, 0.0)
        self._event_handler = event_handler
        self._request_counter = 0
        self._last_dispatch_timestamp = 0.0

    def complete_json(self, system_prompt: str, user_prompt: str) -> JSONDict:
        text = self.complete_text(system_prompt=system_prompt, user_prompt=user_prompt)
        return parse_json_dict(text)

    def runtime_config(self) -> JSONDict:
        """Return operational LLM adapter settings."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
            "retry_base_delay_seconds": self.retry_base_delay_seconds,
            "retry_max_delay_seconds": self.retry_max_delay_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "event_handler_enabled": self._event_handler is not None,
        }

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

        for method_name, method in (
            ("responses", self._complete_via_responses),
            ("chat_completions", self._complete_via_chat),
        ):
            try:
                return self._invoke_with_resilience(
                    method=method,
                    method_name=method_name,
                    call_kwargs={
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    },
                )
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

        for method_name, method in (
            ("responses_image", self._complete_via_responses_with_image),
            ("chat_completions_image", self._complete_via_chat_with_image),
        ):
            try:
                return self._invoke_with_resilience(
                    method=method,
                    method_name=method_name,
                    call_kwargs={
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "image_data_url": image_data_url,
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime client differences
                last_error = exc

        raise ResponseFormatError(
            "Unable to obtain textual output from the provided OpenAI client with image input."
        ) from last_error

    def _invoke_with_resilience(
        self,
        *,
        method: Callable[..., str],
        method_name: str,
        call_kwargs: JSONDict,
    ) -> str:
        request_id = self._next_request_id()
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            self._respect_min_request_interval()
            started = time.monotonic()
            self._emit_event(
                event_name="llm_request_start",
                request_id=request_id,
                method=method_name,
                attempt=attempt,
                max_attempts=attempts,
            )
            try:
                text = method(**call_kwargs)
                duration_ms = int((time.monotonic() - started) * 1000)
                if not text.strip():
                    raise ResponseFormatError("LLM returned empty text output.")
                self._emit_event(
                    event_name="llm_request_success",
                    request_id=request_id,
                    method=method_name,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    output_chars=len(text),
                )
                return text
            except Exception as exc:  # pragma: no cover - runtime client differences
                duration_ms = int((time.monotonic() - started) * 1000)
                retryable = attempt < attempts and _is_retryable_error(exc)
                self._emit_event(
                    event_name="llm_request_error",
                    request_id=request_id,
                    method=method_name,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    retryable=retryable,
                    error_class=exc.__class__.__name__,
                    error_message=str(exc),
                )
                if not retryable:
                    raise

                sleep_seconds = self._compute_backoff(attempt)
                LOGGER.warning(
                    "Retrying LLM request id=%s method=%s attempt=%s/%s sleep=%.2fs error=%s",
                    request_id,
                    method_name,
                    attempt,
                    attempts,
                    sleep_seconds,
                    exc,
                )
                self._emit_event(
                    event_name="llm_request_retry_scheduled",
                    request_id=request_id,
                    method=method_name,
                    attempt=attempt,
                    sleep_seconds=sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise ResponseFormatError("LLM request failed after retry attempts.")

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
        if self.request_timeout_seconds is not None:
            kwargs["timeout"] = self.request_timeout_seconds
        response = _create_with_optional_parameters(
            create_callable=responses_api.create,
            kwargs=kwargs,
            optional_keys=("temperature", "max_output_tokens", "timeout"),
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
        if self.request_timeout_seconds is not None:
            kwargs["timeout"] = self.request_timeout_seconds
        response = _create_with_optional_parameters(
            create_callable=chat_api.create,
            kwargs=kwargs,
            optional_keys=("response_format", "timeout"),
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
            if self.request_timeout_seconds is not None:
                kwargs["timeout"] = self.request_timeout_seconds
            try:
                response = _create_with_optional_parameters(
                    create_callable=responses_api.create,
                    kwargs=kwargs,
                    optional_keys=("temperature", "max_output_tokens", "timeout"),
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
        if self.request_timeout_seconds is not None:
            kwargs["timeout"] = self.request_timeout_seconds
        response = _create_with_optional_parameters(
            create_callable=chat_api.create,
            kwargs=kwargs,
            optional_keys=("response_format", "timeout"),
        )
        return _extract_text(response)

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"llm-{self._request_counter}"

    def _respect_min_request_interval(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_dispatch_timestamp
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_dispatch_timestamp = time.monotonic()

    def _compute_backoff(self, attempt: int) -> float:
        exponential = self.retry_base_delay_seconds * (2 ** max(attempt - 1, 0))
        capped = min(exponential, self.retry_max_delay_seconds)
        jitter = random.uniform(0.0, max(self.retry_base_delay_seconds * 0.2, 0.01))
        return capped + jitter

    def _emit_event(self, *, event_name: str, **payload: Any) -> None:
        if self._event_handler is None:
            return
        event: JSONDict = {
            "event": event_name,
            "timestamp_unix": int(time.time()),
            "model": self.model,
        }
        event.update(payload)
        try:
            self._event_handler(event)
        except Exception:  # pragma: no cover - event consumers are external
            LOGGER.exception("LLM event handler raised unexpectedly.")


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


def _is_retryable_error(error: Exception, *, _depth: int = 0) -> bool:
    if _depth > 3:
        return False
    if isinstance(error, ResponseFormatError):
        return False

    status_code = _extract_status_code(error)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True

    class_name = error.__class__.__name__.lower()
    if any(marker.lower() in class_name for marker in _RETRYABLE_CLASS_MARKERS):
        return True

    message = str(error).lower()
    if any(marker in message for marker in _RETRYABLE_MESSAGE_MARKERS):
        return True

    cause = getattr(error, "__cause__", None)
    if isinstance(cause, Exception) and cause is not error:
        if _is_retryable_error(cause, _depth=_depth + 1):
            return True

    context = getattr(error, "__context__", None)
    if isinstance(context, Exception) and context is not error:
        if _is_retryable_error(context, _depth=_depth + 1):
            return True

    return False


def _extract_status_code(error: Exception) -> int | None:
    for key in ("status_code", "http_status", "status"):
        value = getattr(error, key, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    if response is not None:
        candidate = getattr(response, "status_code", None)
        if isinstance(candidate, int):
            return candidate
    return None
