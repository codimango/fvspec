"""Avocado inspect-ai provider adapter (mTLS to Meta AI Gateway)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import subprocess
from typing import Any

import anyio

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.solver._task_state import sample_state
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

# Reuse the text-parse-and-write shim plus workspace lookup from claude_cli.
# Avocado's OpenAI-compat endpoint does not emit `tool_calls` — the model
# returns prose even when a `tools` payload is present and
# `tool_choice="required"` is set (verified by direct probe).
from baselines.providers.claude_cli import _current_workspace, _write_spec_from_output


# Whitelist of stop-reason strings inspect-ai's Literal accepts. Vendor-specific
# values ("error", "eos", "safety", ...) would otherwise crash pydantic validation.
_STOP_REASONS = {"stop", "max_tokens", "model_length", "tool_calls",
                 "content_filter", "unknown"}


def _normalize_stop_reason(raw: str | None) -> str:
    if raw in _STOP_REASONS:
        return raw
    if raw == "length":
        return "max_tokens"
    return "unknown"


_DEFAULT_URL = "https://metacode-modelapi.ai-gateway.fbinfra.net/v1/chat/completions"
_DEFAULT_CERT = "/var/facebook/x509_identities/server.pem"
# The AI Gateway silently defaults max_tokens=128 when the field is omitted,
# which truncates every fvspec response mid-first-theorem. Explicit high default
# lets the model actually finish; if config.max_tokens is set, we honor that.
_DEFAULT_MAX_TOKENS = 32768
_DEFAULT_REFINEMENTS = 2
logger = logging.getLogger(__name__)


class AvocadoAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        client_cert: str = _DEFAULT_CERT,
        reasoning_effort: str = "xhigh",
        timeout: int = 1800,
        refinements: int = _DEFAULT_REFINEMENTS,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url or _DEFAULT_URL,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        self._client_cert = client_cert
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._refinements = refinements

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        try:
            messages = _messages_to_openai(input)
            last_error: str | None = None
            last_output: ModelOutput | None = None
            attempts_used = 0
            self._record_attempt_state(attempts_used=0, last_error=None)

            for attempt in range(self._refinements + 1):
                attempts_used = attempt + 1
                self._record_attempt_state(
                    attempts_used=attempts_used,
                    last_error=last_error,
                )
                payload = self._build_payload(messages, tools, tool_choice, config)
                response, err = await self._call_with_retries(payload)
                if err or response is None:
                    logger.error(
                        "avocado call failed; continuing with empty output: %s",
                        err,
                    )
                    self._record_attempt_state(
                        attempts_used=attempts_used,
                        last_error=err,
                    )
                    return self._error_output(f"avocado call failed: {err}")

                out = self._response_to_output(response)
                last_output = out
                content = out.choices[0].message.text
                # avocado ignores the `tools` payload; write the last ```lean
                # block from the model's text response into Spec.lean so
                # lake_build_scorer sees it.
                _write_spec_from_output(content)
                ok, error = await asyncio.to_thread(self._evaluate_workspace)
                if ok:
                    if attempts_used > 1:
                        logger.warning(
                            "avocado completed after %d model attempts",
                            attempts_used,
                        )
                    self._record_attempt_state(
                        attempts_used=attempts_used,
                        last_error=None,
                    )
                    return out

                last_error = error
                self._record_attempt_state(
                    attempts_used=attempts_used,
                    last_error=last_error,
                )
                if attempt < self._refinements:
                    logger.warning(
                        "avocado attempt %d failed; retrying with compile feedback: %s",
                        attempt + 1,
                        last_error,
                    )
                    messages = self._build_refinement_messages(
                        messages,
                        content,
                        last_error or "Unknown error",
                    )

            assert last_output is not None
            self._record_attempt_state(
                attempts_used=attempts_used,
                last_error=last_error,
            )
            return last_output
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("avocado generate crashed; continuing with empty output")
            return self._error_output("avocado generate crashed")

    def _error_output(self, message: str) -> ModelOutput:
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=f"[avocado error] {message}"),
                    stop_reason="unknown",
                )
            ],
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "reasoning_effort": self._reasoning_effort,
            "max_tokens": config.max_tokens if config.max_tokens is not None else _DEFAULT_MAX_TOKENS,
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if tools:
            payload["tools"] = _tools_to_openai(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        return payload

    def _call_once(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            proc = subprocess.run(
                [
                    "curl", "-sS", "-X", "POST", "--cert", self._client_cert,
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload),
                    self.base_url,
                ],
                capture_output=True, text=True, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        if proc.returncode != 0:
            return None, f"curl_rc={proc.returncode} stderr={proc.stderr[:200]}"
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            return None, f"json_decode: {e} raw={proc.stdout[:200]}"
        if "error" in body:
            return None, f"api_error: {body['error']}"
        return body, None

    async def _call_with_retries(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        # Retry loop stays on the coroutine side so anyio.sleep() propagates
        # inspect-ai's task-group cancellation immediately. Only the blocking
        # curl call goes through asyncio.to_thread.
        last = None
        for attempt in range(1, 9):
            resp, err = await asyncio.to_thread(self._call_once, payload)
            if err is None:
                return resp, None
            last = err
            if "rate" not in err.lower() and "429" not in err and "TIMEOUT" not in err:
                return None, err
            await anyio.sleep(60 + 30 * attempt + random.uniform(0, 5))
        return None, f"retries_exhausted last={last}"

    def _response_to_output(self, response: dict[str, Any]) -> ModelOutput:
        choice = response["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args_parsed = json.loads(args)
                except json.JSONDecodeError:
                    args_parsed = {}
            else:
                args_parsed = args
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    function=tc["function"]["name"],
                    arguments=args_parsed,
                )
            )
        usage = response.get("usage") or {}
        return ModelOutput(
            model=response.get("model", self.model_name),
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=content,
                        tool_calls=tool_calls or None,
                    ),
                    stop_reason=_normalize_stop_reason(choice.get("finish_reason")),
                )
            ],
            usage=ModelUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    def _evaluate_workspace(self) -> tuple[bool, str | None]:
        workspace = _current_workspace()
        if workspace is None:
            return True, None

        spec_file = workspace / "Fvspec" / "Spec.lean"
        if not spec_file.exists():
            return False, "No Spec.lean was written."

        spec_content = spec_file.read_text()
        sorries = len(re.findall(r"\bsorry\b", spec_content))
        if sorries > 0:
            return False, f"The candidate still contains {sorries} `sorry` placeholder(s)."

        try:
            result = subprocess.run(
                ["lake", "build"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "lake build timed out after 120s."
        except Exception as e:
            return False, f"lake build failed to run: {e}"

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode == 0 and "declaration uses 'sorry'" not in combined:
            return True, None

        error_text = (result.stderr or result.stdout or "").strip()
        if not error_text:
            error_text = f"lake build failed with return code {result.returncode}"
        return False, error_text[:2000]

    def _record_attempt_state(
        self,
        *,
        attempts_used: int,
        last_error: str | None,
    ) -> None:
        state = sample_state()
        if state is None:
            return
        state.metadata["avocado_model_attempts"] = attempts_used
        state.metadata["avocado_refinements_used"] = max(0, attempts_used - 1)
        state.metadata["avocado_max_refinements"] = self._refinements
        if last_error:
            state.metadata["avocado_last_checker_error"] = last_error
        else:
            state.metadata.pop("avocado_last_checker_error", None)

    def _build_refinement_messages(
        self,
        messages: list[dict[str, Any]],
        previous_output: str,
        error_message: str,
    ) -> list[dict[str, Any]]:
        return [
            *messages,
            {"role": "assistant", "content": previous_output},
            {
                "role": "user",
                "content": (
                    "Your previous attempt did not satisfy the checker.\n\n"
                    "Checker feedback:\n"
                    f"{error_message}\n\n"
                    "Revise the Lean proof/spec accordingly. Return only a single ```lean fenced block "
                    "containing the full updated Spec.lean contents."
                ),
            },
        ]


def _messages_to_openai(input: list[ChatMessage]) -> list[dict[str, Any]]:
    out = []
    for m in input:
        role = m.role
        d: dict[str, Any] = {"role": role, "content": m.text}
        if role == "assistant" and getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if role == "tool":
            d["tool_call_id"] = getattr(m, "tool_call_id", "")
        out.append(d)
    return out


def _tools_to_openai(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        params = (
            t.parameters.model_dump(exclude_none=True)
            if hasattr(t.parameters, "model_dump")
            else t.parameters
        )
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _strip_nulls(params),
            },
        })
    return out


def _strip_nulls(obj: Any) -> Any:
    """Recursively drop None-valued keys and None list items. Meta AI Gateway's
    schema validator rejects explicit nulls where a typed value is expected."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj if v is not None]
    return obj


@modelapi(name="avocado")
def avocado() -> type[ModelAPI]:
    return AvocadoAPI
