"""Avocado inspect-ai provider adapter (mTLS to Meta AI Gateway)."""

from __future__ import annotations

import asyncio
import json
import random
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
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo


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

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        payload = self._build_payload(input, tools, tool_choice, config)
        response, err = await self._call_with_retries(payload)
        if err or response is None:
            raise RuntimeError(f"avocado call failed: {err}")
        return self._response_to_output(response)

    def _build_payload(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _messages_to_openai(input),
            "reasoning_effort": self._reasoning_effort,
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
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
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.model_dump() if hasattr(t.parameters, "model_dump") else t.parameters,
            },
        }
        for t in tools
    ]


@modelapi(name="avocado")
def avocado() -> type[ModelAPI]:
    return AvocadoAPI
