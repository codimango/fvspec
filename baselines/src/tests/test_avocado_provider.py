"""Unit tests for the avocado inspect-ai provider adapter."""

import asyncio
from unittest.mock import patch

from inspect_ai.model import ChatMessageUser, GenerateConfig

from baselines.providers.avocado import AvocadoAPI


def _mock_ok_response():
    return (
        {
            "id": "chatcmpl-x",
            "model": "avocado-5.14-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        None,
    )


def test_avocado_provider_returns_model_output():
    api = AvocadoAPI(model_name="avocado-5.14-agent")
    with patch.object(api, "_call_once", return_value=_mock_ok_response()):
        out = asyncio.run(
            api.generate(
                input=[ChatMessageUser(content="hi")],
                tools=[],
                tool_choice=None,
                config=GenerateConfig(max_tokens=32),
            )
        )
    assert out.choices[0].message.text == "hello world"
    assert out.usage.total_tokens == 7


def test_avocado_tools_payload_strips_nulls():
    from baselines.providers.avocado import _tools_to_openai

    class _ToolInfo:
        name = "write_lean_spec"
        description = "write to Spec.lean"

        class _Params:
            @staticmethod
            def model_dump(**_):
                # Simulates a pydantic dump with unset Optional fields = None.
                return {
                    "type": "object",
                    "properties": {"content": {"type": "string", "default": None}},
                    "required": None,
                    "additionalProperties": None,
                }

        parameters = _Params()

    out = _tools_to_openai([_ToolInfo()])
    assert out[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"content": {"type": "string"}},
    }


def test_avocado_generate_writes_spec_via_shim(tmp_path, monkeypatch):
    """Avocado.generate must call _write_spec_from_output on the model's text
    response so lake_build_scorer sees the extracted lean block."""
    import asyncio
    from unittest.mock import patch
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from baselines.providers import claude_cli as cc_mod
    from baselines.providers.avocado import AvocadoAPI

    workspace = tmp_path
    (workspace / "Fvspec").mkdir()

    class _State:
        metadata = {"workspace": str(workspace)}

    monkeypatch.setattr(cc_mod, "sample_state", lambda: _State())

    api = AvocadoAPI(model_name="avocado-5.14-agent")
    fake = (
        {
            "id": "x", "model": "avocado-5.14-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here's the spec:\n```lean\ntheorem t : 1 = 1 := rfl\n```",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        None,
    )
    with patch.object(api, "_call_once", return_value=fake):
        asyncio.run(api.generate(
            input=[ChatMessageUser(content="proofs please")],
            tools=[], tool_choice="auto",
            config=GenerateConfig(max_tokens=32),
        ))
    assert (workspace / "Fvspec" / "Spec.lean").read_text() == "theorem t : 1 = 1 := rfl\n"


def test_avocado_generate_logs_error_and_continues(caplog):
    api = AvocadoAPI(model_name="avocado-5.14-agent")
    with patch.object(
        api,
        "_call_with_retries",
        return_value=(None, "retries_exhausted last=TIMEOUT"),
    ):
        out = asyncio.run(
            api.generate(
                input=[ChatMessageUser(content="hi")],
                tools=[],
                tool_choice=None,
                config=GenerateConfig(max_tokens=32),
            )
        )

    assert out.choices[0].message.text == "[avocado error] avocado call failed: retries_exhausted last=TIMEOUT"
    assert "avocado call failed; continuing with empty output" in caplog.text


def test_avocado_refines_after_compile_feedback(tmp_path, monkeypatch):
    from baselines.providers import claude_cli as cc_mod

    workspace = tmp_path
    (workspace / "Fvspec").mkdir()

    class _State:
        metadata = {"workspace": str(workspace)}

    monkeypatch.setattr(cc_mod, "sample_state", lambda: _State())

    api = AvocadoAPI(model_name="avocado-5.14-agent")
    first = (
        {
            "id": "x1",
            "model": "avocado-5.14-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```lean\ntheorem foo : True := by\n  sorry\n```",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        None,
    )
    second = (
        {
            "id": "x2",
            "model": "avocado-5.14-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```lean\ntheorem foo : True := by\n  trivial\n```",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        None,
    )

    with (
        patch.object(api, "_call_with_retries", side_effect=[first, second]) as run_api,
        patch.object(
            api,
            "_evaluate_workspace",
            side_effect=[
                (False, "The candidate still contains 1 `sorry` placeholder(s)."),
                (True, None),
            ],
        ),
    ):
        out = asyncio.run(
            api.generate(
                input=[ChatMessageUser(content="prove this")],
                tools=[],
                tool_choice=None,
                config=GenerateConfig(max_tokens=32),
            )
        )

    assert out.choices[0].message.text == "```lean\ntheorem foo : True := by\n  trivial\n```"
    assert run_api.call_count == 2
    assert (workspace / "Fvspec" / "Spec.lean").read_text() == "theorem foo : True := by\n  trivial\n"
