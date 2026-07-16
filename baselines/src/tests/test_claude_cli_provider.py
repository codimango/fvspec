"""Unit tests for the claude_cli inspect-ai provider adapter."""

import asyncio
from unittest.mock import patch

from inspect_ai.model import ChatMessageUser, GenerateConfig

from baselines.providers.claude_cli import ClaudeCLIAPI


def test_claude_cli_returns_completion_text():
    api = ClaudeCLIAPI(model_name="claude-opus-4-8[1m]")
    with patch.object(api, "_run_cli", return_value=("The answer is 42.", False)):
        out = asyncio.run(
            api.generate(
                input=[ChatMessageUser(content="what is 6*7?")],
                tools=[],
                tool_choice=None,
                config=GenerateConfig(max_tokens=32),
            )
        )
    assert out.choices[0].message.text == "The answer is 42."


def test_claude_cli_write_shim_extracts_and_writes(tmp_path, monkeypatch):
    from baselines.providers import claude_cli as mod

    workspace = tmp_path
    (workspace / "Fvspec").mkdir()

    class _State:
        metadata = {"workspace": str(workspace)}

    monkeypatch.setattr(mod, "sample_state", lambda: _State())

    text = (
        "Sure — here is the completed spec:\n"
        "```lean\n"
        "theorem foo : True := trivial\n"
        "```\n"
        "Done."
    )
    code = mod._write_spec_from_output(text)
    assert code == "theorem foo : True := trivial"
    written = (workspace / "Fvspec" / "Spec.lean").read_text()
    assert written == "theorem foo : True := trivial\n"


def test_claude_cli_write_shim_noop_without_block(tmp_path, monkeypatch):
    from baselines.providers import claude_cli as mod

    workspace = tmp_path
    (workspace / "Fvspec").mkdir()

    class _State:
        metadata = {"workspace": str(workspace)}

    monkeypatch.setattr(mod, "sample_state", lambda: _State())
    assert mod._write_spec_from_output("no fenced block here") is None
    assert not (workspace / "Fvspec" / "Spec.lean").exists()
