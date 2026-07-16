"""claude_cli inspect-ai provider adapter (shell-out to Claude Code + write shim).

Runs `claude --print` one-shot, extracts the last ```lean fenced block from
stdout, writes it to <workspace>/Fvspec/Spec.lean via
inspect_ai.solver._task_state.sample_state() (same hook tools.py::_get_workspace
uses), and returns the raw text as a plain assistant message with no tool_calls
so the tool_calls="loop" in solver.proof_solver exits after one turn.

Streaming-reader thread pattern ported from
verina_upstream/src/verina/utils/claude_cli_lm.py::_run_claude_cli — preserves
partial stdout on timeout.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import threading
from pathlib import Path

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
from inspect_ai.tool import ToolChoice, ToolInfo


_CLI_PREAMBLE_MARKERS = (
    "Claude Code at Meta",
    "Using AI Gateway",
    "Warning: no stdin",
)

_LEAN_BLOCK_RE = re.compile(
    r"```(?:lean4?|Lean4?|LEAN4?)[^\n]*\n(.*?)```",
    re.DOTALL,
)


def _strip_cli_preamble(text: str) -> str:
    lines = text.splitlines()
    while lines and any(m in lines[0] for m in _CLI_PREAMBLE_MARKERS):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _last_lean_block(text: str) -> str | None:
    blocks = _LEAN_BLOCK_RE.findall(text or "")
    return blocks[-1].strip() if blocks else None


def _run_cli_blocking(
    prompt: str, model: str, cli: str, timeout: int
) -> tuple[str, bool]:
    """Return (content, timed_out). Reader-thread pattern from
    verina_upstream/src/verina/utils/claude_cli_lm.py::_run_claude_cli."""
    proc = subprocess.Popen(
        [cli, "--print", "--model", model, prompt],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    chunks: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), ""):
            chunks.append(chunk)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    reader.join(timeout=timeout)

    timed_out = False
    if reader.is_alive():
        timed_out = True
        proc.kill()
        reader.join(timeout=5)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    return _strip_cli_preamble("".join(chunks)).strip(), timed_out


def _messages_to_prompt(input: list[ChatMessage]) -> str:
    """Flatten inspect-ai messages into a single positional prompt for the CLI."""
    if len(input) == 1 and input[0].role == "user":
        return input[0].text
    parts = []
    for m in input:
        parts.append(f"[{m.role}]\n{m.text}")
    return "\n\n".join(parts)


def _write_spec_from_output(text: str) -> str | None:
    """Extract the last ```lean block from text and write it to
    <workspace>/Fvspec/Spec.lean. Returns the extracted code, or None if no
    block was found or no workspace is available. Silently no-op if
    sample_state() returns None (e.g. running outside inspect-ai's task loop).
    """
    code = _last_lean_block(text)
    if not code:
        return None
    state = sample_state()
    if state is None:
        return code
    workspace_path = state.metadata.get("workspace")
    if not workspace_path:
        return code
    spec_file = Path(workspace_path) / "Fvspec" / "Spec.lean"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    # rstrip + explicit trailing newline avoids Lean's no-final-newline warning.
    spec_file.write_text(code.rstrip() + "\n")
    return code


class ClaudeCLIAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        cli: str = "claude",
        timeout: int = 3600,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        self._cli = cli
        self._timeout = timeout

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # tools are ignored — the CLI runs its own internal tool loop opaquely.
        prompt = _messages_to_prompt(input)
        content, timed_out = await asyncio.to_thread(self._run_cli, prompt)
        if timed_out and not content:
            raise RuntimeError(f"claude CLI timed out after {self._timeout}s")
        # write shim: put the extracted lean block into Spec.lean so
        # lake_build_scorer sees it without any inspect-ai tool_call round-trip.
        _write_spec_from_output(content)
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=content),
                    stop_reason="length" if timed_out else "stop",
                )
            ],
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    def _run_cli(self, prompt: str) -> tuple[str, bool]:
        """Instance wrapper so tests can patch `api._run_cli`."""
        return _run_cli_blocking(prompt, self.model_name, self._cli, self._timeout)


@modelapi(name="claude_cli")
def claude_cli() -> type[ModelAPI]:
    return ClaudeCLIAPI
