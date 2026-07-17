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
import logging
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

logger = logging.getLogger(__name__)

_CLI_PREAMBLE_MARKERS = (
    "Claude Code at Meta",
    "Using AI Gateway",
    "Warning: no stdin",
)
_DEFAULT_REFINEMENTS = 2

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


def _current_workspace() -> Path | None:
    state = sample_state()
    if state is None:
        return None
    workspace_path = state.metadata.get("workspace")
    if not workspace_path:
        return None
    return Path(workspace_path)


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
        refinements: int = _DEFAULT_REFINEMENTS,
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
        self._refinements = refinements

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # tools are ignored — the CLI runs its own internal tool loop opaquely.
        prompt = _messages_to_prompt(input)
        content = ""
        timed_out = False
        last_error: str | None = None

        for attempt in range(self._refinements + 1):
            if attempt > 0:
                prompt = self._build_refinement_prompt(prompt, content, last_error or "Unknown error")
                logger.info(
                    "claude_cli refinement attempt %d/%d",
                    attempt,
                    self._refinements,
                )

            content, timed_out = await asyncio.to_thread(self._run_cli, prompt)
            if timed_out and not content:
                last_error = f"claude CLI timed out after {self._timeout}s"
            else:
                _write_spec_from_output(content)
                ok, error = await asyncio.to_thread(self._evaluate_workspace)
                if ok:
                    return self._model_output(content, timed_out)
                last_error = error

            if attempt < self._refinements:
                logger.info(
                    "claude_cli attempt %d failed; retrying with compile feedback: %s",
                    attempt + 1,
                    last_error,
                )

        if timed_out and not content:
            raise RuntimeError(last_error or f"claude CLI timed out after {self._timeout}s")
        return self._model_output(content, timed_out)

    def _run_cli(self, prompt: str) -> tuple[str, bool]:
        """Instance wrapper so tests can patch `api._run_cli`."""
        return _run_cli_blocking(prompt, self.model_name, self._cli, self._timeout)

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

    def _build_refinement_prompt(
        self,
        base_prompt: str,
        previous_output: str,
        error_message: str,
    ) -> str:
        return (
            f"{base_prompt}\n\n"
            "Your previous attempt did not satisfy the checker.\n\n"
            "Previous answer:\n"
            f"{previous_output}\n\n"
            "Checker feedback:\n"
            f"{error_message}\n\n"
            "Revise the Lean proof/spec accordingly. Return only a single ```lean fenced block containing the full updated Spec.lean contents."
        )

    def _model_output(self, content: str, timed_out: bool) -> ModelOutput:
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


@modelapi(name="claude_cli")
def claude_cli() -> type[ModelAPI]:
    return ClaudeCLIAPI
