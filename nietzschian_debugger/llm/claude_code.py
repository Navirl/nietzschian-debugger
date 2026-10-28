"""Claude Code CLI backend: run every LLM call through `claude -p`.

Lets a Claude Pro/Max subscriber use the debugger without an API key — the
official `claude` CLI handles auth (your logged-in subscription), and usage
counts against your plan's limits instead of API billing.

Select it with `--model claude-code` (Haiku) or `--model claude-code:sonnet`
/ `claude-code:opus` / any model name `claude --model` accepts.

Environment knobs:
  NIETZSCHIAN_CLAUDE_BIN    path to the claude executable (default: found on PATH)
  NIETZSCHIAN_CLAUDE_ARGS   extra CLI args, shell-split (e.g. "--setting-sources project")
  NIETZSCHIAN_CLAUDE_TIMEOUT seconds per call (default 180)
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Optional

DEFAULT_MODEL = "haiku"

# The debugger's own Anthropic model IDs -> CLI aliases.
_ALIASES = ("haiku", "sonnet", "opus")


class ClaudeCodeError(Exception):
    """Raised when the claude CLI is missing or a call fails."""


def _resolve_bin() -> str:
    explicit = os.environ.get("NIETZSCHIAN_CLAUDE_BIN")
    if explicit:
        return explicit
    # shutil.which resolves claude.cmd / claude.exe on Windows too.
    found = shutil.which("claude")
    if not found:
        raise ClaudeCodeError(
            "The 'claude' CLI was not found on PATH. Install Claude Code and run "
            "`claude` once to log in with your Pro/Max account."
        )
    return found


def _model_arg(model: str) -> str:
    m = (model or DEFAULT_MODEL).lower()
    for alias in _ALIASES:
        if alias in m:
            return alias
    return model or DEFAULT_MODEL


def _render_prompt(messages: list[dict[str, str]]) -> str:
    """Flatten a chat history into one prompt (claude -p is single-shot)."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")
    parts = [
        "Below is the conversation so far. Write ONLY your next reply as the "
        "assistant — no role labels, no tags, no commentary about the transcript.",
        "",
    ]
    for m in messages:
        role = "user" if m.get("role") == "user" else "assistant"
        parts.append(f"<{role}>\n{m.get('content', '')}\n</{role}>")
    return "\n".join(parts)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # If an API key is set, `claude -p` would bill the API instead of the
    # subscription. Drop it so the logged-in plan is used.
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _build_cmd(system_file: str, model: str) -> list[str]:
    cmd = [
        _resolve_bin(),
        "-p",
        "--system-prompt-file", system_file,   # replace Claude Code's coding-agent prompt
        "--model", _model_arg(model),
        "--tools", "",                         # pure text: no file/shell tools
        "--strict-mcp-config",                 # don't spin up your MCP servers
        "--no-session-persistence",            # don't litter `claude --resume`
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",                           # required for stream-json with -p
    ]
    extra = os.environ.get("NIETZSCHIAN_CLAUDE_ARGS")
    if extra:
        cmd += shlex.split(extra)
    return cmd


def _run(system: str, messages: list[dict[str, str]], model: str, on_text=None) -> str:
    timeout = float(os.environ.get("NIETZSCHIAN_CLAUDE_TIMEOUT", "180"))
    prompt = _render_prompt(messages)

    with tempfile.TemporaryDirectory(prefix="nietzschian-cc-") as tmp:
        system_file = os.path.join(tmp, "system.md")
        with open(system_file, "w", encoding="utf-8") as f:
            f.write(system)
        cmd = _build_cmd(system_file, model)

        # cwd = empty temp dir so no project CLAUDE.md gets pulled in.
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            env=_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(prompt)
        proc.stdin.close()

        streamed = ""
        final: Optional[str] = None
        error: Optional[str] = None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type")
                if etype == "stream_event":
                    inner = evt.get("event", {})
                    delta = inner.get("delta", {}) if inner.get("type") == "content_block_delta" else {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            streamed += text
                            if on_text:
                                on_text(text)
                elif etype == "result":
                    if evt.get("is_error"):
                        error = str(evt.get("result") or evt.get("subtype") or "unknown error")
                    else:
                        final = evt.get("result")
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise ClaudeCodeError(f"claude -p timed out after {timeout:.0f}s")

        stderr = proc.stderr.read() if proc.stderr else ""
        if error:
            raise ClaudeCodeError(f"claude -p failed: {error}")
        if proc.returncode != 0 and not (streamed or final):
            raise ClaudeCodeError(
                f"claude -p exited with code {proc.returncode}: {stderr.strip()[:500]}"
            )

    text = streamed or (final or "")
    # Nothing was streamed (e.g. partial messages unsupported): emit it whole.
    if on_text and not streamed and text:
        on_text(text)
    return text.strip()


def complete(system: str, messages: list[dict[str, str]], model: str) -> str:
    return _run(system, messages, model)


# The validator requires a literal '?'. Newer models reached through the CLI
# alias often phrase the challenge as a command ("Name the field ..."), which
# fails validation and drops the turn to a canned fallback question. stream() is
# only used for question generation, so the hint never reaches the judge/summary.
QUESTION_FORMAT_HINT = (
    "\n\nFORMAT: your reply must end with exactly one sentence that is a question "
    "ending in '?'. Never phrase it as a command."
)


def stream(system: str, messages: list[dict[str, str]], model: str, callbacks) -> str:
    try:
        text = _run(system + QUESTION_FORMAT_HINT, messages, model, on_text=callbacks.on_text)
        callbacks.on_complete(text)
        return text
    except Exception as error:
        callbacks.on_error(error)
        raise
