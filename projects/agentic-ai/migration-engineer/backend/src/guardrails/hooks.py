"""The guardrail / hook plane - policy that runs BEFORE every tool call.

In `live-sdk` mode these are wired as a Claude Agent SDK `PreToolUse` hook; in `stub`
mode the `ToolInvoker` calls the exact same `HookManager.pre_tool_use(...)`. Same policy,
both modes - that is the whole point of putting policy in the harness contract.

What it enforces (defense in depth, mapping to the OWASP Top 10 for Agentic Apps):

  * Path containment  - a tool may only write INSIDE the repo's worktree. Blocks path
                        traversal / "helpfully" editing something outside the sandbox.
  * Test protection   - the worker may not edit the repository's tests. This is the
                        first line of defense against reward hacking (make the tests
                        pass by deleting them). The reviewer agent and an eval scorer
                        are the second and third lines.
  * Secret redaction  - any secret-looking string flowing through a tool's input is
                        redacted before it is executed or logged, so credentials never
                        leak into traces or model context.

A blocked call does not crash the run: the worker receives the denial as an
observation and must find another way - which is exactly how a safe harness behaves.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

# Files the agent is never allowed to write. Protecting the tests is what makes
# "just delete the failing test" impossible at the guardrail layer.
PROTECTED_GLOBS = ("test_*.py", "*_test.py", "conftest.py")

# Only these extensions may be written at all (least privilege for a code migrator).
WRITABLE_SUFFIXES = (".py", ".cfg", ".ini", ".toml", ".txt", ".md")

# Secret-looking patterns, redacted before any tool executes or logs them.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google / Gemini API keys
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{12,}"),
]
_REDACTED = "***REDACTED***"


@dataclass
class HookDecision:
    allow: bool
    reason: str = ""
    redactions: int = 0
    tool_input: dict | None = None  # possibly-redacted copy to actually run


@dataclass
class HookManager:
    worktree: Path
    blocks: list[dict] = field(default_factory=list)
    # TEACHING KNOB: when False, write-policy checks are skipped (secret redaction
    # still runs). Exists only so learners can observe the reviewer's tamper gate
    # catch what this hook normally blocks. Production keeps this True.
    enforce: bool = True

    def pre_tool_use(self, tool_name: str, mutating: bool, tool_input: dict) -> HookDecision:
        """Vet one tool call. Returns an allow/deny decision + a redacted input."""
        redacted, n = self._redact_mapping(tool_input)

        if mutating and self.enforce:
            path = redacted.get("path") or redacted.get("file") or ""
            verdict = self._check_write_path(str(path))
            if verdict is not None:
                self._record(tool_name, str(path), verdict)
                return HookDecision(allow=False, reason=verdict, redactions=n, tool_input=redacted)

        return HookDecision(allow=True, redactions=n, tool_input=redacted)

    # --- path policy --------------------------------------------------------

    def _check_write_path(self, rel_or_abs: str) -> str | None:
        """Return a denial reason, or None if the write is permitted."""
        if not rel_or_abs:
            return "write blocked: no target path given"
        name = Path(rel_or_abs).name
        for glob in PROTECTED_GLOBS:
            if fnmatch.fnmatch(name, glob):
                return f"write blocked: '{name}' is a protected test file (tests may not be modified)"
        if Path(rel_or_abs).suffix not in WRITABLE_SUFFIXES:
            return f"write blocked: '{Path(rel_or_abs).suffix or name}' is not a writable file type"
        # Containment: resolve against the worktree and ensure it stays inside.
        target = (self.worktree / rel_or_abs).resolve()
        try:
            target.relative_to(self.worktree.resolve())
        except ValueError:
            return "write blocked: path escapes the repository worktree (traversal)"
        return None

    # --- secret redaction ---------------------------------------------------

    def _redact_mapping(self, data: dict) -> tuple[dict, int]:
        count = 0
        out: dict = {}
        for k, v in data.items():
            if isinstance(v, str):
                red, c = self.redact(v)
                out[k], count = red, count + c
            else:
                out[k] = v
        return out, count

    @staticmethod
    def redact(text: str) -> tuple[str, int]:
        count = 0
        for pat in _SECRET_PATTERNS:
            text, n = pat.subn(_REDACTED, text)
            count += n
        return text, count

    def _record(self, tool: str, target: str, reason: str) -> None:
        self.blocks.append({"tool": tool, "target": target, "reason": reason})
