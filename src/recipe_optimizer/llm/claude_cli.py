"""ClaudeCliClient: invokes ``claude -p`` to use Claude Code's auth.

This is a comparison / debugging backend that piggy-backs on the
user's existing Claude Code subscription instead of requiring a
separate ``ANTHROPIC_API_KEY``.

Trade-offs:

- Each call spawns the ``claude`` CLI as a subprocess, which carries
  the full Claude Code system prompt (~130k tokens). The first call
  pays a cache-creation cost; subsequent calls inside the 5-min
  ephemeral window benefit from prompt caching.
- Output is the assistant text only — we extract JSON from prose. No
  tool_use structured output (which would require direct Messages
  API access).
- Best suited for **quality comparison**, not production traffic.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .client import (
    FewShotExamples,
    LLMCallContext,
    LLMCallLog,
    LLMClient,
    LLMOutputError,
)

T = TypeVar("T", bound=BaseModel)

SCHEMA_INSTRUCTION_TEMPLATE = """\
Output a single JSON object that conforms exactly to this JSON Schema:

```
{schema}
```

Rules:
- Output the JSON object and nothing else.
- No markdown fences, no prose, no comments.
"""


class ClaudeCliClient(LLMClient):
    """LLM client backed by the local ``claude -p`` CLI."""

    def __init__(
        self,
        model: str = "haiku",
        timeout_sec: int = 300,
    ) -> None:
        super().__init__()
        self.model = model
        self.timeout_sec = timeout_sec

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------

    def _compose_full_prompt(
        self,
        system: str | None,
        examples: FewShotExamples | None,
        prompt: str,
        output_schema: type[BaseModel],
        inject_schema: bool,
    ) -> str:
        """Assemble a single prompt string for the CLI.

        Claude Code's ``claude -p`` does not accept structured chat
        messages with assistant turns, so we collapse system, examples,
        and user prompt into a single textual block delimited by tags.
        The model still respects this hierarchy in practice.
        """
        parts: list[str] = []

        sys_text = system or "You are a precise assistant."
        if inject_schema:
            schema_json = json.dumps(
                output_schema.model_json_schema(), ensure_ascii=False, indent=2
            )
            sys_text += "\n\n" + SCHEMA_INSTRUCTION_TEMPLATE.format(schema=schema_json)
        parts.append(f"<system>\n{sys_text}\n</system>")

        for ex_in, ex_out in examples or []:
            ex_out_text = (
                json.dumps(ex_out.model_dump(), ensure_ascii=False, indent=2)
                if isinstance(ex_out, BaseModel)
                else json.dumps(ex_out, ensure_ascii=False, indent=2)
            )
            parts.append(
                f"<example>\n"
                f"<input>\n{ex_in}\n</input>\n"
                f"<output>\n{ex_out_text}\n</output>\n"
                f"</example>"
            )

        parts.append(f"<task>\n{prompt}\n</task>")
        parts.append("Respond with the JSON object only.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------

    def _call_cli(self, full_prompt: str) -> str:
        """Run the Claude Code CLI and return the assistant text."""
        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--model",
                    self.model,
                    "--output-format",
                    "json",
                    "--permission-mode",
                    "acceptEdits",
                    "--allowed-tools",
                    "",
                ],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise LLMOutputError(
                f"claude CLI timed out after {self.timeout_sec}s"
            ) from e

        if result.returncode != 0:
            raise LLMOutputError(
                f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise LLMOutputError(
                f"claude CLI returned non-JSON stdout: {result.stdout[:500]}"
            ) from e

        if payload.get("is_error"):
            raise LLMOutputError(
                f"claude CLI reported error: {payload.get('api_error_status')}"
            )

        return payload.get("result", "") or ""

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        start = cleaned.find("{")
        if start < 0:
            return cleaned
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        return cleaned[start:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_structured(
        self,
        *,
        prompt: str,
        output_schema: type[T],
        system: str | None = None,
        examples: FewShotExamples | None = None,
        context: LLMCallContext | None = None,
        max_retries: int = 1,
        label: str = "",
        inject_schema: bool = True,
    ) -> T:
        # ``context`` is accepted for API parity but the CLI does not
        # expose temperature/max_tokens controls directly. Decoding
        # settings are inherited from Claude Code defaults.
        del context

        ts = self._now_iso()
        full_prompt = self._compose_full_prompt(
            system, examples, prompt, output_schema, inject_schema
        )

        last_error: str | None = None
        raw_text = ""

        for attempt in range(max_retries + 1):
            raw_text = self._call_cli(full_prompt)
            json_text = self._extract_json(raw_text)
            try:
                result = output_schema.model_validate_json(json_text)
                self._record(
                    LLMCallLog(
                        timestamp=ts,
                        label=label,
                        prompt=prompt,
                        system=system,
                        output_schema_name=output_schema.__name__,
                        raw_output=raw_text,
                        parsed_output=result.model_dump(),
                        attempts=attempt + 1,
                        success=True,
                    )
                )
                return result
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = str(e)
                if attempt < max_retries:
                    full_prompt = (
                        f"{full_prompt}\n\n"
                        f"<retry_feedback>\n"
                        f"Previous output failed validation: {e}\n"
                        f"Correct it. JSON only, no fences, no prose.\n"
                        f"</retry_feedback>"
                    )

        self._record(
            LLMCallLog(
                timestamp=ts,
                label=label,
                prompt=prompt,
                system=system,
                output_schema_name=output_schema.__name__,
                raw_output=raw_text,
                parsed_output=None,
                attempts=max_retries + 1,
                success=False,
                error=last_error,
            )
        )
        raise LLMOutputError(
            f"Claude CLI output for label='{label}' failed validation: {last_error}"
        )
