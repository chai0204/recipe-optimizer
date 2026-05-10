"""LLMClient abstraction.

Every LLM-using module delegates to a concrete LLMClient. The client
takes responsibility for:

- Composing the full prompt (system + few-shot examples + user prompt + schema hint)
- Calling the underlying model
- Parsing/validating the structured output against a Pydantic schema
- Retrying once on validation failure with explicit error feedback
- Logging the call for later variance analysis

This indirection is the discipline that controls LLM output variance:
modules describe *what* they want via Pydantic schemas; the client
handles *how* it is generated and validated.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """Raised when LLM output cannot be validated against the requested schema."""


@dataclass
class LLMCallContext:
    """Decoding parameters passed per call. Kept separate so module code
    does not have to know about model-level settings."""

    temperature: float = 0.0
    max_tokens: int = 2048
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallLog:
    """Single LLM call record for later variance analysis.

    Stored in-memory by default. Use ``LLMClient.dump_logs_jsonl()`` to
    persist for offline analysis.
    """

    timestamp: str
    label: str
    prompt: str
    system: str | None
    output_schema_name: str
    raw_output: str | None
    parsed_output: dict | None
    attempts: int
    success: bool
    error: str | None = None


FewShotExamples = list[tuple[str, BaseModel | dict]]


class LLMClient(ABC):
    """Abstract LLM client.

    Concrete implementations:
        - MockLLMClient: returns canned responses (for tests / fast iteration)
        - LocalLLMClient: wraps a HuggingFace transformers model
    """

    def __init__(self) -> None:
        self._call_logs: list[LLMCallLog] = []

    @property
    def call_logs(self) -> list[LLMCallLog]:
        """Read-only view of call history (a copy)."""
        return list(self._call_logs)

    def clear_logs(self) -> None:
        self._call_logs.clear()

    def dump_logs_jsonl(self, path: str | Path) -> None:
        """Append all current call logs to a JSON-Lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            for log in self._call_logs:
                fp.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")

    def _record(self, log: LLMCallLog) -> None:
        self._call_logs.append(log)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @abstractmethod
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
        """Generate output validated against output_schema.

        Args:
            prompt: User-level instruction text.
            output_schema: Pydantic class the output must conform to.
            system: System / role instruction.
            examples: Few-shot pairs ``(input_text, output_dict_or_model)``.
            context: Decoding parameters (temperature, max_tokens).
            max_retries: How many times to retry on schema validation failure.
            label: Module name used for logging (e.g., "parser").
            inject_schema: When True (default), the client embeds the full
                JSON Schema of ``output_schema`` into the system message
                so the model knows the exact target shape. Set False when
                the caller has already described the structure manually
                in ``system`` — useful for large schemas where automatic
                injection inflates the prompt past the model's effective
                attention window.

        Returns:
            Validated instance of ``output_schema``.

        Raises:
            LLMOutputError: validation failed after all retries.
        """
        raise NotImplementedError
