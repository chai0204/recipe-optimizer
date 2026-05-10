"""MockLLMClient for fast development and unit tests.

Register a response per label; the mock returns it (after schema validation)
without invoking any actual model. This is the workhorse for iterating on
algorithm code without paying CPU inference latency.

Typical usage in a test:

    >>> client = MockLLMClient()
    >>> client.register("parser", canned_dag_dict)
    >>> dag = parser.parse(recipe_text)  # internally calls client
"""

from __future__ import annotations

from collections.abc import Callable
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

MockHandler = Callable[[str], BaseModel | dict]


class MockLLMClient(LLMClient):
    """Returns canned responses keyed by label.

    Each label can be registered with either:
        - a static value (BaseModel instance or dict)
        - a callable taking the prompt and returning either of the above
    """

    def __init__(self) -> None:
        super().__init__()
        self._handlers: dict[str, MockHandler] = {}

    def register(self, label: str, response: BaseModel | dict | MockHandler) -> None:
        """Register a response for ``label``.

        If ``response`` is callable, it is invoked with the prompt and must
        return a BaseModel instance or a dict. Otherwise the value is
        returned as-is on every call.
        """
        if callable(response):
            self._handlers[label] = response  # type: ignore[assignment]
        else:
            const = response
            self._handlers[label] = lambda _prompt: const  # type: ignore[return-value]

    def is_registered(self, label: str) -> bool:
        return label in self._handlers

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
    ) -> T:
        ts = self._now_iso()

        if label not in self._handlers:
            err = f"No mock handler registered for label '{label}'"
            self._record(
                LLMCallLog(
                    timestamp=ts,
                    label=label,
                    prompt=prompt,
                    system=system,
                    output_schema_name=output_schema.__name__,
                    raw_output=None,
                    parsed_output=None,
                    attempts=0,
                    success=False,
                    error=err,
                )
            )
            raise LLMOutputError(err)

        raw = self._handlers[label](prompt)

        try:
            if isinstance(raw, output_schema):
                result: T = raw
            elif isinstance(raw, BaseModel):
                # Coerce one Pydantic model into another via dict round-trip
                result = output_schema.model_validate(raw.model_dump())
            elif isinstance(raw, dict):
                result = output_schema.model_validate(raw)
            else:
                raise LLMOutputError(
                    f"Mock handler for '{label}' returned unsupported type "
                    f"{type(raw).__name__}; expected BaseModel or dict"
                )
        except ValidationError as e:
            self._record(
                LLMCallLog(
                    timestamp=ts,
                    label=label,
                    prompt=prompt,
                    system=system,
                    output_schema_name=output_schema.__name__,
                    raw_output=str(raw),
                    parsed_output=None,
                    attempts=1,
                    success=False,
                    error=str(e),
                )
            )
            raise LLMOutputError(f"Mock response failed validation for '{label}': {e}") from e

        self._record(
            LLMCallLog(
                timestamp=ts,
                label=label,
                prompt=prompt,
                system=system,
                output_schema_name=output_schema.__name__,
                raw_output=str(raw),
                parsed_output=result.model_dump(),
                attempts=1,
                success=True,
            )
        )
        return result
