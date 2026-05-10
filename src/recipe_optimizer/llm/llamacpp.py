"""LlamaCppClient: GGUF backend via ``llama-cpp-python``.

For CPU-only environments this is dramatically faster than the
``transformers`` path because llama.cpp uses hand-tuned SIMD kernels
(AVX2/AVX-512) and quantized weights. With Q4_K_M of a 4B-class model
on a modern CPU, expect 5-15 tokens/sec and ~5GB RAM, vs ~0.3-2.7
tokens/sec and ~10GB for the bf16 path.

The class implements the same :class:`LLMClient` contract as
:class:`LocalLLMClient` so callers can swap backends transparently.
"""

from __future__ import annotations

import json
import re
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


SCHEMA_INSTRUCTION_TEMPLATE = """
You must output a single JSON object that conforms exactly to this JSON Schema:

```
{schema}
```

Output rules:
- Output the JSON object and nothing else.
- Do not wrap the JSON in markdown code fences.
- Do not add any prose, explanation, or comments before or after.
"""


class LlamaCppClient(LLMClient):
    """GGUF inference via the ``llama_cpp.Llama`` class."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        # Heavy import deferred so the module can be imported without
        # the GGUF backend installed.
        from llama_cpp import Llama  # noqa: PLC0415

        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=verbose,
            chat_format="gemma",
        )

    # ------------------------------------------------------------------
    # Prompt / output handling
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system: str | None,
        examples: FewShotExamples | None,
        prompt: str,
        output_schema: type[BaseModel],
        inject_schema: bool,
    ) -> list[dict[str, str]]:
        sys_text = system or "You are a precise assistant."
        if inject_schema:
            schema_json = json.dumps(
                output_schema.model_json_schema(), ensure_ascii=False, indent=2
            )
            sys_text += SCHEMA_INSTRUCTION_TEMPLATE.format(schema=schema_json)

        messages: list[dict[str, str]] = [{"role": "system", "content": sys_text}]
        for ex_in, ex_out in examples or []:
            ex_out_text = (
                json.dumps(ex_out.model_dump(), ensure_ascii=False)
                if isinstance(ex_out, BaseModel)
                else json.dumps(ex_out, ensure_ascii=False)
            )
            messages.append({"role": "user", "content": ex_in})
            messages.append({"role": "assistant", "content": ex_out_text})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first balanced ``{...}`` block out of generated text."""
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

    def _generate_chat(
        self,
        messages: list[dict[str, str]],
        context: LLMCallContext,
        json_schema: dict | None = None,
    ) -> str:
        """Run a chat completion and return the assistant's full text.

        When ``json_schema`` is provided, llama.cpp will compile it into
        a GBNF grammar that **strictly constrains sampling** so the model
        cannot produce tokens that would violate the schema. This is
        much more reliable than asking the model in prose to follow a
        shape — small quantized models often invent their own keys
        when given only textual hints.
        """
        kwargs = {
            "messages": messages,
            "max_tokens": context.max_tokens,
            "temperature": context.temperature,
            "stream": False,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }
        result = self.llm.create_chat_completion(**kwargs)
        return result["choices"][0]["message"]["content"] or ""

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
        ctx = context or LLMCallContext()
        ts = self._now_iso()

        messages = self._build_messages(
            system, examples, prompt, output_schema, inject_schema
        )

        # Pass the Pydantic schema as a sampling-time grammar constraint
        # so the model cannot produce tokens that violate the shape.
        json_schema = output_schema.model_json_schema()

        last_error: str | None = None
        raw_text = ""

        for attempt in range(max_retries + 1):
            raw_text = self._generate_chat(messages, ctx, json_schema=json_schema)
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
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output failed JSON-Schema validation:\n"
                                f"{e}\n\n"
                                f"Output a corrected JSON object. "
                                f"JSON only, no prose, no fences."
                            ),
                        }
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
            f"LlamaCpp output for label='{label}' failed validation after retries: "
            f"{last_error}"
        )
