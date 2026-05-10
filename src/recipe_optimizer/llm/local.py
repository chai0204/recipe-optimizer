"""LocalLLMClient: HuggingFace transformers backend.

Loads a local causal-LM (default: ``google/gemma-4-E4B-it``) and uses
its chat template to generate JSON that conforms to a Pydantic schema.

Heavy imports (torch, transformers) are deferred to ``__init__`` so that
mock-only development (and tests) does not pay the cost.

Strategy for structured output without native function-calling support:

1. Inject the JSON schema of the target Pydantic model into the system
   message and instruct the model to emit JSON only.
2. After generation, strip code fences and extract the first balanced
   ``{...}`` substring.
3. Validate against the Pydantic schema. On failure, prepend the
   validation error to the next attempt and retry.

For tighter constraint we may later swap in ``outlines`` /
``lm-format-enforcer`` / GBNF grammars (llama-cpp), but the boundary
stays the same: ``LLMClient.generate_structured``.
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


class LocalLLMClient(LLMClient):
    """Run a HuggingFace causal-LM locally on CPU/GPU."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        super().__init__()

        # Heavy imports kept local so that ``import recipe_optimizer.llm``
        # without using LocalLLMClient does not pull in torch.
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_id = model_id
        self.device = device
        dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt / output handling
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system: str | None,
        examples: FewShotExamples | None,
        prompt: str,
        output_schema: type[BaseModel],
    ) -> list[dict[str, str]]:
        schema_json = json.dumps(
            output_schema.model_json_schema(), ensure_ascii=False, indent=2
        )
        sys_text = (system or "You are a precise assistant.") + SCHEMA_INSTRUCTION_TEMPLATE.format(
            schema=schema_json
        )

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

    def _generate_text(
        self, messages: list[dict[str, str]], context: LLMCallContext
    ) -> str:
        import torch  # noqa: PLC0415

        chat_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                chat_input,
                max_new_tokens=context.max_tokens,
                do_sample=context.temperature > 0.0,
                temperature=max(context.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][chat_input.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

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
    ) -> T:
        ctx = context or LLMCallContext()
        ts = self._now_iso()

        messages = self._build_messages(system, examples, prompt, output_schema)
        last_error: str | None = None
        raw_text = ""

        for attempt in range(max_retries + 1):
            raw_text = self._generate_text(messages, ctx)
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
                    # Append error feedback for the next attempt
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output failed JSON-Schema validation:\n"
                                f"{e}\n\n"
                                f"Output a corrected JSON object conforming to the schema. "
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
            f"Local LLM output for label='{label}' failed validation after retries: {last_error}"
        )
