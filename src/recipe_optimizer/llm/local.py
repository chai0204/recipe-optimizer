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
        torch_dtype: str = "bfloat16",
        stream_to_stdout: bool = False,
    ) -> None:
        super().__init__()

        # Heavy imports kept local so that ``import recipe_optimizer.llm``
        # without using LocalLLMClient does not pull in torch.
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        # gemma-4-E4B-it is a multimodal (image+audio+text) model whose
        # architecture is ``Gemma4ForConditionalGeneration``. The right
        # auto class is ``AutoModelForImageTextToText``. For text-only
        # inputs we simply omit images/audio in the chat template.
        # We try the multimodal path first and fall back to plain
        # CausalLM for text-only models like gemma-2-2b-it.
        try:
            from transformers import AutoModelForImageTextToText  # noqa: PLC0415

            ModelClass = AutoModelForImageTextToText
        except ImportError:  # pragma: no cover — older transformers
            from transformers import AutoModelForCausalLM  # noqa: PLC0415

            ModelClass = AutoModelForCausalLM

        self.model_id = model_id
        self.device = device
        self.stream_to_stdout = stream_to_stdout
        dtype = getattr(torch, torch_dtype)

        # AutoProcessor wraps tokenizer + image_processor + audio_feature_extractor.
        # For text-only models this still works (it just exposes tokenizer pieces).
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Some processors expose tokenizer directly; some do via processor.tokenizer.
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # transformers ≥5.0 deprecated ``torch_dtype`` in favour of ``dtype``.
        self.model = ModelClass.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
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
        inject_schema: bool = True,
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

    @staticmethod
    def _normalize_messages_for_multimodal(
        messages: list[dict[str, str]],
    ) -> list[dict]:
        """Multimodal processors (Gemma4, PaliGemma) expect ``content`` as a
        list of ``{type, text|image|audio}`` dicts, not a bare string.

        Convert ``{"role": "user", "content": "hi"}`` →
        ``{"role": "user", "content": [{"type": "text", "text": "hi"}]}``.
        Already-structured content (list of dicts) is left untouched.
        """
        normalized: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                normalized.append({**msg, "content": [{"type": "text", "text": content}]})
            else:
                normalized.append(dict(msg))
        return normalized

    def _generate_text(
        self, messages: list[dict[str, str]], context: LLMCallContext
    ) -> str:
        import torch  # noqa: PLC0415

        # Prefer the processor's apply_chat_template (multimodal-aware) and
        # fall back to the tokenizer's for plain causal-LM models.
        applier = getattr(self.processor, "apply_chat_template", None) or (
            self.tokenizer.apply_chat_template
        )

        # Multimodal processors usually need typed content blocks.
        prepared_messages = self._normalize_messages_for_multimodal(messages)

        try:
            inputs = applier(
                prepared_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            # Older transformers / plain tokenizer: returns a tensor of ids
            ids = applier(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": ids}

        # Move all tensor inputs to the target device
        moved: dict[str, "torch.Tensor"] = {}
        for k, v in dict(inputs).items():
            moved[k] = v.to(self.device) if hasattr(v, "to") else v

        prompt_len = moved["input_ids"].shape[1]

        # Optional streaming: prints tokens to stdout as they are produced,
        # giving real-time visibility for long generations on CPU.
        streamer = None
        if self.stream_to_stdout:
            from transformers import TextStreamer  # noqa: PLC0415

            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

        with torch.no_grad():
            output = self.model.generate(
                **moved,
                max_new_tokens=context.max_tokens,
                do_sample=context.temperature > 0.0,
                temperature=max(context.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

        generated = output[0][prompt_len:]
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
        inject_schema: bool = True,
    ) -> T:
        ctx = context or LLMCallContext()
        ts = self._now_iso()

        messages = self._build_messages(
            system, examples, prompt, output_schema, inject_schema=inject_schema
        )
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
