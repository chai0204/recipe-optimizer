"""LLM client subpackage.

Note: ``LocalLLMClient`` is intentionally NOT re-exported here because
its module imports ``torch``/``transformers`` at runtime. Import it
explicitly when needed:

    >>> from recipe_optimizer.llm.local import LocalLLMClient
"""

from .client import (
    FewShotExamples,
    LLMCallContext,
    LLMCallLog,
    LLMClient,
    LLMOutputError,
)
from .mock import MockHandler, MockLLMClient

__all__ = [
    "FewShotExamples",
    "LLMCallContext",
    "LLMCallLog",
    "LLMClient",
    "LLMOutputError",
    "MockHandler",
    "MockLLMClient",
]
