"""Pipeline modules.

Each module is a thin wrapper that either:
  - calls LLMClient.generate_structured with a single bounded contract
    (parser, proposer, renderer), or
  - performs a pure-algorithmic transformation on Pydantic objects
    (checker, selector, rewriter, scheduler).
"""
