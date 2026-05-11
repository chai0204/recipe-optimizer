"""Probe script for the LLM integration.

Backend: ``LlamaCppClient`` (GGUF Q4_K_M of unsloth/gemma-4-E4B-it).
GGUF + llama.cpp on CPU is roughly an order of magnitude faster than
transformers bf16 because of int4 weights and SIMD-tuned kernels.

Stages:

  1. plain-text generation (load + 1 inference)
  2. structured output (small Pydantic model)
  3. parser on a minimal recipe (real RawRecipe → RecipeDAG path)

Run with ``python -u`` so progress prints flush in real time.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


GGUF_REPO = "unsloth/gemma-4-E4B-it-GGUF"
GGUF_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"


def log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_gguf_path() -> str:
    """Return the locally cached path to the Q4_K_M GGUF, downloading
    on demand. Honours ``RECIPE_OPTIMIZER_GGUF_PATH`` env override for
    tests / alternative quantizations."""
    override = os.environ.get("RECIPE_OPTIMIZER_GGUF_PATH")
    if override:
        return override
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(GGUF_REPO, GGUF_FILE)


def _new_client():
    from recipe_optimizer.llm.llamacpp import LlamaCppClient  # noqa: PLC0415

    log(f"  resolving GGUF path ({GGUF_REPO} / {GGUF_FILE}) ...")
    path = _resolve_gguf_path()
    log(f"  path: {path}")
    log("  instantiating LlamaCppClient (loads quantized weights, ~10-20s) ...")
    t0 = time.time()
    client = LlamaCppClient(model_path=path)
    log(f"  load: {time.time() - t0:.1f}s")
    return client


def stage_1_plain_text():
    log("\n=== Stage 1: plain-text generation (GGUF) ===")
    client = _new_client()

    from recipe_optimizer.llm.client import LLMCallContext

    messages = [
        {"role": "user", "content": "次の文章を一文で日本語に訳して: 'I love cooking.'"},
    ]
    log("  calling _generate_chat (target: 64 tokens) ...")
    t1 = time.time()
    out = client._generate_chat(messages, LLMCallContext(max_tokens=64))
    elapsed = time.time() - t1
    log(f"  generate: {elapsed:.1f}s")
    log(f"  output: {out!r}")
    return client


def stage_2_structured_small(client):
    log("\n=== Stage 2: structured small (Greeting schema) ===")
    from pydantic import BaseModel

    class Greeting(BaseModel):
        language: str
        greeting: str

    log("  calling generate_structured (target: short JSON) ...")
    t0 = time.time()
    result = client.generate_structured(
        prompt="Output a Japanese greeting.",
        output_schema=Greeting,
        label="probe_stage2",
    )
    log(f"  generate: {time.time() - t0:.1f}s")
    log(f"  output: {result.model_dump_json()}")


def stage_3_parser_minimal(client):
    """Parser smoke test with the tiniest possible recipe."""
    log("\n=== Stage 3: parser on a minimal recipe ===")
    log("  building RawRecipe ...")
    from recipe_optimizer.modules.parser import RecipeParser
    from recipe_optimizer.schemas import RawRecipe

    raw = RawRecipe(
        id="hot_water",
        title="お湯を沸かす",
        servings=1,
        ingredients_text=["水 200ml"],
        steps_text=["やかんに水を入れて強火で沸騰させる"],
    )

    log("  invoking RecipeParser.parse() — compact prompt, no retries ...")
    parser = RecipeParser(client=client, max_retries=0)
    t0 = time.time()
    try:
        dag = parser.parse(raw)
    except Exception as exc:
        log(f"  parse failed after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        if client.call_logs:
            last = client.call_logs[-1]
            log(f"  raw_output (first 1000 chars): {(last.raw_output or '')[:1000]!r}")
        raise
    log(f"  parse: {time.time() - t0:.1f}s")

    log(f"  nodes ({len(dag.nodes)}):")
    for n in dag.nodes:
        marker = " (final)" if n.is_final else ""
        log(f"    {n.id}: {n.description!r}{marker}")
    log(f"  edges ({len(dag.edges)}):")
    for e in dag.edges:
        uses = [
            f"{u.kind.value}({u.name_hint or '*'},{u.hold_duration_min}m)"
            for u in e.resource_uses
        ]
        log(
            f"    {e.id}: {e.action.value} dur={e.duration_min} "
            f"resource_uses=[{', '.join(uses)}]"
        )
    log(f"  final_node_id: {dag.final_node_id}")


def main():
    requested = sys.argv[1:] or ["1", "2", "3"]
    log(f"Stages requested: {requested}")
    log(f"Started at {time.strftime('%H:%M:%S')}")

    client = None
    if "1" in requested:
        client = stage_1_plain_text()
    if "2" in requested:
        if client is None:
            client = _new_client()
        stage_2_structured_small(client)
    if "3" in requested:
        if client is None:
            client = _new_client()
        stage_3_parser_minimal(client)

    log(f"\nFinished at {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
