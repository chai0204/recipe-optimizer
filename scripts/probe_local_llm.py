"""Probe script for the LocalLLMClient + gemma-4-E4B-it integration.

Stages — run independently so we can stop at the first failure:

  1. plain-text generation: confirm the model loads and produces output
  2. structured output (small): confirm Pydantic schema injection works
  3. parser end-to-end on mugicha: confirm the real LLM parses a recipe
     and produces a valid RecipeDAG with sensible duration_min values

Each stage prints a header and timings so we can compare against the
0.3-0.5 tok/s theoretical estimate for CPU bf16 inference of a 4B model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def stage_1_plain_text():
    print("\n=== Stage 1: plain-text generation ===")
    from recipe_optimizer.llm.local import LocalLLMClient

    t0 = time.time()
    client = LocalLLMClient()
    print(f"  load: {time.time() - t0:.1f}s")

    from recipe_optimizer.llm.client import LLMCallContext

    messages = [
        {"role": "user", "content": "次の文章を一文で日本語に訳して: 'I love cooking.'"},
    ]
    t1 = time.time()
    out = client._generate_text(messages, LLMCallContext(max_tokens=64))
    elapsed = time.time() - t1
    print(f"  generate: {elapsed:.1f}s")
    print(f"  output: {out!r}")
    return client


def stage_2_structured_small(client):
    print("\n=== Stage 2: structured small ===")
    from pydantic import BaseModel

    class Greeting(BaseModel):
        language: str
        greeting: str

    t0 = time.time()
    result = client.generate_structured(
        prompt="Output a Japanese greeting.",
        output_schema=Greeting,
        label="probe_stage2",
    )
    print(f"  generate: {time.time() - t0:.1f}s")
    print(f"  output: {result.model_dump_json(ensure_ascii=False)}")
    return client


def stage_3_parser_mugicha(client):
    print("\n=== Stage 3: parser on mugicha (full real-LLM path) ===")
    from recipe_optimizer.modules.parser import RecipeParser
    from recipe_optimizer.schemas import RawRecipe

    raw = RawRecipe(
        id="mugicha",
        title="麦茶",
        servings=4,
        ingredients_text=["水 1L", "麦茶パック 1個"],
        steps_text=[
            "やかんに水を入れ、強火で沸騰させる",
            "火を止めて麦茶パックを入れ、10分蒸らす",
            "粗熱を取り冷蔵庫で冷やす",
        ],
    )

    parser = RecipeParser(client=client)
    t0 = time.time()
    dag = parser.parse(raw)
    print(f"  parse: {time.time() - t0:.1f}s")

    print(f"  nodes ({len(dag.nodes)}):")
    for n in dag.nodes:
        print(f"    {n.id}: {n.description!r}{' (final)' if n.is_final else ''}")
    print(f"  edges ({len(dag.edges)}):")
    for e in dag.edges:
        tools = [t.name for t in e.tools_required]
        print(
            f"    {e.id}: {e.action.value} {tools} "
            f"dur={e.duration_min} attentive={e.attentive_min}"
        )
    print(f"  final_node_id: {dag.final_node_id}")


def main():
    stages = sys.argv[1:] or ["1", "2", "3"]
    client = None
    if "1" in stages:
        client = stage_1_plain_text()
    if "2" in stages:
        if client is None:
            from recipe_optimizer.llm.local import LocalLLMClient

            client = LocalLLMClient()
        stage_2_structured_small(client)
    if "3" in stages:
        if client is None:
            from recipe_optimizer.llm.local import LocalLLMClient

            client = LocalLLMClient()
        stage_3_parser_mugicha(client)


if __name__ == "__main__":
    main()
