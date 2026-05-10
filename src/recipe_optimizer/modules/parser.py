"""Recipe parser: RawRecipe (text) → RecipeDAG (structured graph).

This is the first LLM-bounded module. The contract is narrow:

    RawRecipe → RecipeDAG

The LLM's responsibility is to identify cooking states (GoalNodes)
and the actions that connect them (ProcessEdges), forming a directed
acyclic graph that terminates at a single ``final_node_id``.

DAG structural validity is enforced by ``RecipeDAG``'s own
``model_validator`` — if the LLM produces dangling edge references or
duplicate node IDs, validation fails and ``LLMClient`` retries with the
error in the feedback turn.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..schemas import ProcessType, RawRecipe, RecipeDAG

PARSER_LABEL = "parser"


SYSTEM_PROMPT = """\
You are a precise recipe analyzer. You convert a Japanese home-cooking
recipe into a directed acyclic graph (DAG) describing the cooking process.

Concepts:
- A GoalNode is a state of ingredients (raw, prepared, partially cooked,
  or finished). Each node has a stable ``id`` (e.g., "n_onion_sliced") and
  a short Japanese description (e.g., "薄切りにした玉ねぎ").
- A ProcessEdge is an action that consumes one or more input states
  (``from_nodes``) and produces a single output state (``to_node``).

Required behaviour:

1. Create explicit nodes for raw ingredients that appear in the recipe
   (e.g., "n_chicken_raw"), even if the recipe text does not mention
   them as a separate step.
2. Detect implicit prep steps. If a step says "切った鶏肉を加える" but
   no prior step explicitly cuts the chicken, you MUST add an edge for
   "鶏肉を切る" producing the cut state.
3. Each ProcessEdge specifies the canonical ``action`` from this set:
   chop, slice, mince, peel, boil, boil_water, simmer, saute, fry,
   bake, steam, grill, microwave, mix, beat, knead, rest, serve,
   unknown. Use ``unknown`` only when no other choice fits.
4. Each ProcessEdge specifies ``tools_required`` (list of Tool objects
   with ``name`` and ``kind``). ``kind`` ∈ {heat_source, container,
   utensil, appliance}. Composite tools may be written as e.g.
   "包丁+まな板" with ``kind: utensil``.
5. ``duration_min`` is wall-clock time in minutes. ``attentive_min`` is
   how long the cook cannot leave (≤ duration_min). For "煮る30秒",
   set duration_min=0.5 attentive_min=0.5. For "弱火で15分煮込む",
   duration_min=15 but attentive_min may be 1.0 (occasional check).
6. Output exactly one final node with ``is_final: true`` and reference
   it as ``final_node_id``.
7. All ``from_nodes`` and ``to_node`` references MUST be node IDs that
   exist in your ``nodes`` array. No dangling references.
8. Output JSON only. No prose, no markdown fences.
"""


def _build_prompt(recipe: RawRecipe) -> str:
    ingredients_block = "\n".join(f"- {line}" for line in recipe.ingredients_text)
    steps_block = "\n".join(
        f"{i + 1}. {line}" for i, line in enumerate(recipe.steps_text)
    )
    tips_block = f"\n\n## ポイント\n{recipe.tips}" if recipe.tips else ""
    valid_actions = ", ".join(p.value for p in ProcessType)

    return f"""\
Convert the following recipe into a RecipeDAG JSON object.

Title: {recipe.title}
Servings: {recipe.servings}

## 材料
{ingredients_block}

## 作り方
{steps_block}{tips_block}

Valid action values for ProcessEdge.action: {valid_actions}

Produce the RecipeDAG JSON now."""


class RecipeParser:
    """LLM-bounded parser. Single responsibility: ``RawRecipe → RecipeDAG``."""

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def parse(self, recipe: RawRecipe) -> RecipeDAG:
        prompt = _build_prompt(recipe)
        dag = self.client.generate_structured(
            prompt=prompt,
            output_schema=RecipeDAG,
            system=SYSTEM_PROMPT,
            max_retries=self.max_retries,
            label=PARSER_LABEL,
        )
        # Stash the original recipe text on the DAG for downstream traceability
        if not dag.raw_text:
            dag = dag.model_copy(update={"raw_text": prompt})
        return dag
