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
You convert a Japanese home-cooking recipe into a directed acyclic graph
(DAG) JSON. A GoalNode is a state of ingredients; a ProcessEdge is an
action that consumes one or more nodes (from_nodes) and produces one
node (to_node). Each edge declares which resources it occupies for how
long.

Output JSON shape:

{
  "title": str,
  "servings": int,
  "nodes": [{"id": str, "description": str, "is_final": bool}],
  "edges": [
    {
      "id": str,
      "from_nodes": [str],
      "to_node": str,
      "action": <chop|slice|mince|peel|boil|boil_water|simmer|saute|fry|bake|steam|grill|microwave|mix|beat|knead|rest|serve|unknown>,
      "description": str,
      "duration_min": float,
      "resource_uses": [
        {
          "kind": <cook|burner|workstation|container|appliance|utensil>,
          "name_hint": str | null,
          "hold_duration_min": float,
          "start_offset_min": float
        }
      ]
    }
  ],
  "final_node_id": str
}

Rules:
1. Create explicit nodes for raw ingredients (e.g. "n_chicken_raw").
2. If a step references already-prepared ingredients ("切った鶏肉を加える"),
   add a missing prep edge that produces the prepared state.
3. ``duration_min`` is the edge's total wall-clock time.
4. ``resource_uses`` MUST be populated. Typical patterns:
   - chop / slice / mince / peel: {cook, hold=duration} + {workstation, hold=duration}
   - simmer / boil: {cook, hold=short attentive part} + {burner, hold=duration}
                    + {container, name_hint="片手鍋"等, hold=duration}
   - saute / fry: {cook, hold=duration} + {burner, hold=duration}
                  + {container, name_hint="フライパン"等, hold=duration}
   - bake / microwave: {cook, hold=short} + {appliance, name_hint, hold=duration}
   - mix / beat: {cook, hold=duration} + {container, name_hint="ボウル"等, hold=duration}
   - serve: {cook, hold=duration}
5. ``hold_duration_min`` ≤ ``duration_min``. For "弱火で15分煮込む":
   duration_min=15, cook hold=1.0 (initial set-up), burner hold=15, container hold=15.
6. Exactly one node has is_final=true; final_node_id matches its id.
7. All from_nodes / to_node values must reference existing node ids.
8. Output the JSON object only — no prose, no markdown fences.
"""


def _build_prompt(recipe: RawRecipe) -> str:
    ingredients_block = "\n".join(f"- {line}" for line in recipe.ingredients_text)
    steps_block = "\n".join(
        f"{i + 1}. {line}" for i, line in enumerate(recipe.steps_text)
    )
    tips_block = f"\n\n## ポイント\n{recipe.tips}" if recipe.tips else ""

    return f"""\
Convert the following recipe into a RecipeDAG JSON object.

Title: {recipe.title}
Servings: {recipe.servings}

## 材料
{ingredients_block}

## 作り方
{steps_block}{tips_block}

Produce the RecipeDAG JSON now (resource_uses required on every edge)."""


class RecipeParser:
    """LLM-bounded parser. Single responsibility: ``RawRecipe → RecipeDAG``."""

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def parse(self, recipe: RawRecipe) -> RecipeDAG:
        prompt = _build_prompt(recipe)
        # ``inject_schema=False`` because backends like LlamaCppClient
        # apply the schema as a sampling-time grammar constraint —
        # the textual schema dump becomes redundant. The compact shape
        # description in SYSTEM_PROMPT remains as a guide for which
        # values to populate (durations, action enum, etc.).
        dag = self.client.generate_structured(
            prompt=prompt,
            output_schema=RecipeDAG,
            system=SYSTEM_PROMPT,
            max_retries=self.max_retries,
            label=PARSER_LABEL,
            inject_schema=False,
        )
        # Stash the original recipe text on the DAG for downstream traceability
        if not dag.raw_text:
            dag = dag.model_copy(update={"raw_text": prompt})
        return dag
