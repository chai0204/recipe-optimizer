"""Renderer: scheduled DAG → polished Japanese numbered steps.

Third LLM-bounded module. The parser's ``edge.description`` works as a
fallback but tends to be verbose (e.g. concatenating six seasonings into
a single mix step, or carrying long parenthetical substitution notes
like "（片手鍋で代替）"). The renderer rewrites the step list into the
style of a home-cooking recipe: concise sentences, ingredient names
woven in naturally, timing called out where it matters.

Contract:

    (RecipeDAG, Schedule, RawRecipe | None) → list[str]

Same Mock-first discipline as the other LLM modules — callers that
pass a Mock client (or any client lacking a registered handler) can
fall back to ``output.to_numbered_steps`` deterministically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..schemas import RawRecipe, RecipeDAG, Schedule

RENDERER_LABEL = "renderer"


class RenderedSteps(BaseModel):
    """LLM output schema."""

    steps: list[str] = Field(default_factory=list, min_length=1)


SYSTEM_PROMPT = """\
You are a recipe writer producing the final cooking instructions for a
Japanese home cook. You receive:

- The original recipe (title, ingredients, raw steps, tips)
- The optimised DAG, where each edge is a cooking action with its own
  duration and any substitutions already applied
- A schedule giving each edge a start/end time within the wall-clock plan

Produce a clean list of numbered steps in natural Japanese (家庭料理レシピ調).
Constraints:

- Combine related fine-grained edges (e.g. "醤油+みりん+砂糖を混ぜる" →
  one line "合わせ調味料を作る")
- Mention concrete quantities from the ingredients list when relevant
- Spell out timing in natural language ("中火で2分" rather than "duration 2.0min")
- When substitutions were applied (e.g. やかん→片手鍋), write the step
  using the substitute tool without parenthetical commentary
- Don't add steps that aren't in the DAG
- Order strictly by schedule.start_min (chronological)

Output JSON: {"steps": ["1の手順", "2の手順", ...]}.
"""


def _build_prompt(
    dag: RecipeDAG,
    schedule: Schedule,
    raw_recipe: RawRecipe | None,
) -> str:
    edges_by_id = {e.id: e for e in dag.edges}
    chronological = sorted(schedule.steps, key=lambda s: (s.start_min, s.edge_id))

    raw_block = ""
    if raw_recipe is not None:
        ing = "\n".join(f"- {line}" for line in raw_recipe.ingredients_text)
        original = "\n".join(
            f"{i + 1}. {line}" for i, line in enumerate(raw_recipe.steps_text)
        )
        raw_block = f"\n## 元レシピ（参考）\n\n### 材料\n{ing}\n\n### 作り方\n{original}\n"
        if raw_recipe.tips:
            raw_block += f"\n### ポイント\n{raw_recipe.tips}\n"

    sched_lines = []
    for step in chronological:
        e = edges_by_id[step.edge_id]
        sched_lines.append(
            f"  [{step.start_min:.1f}-{step.end_min:.1f}min] "
            f"id={e.id} action={e.action.value} duration={e.duration_min} "
            f"desc={e.description!r}"
        )
    sched_block = "\n".join(sched_lines)

    return f"""\
Recipe title: {dag.title}
Servings: {dag.servings}
Total makespan: {schedule.total_duration_min:.1f} min
{raw_block}
## 最適化済み DAG（実行順）
{sched_block}

Produce the polished numbered steps now."""


class RecipeRenderer:
    """LLM-bounded renderer. ``RecipeDAG + Schedule → list[str]``."""

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self.client = client
        self.max_retries = max_retries

    def render(
        self,
        dag: RecipeDAG,
        schedule: Schedule,
        raw_recipe: RawRecipe | None = None,
    ) -> list[str]:
        prompt = _build_prompt(dag, schedule, raw_recipe)
        out = self.client.generate_structured(
            prompt=prompt,
            output_schema=RenderedSteps,
            system=SYSTEM_PROMPT,
            max_retries=self.max_retries,
            label=RENDERER_LABEL,
            inject_schema=False,
        )
        return out.steps
