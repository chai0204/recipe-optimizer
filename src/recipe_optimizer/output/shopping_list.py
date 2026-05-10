"""Shopping list view (output ③).

Two distinct purposes:

- **Ingredients**: what the user must have on hand (food items).
  Sourced from the original :class:`RawRecipe.ingredients_text`,
  preserved verbatim — parsing free-form Japanese ingredient lines into
  structured (name, quantity, unit) is brittle and not worth the
  complexity for a PoC. Each line is wrapped in an ``Ingredient`` with
  ``name`` set to the full text.

- **Tools**: what equipment must be ready before cooking starts.
  Aggregated from the (post-substitution) DAG's edges, deduplicated by
  ``Tool.name``, and sorted by kind for readability.
"""

from __future__ import annotations

from ..schemas import (
    Ingredient,
    RawRecipe,
    RecipeDAG,
    ShoppingList,
    Tool,
    ToolKind,
)

_KIND_ORDER: dict[ToolKind, int] = {
    ToolKind.HEAT_SOURCE: 0,
    ToolKind.APPLIANCE: 1,
    ToolKind.CONTAINER: 2,
    ToolKind.UTENSIL: 3,
}


def aggregate_tools(dag: RecipeDAG) -> list[Tool]:
    """Unique tools across all edges, sorted by (kind, name) for stability."""
    seen: dict[str, Tool] = {}
    for edge in dag.edges:
        for tool in edge.tools_required:
            seen.setdefault(tool.name, tool)
    return sorted(
        seen.values(),
        key=lambda t: (_KIND_ORDER.get(t.kind, 99), t.name),
    )


def _ingredients_from_text(lines: list[str]) -> list[Ingredient]:
    return [Ingredient(name=line.strip()) for line in lines if line.strip()]


def build_shopping_list(
    dag: RecipeDAG,
    raw_recipe: RawRecipe | None = None,
) -> ShoppingList:
    """Construct a ShoppingList from the optimized DAG.

    If ``raw_recipe`` is provided, its ``ingredients_text`` is used to
    populate the ingredients list. Otherwise the list is empty —
    callers are expected to supply ingredient data from elsewhere.
    """
    ingredients = (
        _ingredients_from_text(raw_recipe.ingredients_text)
        if raw_recipe is not None
        else []
    )
    return ShoppingList(
        ingredients=ingredients,
        tools_needed=aggregate_tools(dag),
    )
