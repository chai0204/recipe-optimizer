"""Shopping list view (output ③).

- **Ingredients**: from ``RawRecipe.ingredients_text`` verbatim (1 line
  per Ingredient, no parsing of quantities).
- **Resources needed**: collected from every ``ResourceRequirement`` on
  every edge. We surface resources that *uniquely identify* a piece of
  equipment — i.e. ones with a ``name_hint`` or a kind that's normally
  enumerated (CONTAINER, APPLIANCE, BURNER). Abstract pools like COOK
  and WORKSTATION are not listed (the user knows they are the cook).
"""

from __future__ import annotations

from ..schemas import (
    Ingredient,
    RawRecipe,
    RecipeDAG,
    Resource,
    ResourceKind,
    ShoppingList,
)

_KIND_ORDER: dict[ResourceKind, int] = {
    ResourceKind.BURNER: 0,
    ResourceKind.APPLIANCE: 1,
    ResourceKind.CONTAINER: 2,
    ResourceKind.UTENSIL: 3,
    ResourceKind.WORKSTATION: 4,
    ResourceKind.COOK: 5,
}

_LISTABLE_KINDS = {
    ResourceKind.CONTAINER,
    ResourceKind.APPLIANCE,
    ResourceKind.BURNER,
    ResourceKind.UTENSIL,
}


def aggregate_resources(dag: RecipeDAG) -> list[Resource]:
    """Unique resources demanded by the DAG, sorted by kind then name.

    Each resource is identified by ``(kind, name_hint)``. Requirements
    without a ``name_hint`` (e.g. anonymous COOK / BURNER pool) are
    folded into a single placeholder per kind for display.
    """
    seen: dict[tuple[ResourceKind, str], Resource] = {}
    for edge in dag.edges:
        for use in edge.resource_uses:
            if use.kind not in _LISTABLE_KINDS:
                continue
            name = use.name_hint or f"({use.kind.value} any)"
            key = (use.kind, name)
            if key not in seen:
                seen[key] = Resource(
                    id=f"need_{use.kind.value}_{name}",
                    kind=use.kind,
                    name=name,
                )
    return sorted(
        seen.values(),
        key=lambda r: (_KIND_ORDER.get(r.kind, 99), r.name),
    )


def _ingredients_from_text(lines: list[str]) -> list[Ingredient]:
    return [Ingredient(name=line.strip()) for line in lines if line.strip()]


def build_shopping_list(
    dag: RecipeDAG,
    raw_recipe: RawRecipe | None = None,
) -> ShoppingList:
    ingredients = (
        _ingredients_from_text(raw_recipe.ingredients_text)
        if raw_recipe is not None
        else []
    )
    return ShoppingList(
        ingredients=ingredients,
        resources_needed=aggregate_resources(dag),
    )
