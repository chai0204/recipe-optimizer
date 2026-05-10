"""End-to-end optimization pipeline.

Orchestrates the seven modules in order:

    parser → checker → proposer → selector → rewriter → checker(verify)
            → scheduler → output formatters → RenderedRecipe

The renderer LLM call (the third LLM module) is **not yet wired in**;
``RenderedRecipe.numbered_steps`` is currently produced by the
deterministic ``output.numbered_list.to_numbered_steps`` formatter.
When ``modules/renderer.py`` lands, it will produce a polished
natural-language rewrite of the same step list.

Two entry points:

- ``optimize_from_dag(...)``: skip parsing — useful for tests and CLI
  demos using hand-crafted DAG fixtures (no LLM cost for the parser
  step).
- ``optimize_from_raw(...)``: full path — RawRecipe through parser
  through everything. Requires a working LLM client.
"""

from __future__ import annotations

from .llm import LLMClient
from .modules.checker import find_violations
from .modules.parser import RecipeParser
from .modules.proposer import RecipeProposer
from .modules.rewriter import apply_substitutions
from .modules.scheduler import schedule
from .modules.selector import Weights, select_best
from .output import build_shopping_list, to_mermaid, to_numbered_steps
from .schemas import (
    ConstraintViolation,
    Constraints,
    RawRecipe,
    RecipeDAG,
    RenderedRecipe,
    SubstitutionCandidate,
    ToolUseTable,
    UserProfile,
)


class UnresolvableRecipeError(RuntimeError):
    """Raised when, after substitution attempts, the recipe still has
    constraint violations the user cannot satisfy."""

    def __init__(self, violations: list[ConstraintViolation]) -> None:
        msg = (
            f"Recipe could not be fully adapted to the user's constraints. "
            f"Remaining violations: {[v.edge_id for v in violations]}"
        )
        super().__init__(msg)
        self.violations = violations


def optimize_from_dag(
    *,
    dag: RecipeDAG,
    profile: UserProfile,
    table: ToolUseTable,
    client: LLMClient,
    raw_recipe: RawRecipe | None = None,
    selector_weights: Weights | None = None,
    raise_on_unresolvable: bool = True,
) -> RenderedRecipe:
    """Optimize a parsed RecipeDAG end-to-end.

    Args:
        dag: parsed recipe DAG.
        profile: user constraints.
        table: tool-use table (mutated when proposer falls back to LLM).
        client: LLM client (only used by proposer's fallback path here).
        raw_recipe: optional; if given, its ingredients_text populates
            the shopping list.
        selector_weights: optional weighting for substitution selection.
        raise_on_unresolvable: if True, raises ``UnresolvableRecipeError``
            when violations remain after substitution. If False, returns
            a partial RenderedRecipe with whatever could be optimized
            and ``substitutions_made`` reflecting what was tried.

    Returns:
        ``RenderedRecipe`` containing the three output views, the
        schedule, and the substitution audit trail.
    """
    constraints = Constraints(profile=profile)

    # 1. Detect static violations
    violations = find_violations(dag, constraints)

    # 2. For each violation, propose + select + collect
    substitutions: list[SubstitutionCandidate] = []
    if violations:
        proposer = RecipeProposer(client=client, table=table)
        edges_by_id = {e.id: e for e in dag.edges}
        for violation in violations:
            edge = edges_by_id[violation.edge_id]
            candidates = proposer.propose(violation, edge, profile)
            best = select_best(candidates, weights=selector_weights)
            if best is not None:
                substitutions.append(best)

        dag = apply_substitutions(dag, substitutions)

    # 3. Verify resolution
    remaining = find_violations(dag, constraints)
    if remaining and raise_on_unresolvable:
        raise UnresolvableRecipeError(remaining)

    # 4. Schedule
    sched = schedule(dag, constraints)

    # 5. Output formatters
    numbered = to_numbered_steps(dag, sched)
    mermaid = to_mermaid(dag, sched)
    shopping = build_shopping_list(dag, raw_recipe=raw_recipe)

    title = raw_recipe.title if raw_recipe else dag.title
    return RenderedRecipe(
        title=title,
        numbered_steps=numbered,
        mermaid_dag=mermaid,
        shopping_list=shopping,
        schedule=sched,
        optimized_dag=dag,
        substitutions_made=substitutions,
    )


def optimize_from_raw(
    *,
    raw_recipe: RawRecipe,
    profile: UserProfile,
    table: ToolUseTable,
    client: LLMClient,
    selector_weights: Weights | None = None,
    raise_on_unresolvable: bool = True,
) -> RenderedRecipe:
    """Full pipeline: RawRecipe → parser → optimization → RenderedRecipe."""
    parser = RecipeParser(client=client)
    dag = parser.parse(raw_recipe)
    return optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=client,
        raw_recipe=raw_recipe,
        selector_weights=selector_weights,
        raise_on_unresolvable=raise_on_unresolvable,
    )


# Re-export the convenience helper for unsatisfied callers
__all__ = [
    "UnresolvableRecipeError",
    "optimize_from_dag",
    "optimize_from_raw",
]
