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
from .llm import LLMOutputError
import networkx as nx

from .modules.checker import find_violations
from .modules.parser import RecipeParser
from .modules.proposer import RecipeProposer
from .modules.renderer import RecipeRenderer
from .modules.rewriter import apply_substitution
from .modules.scheduler import schedule
from .modules.selector import Weights, select_best
from .output import build_shopping_list, to_mermaid, to_numbered_steps
from .schemas import (
    ConstraintViolation,
    Constraints,
    ProcessEdge,
    RawRecipe,
    RecipeDAG,
    RenderedRecipe,
    ResourceKind,
    Schedule,
    SubstitutionCandidate,
    ToolUseTable,
    UserProfile,
)


def _build_edge_dag(dag: RecipeDAG) -> nx.DiGraph:
    """Edge dependency graph: A → B iff A.to_node ∈ B.from_nodes."""
    edges_producing: dict[str, str] = {e.to_node: e.id for e in dag.edges}
    g: nx.DiGraph = nx.DiGraph()
    for e in dag.edges:
        g.add_node(e.id)
    for e in dag.edges:
        for from_node in e.from_nodes:
            pred = edges_producing.get(from_node)
            if pred is not None:
                g.add_edge(pred, e.id)
    return g


def _upstream_container_names(
    dag: RecipeDAG, edge_id: str, edge_g: nx.DiGraph
) -> set[str]:
    """Collect container / appliance ``name_hint``s used by every edge
    that transitively produces input to ``edge_id``. Carries forward
    the cook's pot/bowl choice into downstream steps so the substitution
    layer can prefer continuity."""
    by_id = {e.id: e for e in dag.edges}
    visited: set[str] = set()
    out: set[str] = set()
    container_kinds = {ResourceKind.CONTAINER, ResourceKind.APPLIANCE}

    def walk(eid: str) -> None:
        for pred in edge_g.predecessors(eid):
            if pred in visited:
                continue
            visited.add(pred)
            edge = by_id[pred]
            for use in edge.resource_uses:
                if use.kind in container_kinds and use.name_hint:
                    out.add(use.name_hint)
            walk(pred)

    walk(edge_id)
    return out


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
    use_renderer: bool = False,
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

    # 2. Substitute in topological order so downstream edges see the
    #    containers their predecessors committed to. Each violation is
    #    resolved on the up-to-date DAG and the substitution is applied
    #    immediately, so later proposer calls receive a coherent
    #    ``preferred_resource_names`` hint.
    substitutions: list[SubstitutionCandidate] = []
    if violations:
        proposer = RecipeProposer(client=client, table=table)
        edge_g = _build_edge_dag(dag)
        topo_edge_ids = list(nx.topological_sort(edge_g))

        # Group violations by edge id; ordering inside an edge follows
        # ``find_violations``' deterministic resource-use order.
        violations_by_edge: dict[str, list[ConstraintViolation]] = {}
        for v in violations:
            violations_by_edge.setdefault(v.edge_id, []).append(v)

        for eid in topo_edge_ids:
            edge_violations = violations_by_edge.get(eid)
            if not edge_violations:
                continue
            # Look up the *current* edge (may have been rewritten already
            # by an earlier substitution on the same edge id).
            current_edges_by_id = {e.id: e for e in dag.edges}
            current_edge = current_edges_by_id[eid]
            preferred = _upstream_container_names(dag, eid, edge_g)

            for v in edge_violations:
                candidates = proposer.propose(
                    v,
                    current_edge,
                    profile,
                    preferred_resource_names=preferred,
                )
                best = select_best(candidates, weights=selector_weights)
                if best is None:
                    continue
                substitutions.append(best)
                dag = apply_substitution(dag, best)
                # Refresh the current edge for any remaining violations
                # on the same edge id.
                current_edge = next(e for e in dag.edges if e.id == eid)

    # 3. Verify resolution
    remaining = find_violations(dag, constraints)
    if remaining and raise_on_unresolvable:
        raise UnresolvableRecipeError(remaining)

    # 4. Schedule. If unresolvable but caller is tolerating it, skip the
    #    scheduler (which would raise on missing resources) and return an
    #    empty schedule as the partial result.
    if remaining:
        sched = Schedule(steps=[], total_duration_min=0.0, critical_path_edge_ids=[])
    else:
        sched = schedule(dag, constraints)

    # 5. Output formatters
    numbered = to_numbered_steps(dag, sched)
    if use_renderer and not remaining and sched.steps:
        # Opt-in LLM polish. Falls back silently to the deterministic
        # numbered_steps if the client can't render (e.g. Mock without
        # a registered handler).
        try:
            renderer = RecipeRenderer(client=client)
            numbered = renderer.render(dag, sched, raw_recipe=raw_recipe)
        except LLMOutputError:
            pass
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
    use_renderer: bool = False,
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
        use_renderer=use_renderer,
    )


# Re-export the convenience helper for unsatisfied callers
__all__ = [
    "UnresolvableRecipeError",
    "optimize_from_dag",
    "optimize_from_raw",
]
