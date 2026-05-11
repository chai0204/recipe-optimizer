"""Tests for the rewriter under the resource model."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile, load_table
from recipe_optimizer.llm import MockLLMClient
from recipe_optimizer.modules.checker import find_violations, is_satisfiable
from recipe_optimizer.modules.proposer import RecipeProposer
from recipe_optimizer.modules.rewriter import apply_substitution, apply_substitutions
from recipe_optimizer.modules.selector import select_best
from recipe_optimizer.schemas import (
    Constraints,
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    ResourceKind,
    ResourceRequirement,
    SubstitutionCandidate,
)

from .fixtures.mugicha_dag import make_mugicha_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


def _yakan_dag() -> RecipeDAG:
    return RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="water", description="水"),
            GoalNode(id="boiling", description="沸騰水", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e_boil",
                from_nodes=["water"],
                to_node="boiling",
                action=ProcessType.BOIL_WATER,
                description="やかんで湯を沸かす",
                duration_min=5.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0),
                    ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=5.0),
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="やかん",
                        hold_duration_min=5.0,
                    ),
                ],
            )
        ],
        final_node_id="boiling",
    )


def _pot_replacement_for_e_boil() -> SubstitutionCandidate:
    replacement = ProcessEdge(
        id="e_boil",
        from_nodes=["water"],
        to_node="boiling",
        action=ProcessType.BOIL_WATER,
        description="片手鍋で湯を沸かす",
        duration_min=5.75,
        resource_uses=[
            ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.15),
            ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=5.75),
            ResourceRequirement(
                kind=ResourceKind.CONTAINER,
                name_hint="片手鍋",
                hold_duration_min=5.75,
            ),
        ],
    )
    return SubstitutionCandidate(
        original_edge_id="e_boil",
        replacement_edge=replacement,
        rationale="やかん→片手鍋",
        time_delta_min=0.75,
    )


# ---------------------------------------------------------------------------
# apply_substitution
# ---------------------------------------------------------------------------


def test_apply_substitution_swaps_resource_uses():
    dag = _yakan_dag()
    new_dag = apply_substitution(dag, _pot_replacement_for_e_boil())

    [new_edge] = new_dag.edges
    hints = {u.name_hint for u in new_edge.resource_uses}
    assert "片手鍋" in hints
    assert "やかん" not in hints
    assert new_edge.id == "e_boil"


def test_apply_substitution_preserves_topology():
    dag = _yakan_dag()
    new_dag = apply_substitution(dag, _pot_replacement_for_e_boil())
    [edge] = new_dag.edges
    assert edge.from_nodes == ["water"]
    assert edge.to_node == "boiling"


def test_apply_substitution_does_not_mutate_input():
    dag = _yakan_dag()
    apply_substitution(dag, _pot_replacement_for_e_boil())
    original_hints = {u.name_hint for u in dag.edges[0].resource_uses}
    assert "やかん" in original_hints


def test_apply_substitution_raises_on_missing_edge():
    dag = _yakan_dag()
    bogus = SubstitutionCandidate(
        original_edge_id="does_not_exist",
        replacement_edge=ProcessEdge(
            id="does_not_exist",
            from_nodes=["water"],
            to_node="boiling",
            action=ProcessType.BOIL_WATER,
            description="x",
            duration_min=1.0,
        ),
        rationale="",
    )
    with pytest.raises(ValueError, match="not found in DAG"):
        apply_substitution(dag, bogus)


def test_apply_substitution_rejects_inconsistent_replacement():
    from pydantic import ValidationError

    dag = _yakan_dag()
    bad = ProcessEdge(
        id="e_boil",
        from_nodes=["water"],
        to_node="ghost_node",  # not in dag.nodes
        action=ProcessType.BOIL_WATER,
        description="x",
        duration_min=1.0,
    )
    cand = SubstitutionCandidate(
        original_edge_id="e_boil",
        replacement_edge=bad,
        rationale="",
    )
    with pytest.raises(ValidationError):
        apply_substitution(dag, cand)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_resolves_mugicha():
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    proposer = RecipeProposer(client=MockLLMClient(), table=table)

    dag = make_mugicha_dag()
    constraints = Constraints(profile=profile)
    edges_by_id = {e.id: e for e in dag.edges}

    violations = find_violations(dag, constraints)
    assert len(violations) >= 2  # at least one per edge

    seen_edges = set()
    chosen: list[SubstitutionCandidate] = []
    for v in violations:
        if v.edge_id in seen_edges:
            continue
        seen_edges.add(v.edge_id)
        cands = proposer.propose(v, edges_by_id[v.edge_id], profile)
        best = select_best(cands)
        if best is not None:
            chosen.append(best)

    new_dag = apply_substitutions(dag, chosen)
    assert is_satisfiable(new_dag, constraints)
    for edge in new_dag.edges:
        hints = {u.name_hint for u in edge.resource_uses}
        assert "やかん" not in hints
