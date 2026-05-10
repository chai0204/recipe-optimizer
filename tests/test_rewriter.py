"""Tests for the rewriter module (pure-algorithmic DAG rewriting)."""

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
    SubstitutionCandidate,
    Tool,
    ToolKind,
)

from .fixtures.mugicha_dag import make_mugicha_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


def _trivial_dag_with_yakan_edge() -> RecipeDAG:
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
                tools_required=[
                    Tool(name="やかん", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=5.0,
                attentive_min=1.0,
            )
        ],
        final_node_id="boiling",
    )


def _candidate_pot_for_e_boil() -> SubstitutionCandidate:
    replacement = ProcessEdge(
        id="e_boil",
        from_nodes=["water"],
        to_node="boiling",
        action=ProcessType.BOIL_WATER,
        description="片手鍋で湯を沸かす",
        tools_required=[
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
        ],
        duration_min=5.75,
        attentive_min=1.15,
    )
    return SubstitutionCandidate(
        original_edge_id="e_boil",
        replacement_edge=replacement,
        rationale="やかん→片手鍋",
        quality_delta=0.0,
        time_delta_min=0.75,
    )


# ---------------------------------------------------------------------------
# apply_substitution
# ---------------------------------------------------------------------------


def test_apply_substitution_replaces_target_edge():
    dag = _trivial_dag_with_yakan_edge()
    cand = _candidate_pot_for_e_boil()

    new_dag = apply_substitution(dag, cand)

    [new_edge] = new_dag.edges
    new_tool_names = {t.name for t in new_edge.tools_required}
    assert "やかん" not in new_tool_names
    assert "片手鍋" in new_tool_names
    assert new_edge.id == "e_boil"  # ID preserved


def test_apply_substitution_preserves_topology():
    dag = _trivial_dag_with_yakan_edge()
    cand = _candidate_pot_for_e_boil()

    new_dag = apply_substitution(dag, cand)
    [new_edge] = new_dag.edges

    assert new_edge.from_nodes == ["water"]
    assert new_edge.to_node == "boiling"
    assert new_dag.final_node_id == "boiling"
    assert {n.id for n in new_dag.nodes} == {n.id for n in dag.nodes}


def test_apply_substitution_does_not_mutate_input():
    dag = _trivial_dag_with_yakan_edge()
    original_tool_names = [t.name for t in dag.edges[0].tools_required]
    cand = _candidate_pot_for_e_boil()

    apply_substitution(dag, cand)

    # Input dag's edge should still have やかん
    assert [t.name for t in dag.edges[0].tools_required] == original_tool_names


def test_apply_substitution_raises_on_missing_edge():
    dag = _trivial_dag_with_yakan_edge()
    bogus = SubstitutionCandidate(
        original_edge_id="does_not_exist",
        replacement_edge=ProcessEdge(
            id="does_not_exist",
            from_nodes=["water"],
            to_node="boiling",
            action=ProcessType.BOIL_WATER,
            description="x",
        ),
        rationale="",
    )
    with pytest.raises(ValueError, match="not found in DAG"):
        apply_substitution(dag, bogus)


def test_apply_substitution_rejects_inconsistent_replacement():
    """If replacement_edge references a non-existent node, validation fires."""
    dag = _trivial_dag_with_yakan_edge()
    bad_replacement = ProcessEdge(
        id="e_boil",
        from_nodes=["water"],
        to_node="ghost_node",  # not in dag.nodes
        action=ProcessType.BOIL_WATER,
        description="x",
    )
    cand = SubstitutionCandidate(
        original_edge_id="e_boil",
        replacement_edge=bad_replacement,
        rationale="",
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        apply_substitution(dag, cand)


# ---------------------------------------------------------------------------
# apply_substitutions (batch)
# ---------------------------------------------------------------------------


def test_apply_substitutions_handles_multiple_edges():
    """Two substitutions on different edges both apply."""
    dag = make_mugicha_dag()
    e_boil = next(e for e in dag.edges if e.id == "e_boil")
    e_brew = next(e for e in dag.edges if e.id == "e_brew")

    pot_for_boil = e_boil.model_copy(
        update={
            "tools_required": [
                Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
            ],
            "description": "片手鍋で湯を沸かす",
        }
    )
    pot_for_brew = e_brew.model_copy(
        update={
            "tools_required": [Tool(name="片手鍋", kind=ToolKind.CONTAINER)],
            "description": "片手鍋で麦茶パックを抽出",
        }
    )

    candidates = [
        SubstitutionCandidate(
            original_edge_id="e_boil", replacement_edge=pot_for_boil, rationale=""
        ),
        SubstitutionCandidate(
            original_edge_id="e_brew", replacement_edge=pot_for_brew, rationale=""
        ),
    ]
    new_dag = apply_substitutions(dag, candidates)

    new_boil = next(e for e in new_dag.edges if e.id == "e_boil")
    new_brew = next(e for e in new_dag.edges if e.id == "e_brew")
    assert all(t.name != "やかん" for t in new_boil.tools_required)
    assert all(t.name != "やかん" for t in new_brew.tools_required)


def test_apply_substitutions_preserves_unchanged_edges():
    """Edges that are NOT targets of any substitution stay byte-identical."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description=""),
            GoalNode(id="b", description=""),
            GoalNode(id="c", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.MIX,
                description="keep me",
                tools_required=[Tool(name="ボウル", kind=ToolKind.CONTAINER)],
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["b"],
                to_node="c",
                action=ProcessType.MIX,
                description="replace me",
                tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
            ),
        ],
        final_node_id="c",
    )
    replacement_for_e2 = dag.edges[1].model_copy(
        update={
            "tools_required": [Tool(name="片手鍋", kind=ToolKind.CONTAINER)],
            "description": "片手鍋で混ぜる",
        }
    )
    cand = SubstitutionCandidate(
        original_edge_id="e2", replacement_edge=replacement_for_e2, rationale=""
    )

    new_dag = apply_substitutions(dag, [cand])

    e1_before = dag.edges[0]
    e1_after = next(e for e in new_dag.edges if e.id == "e1")
    assert e1_before == e1_after  # untouched edge preserved exactly


def test_apply_substitutions_empty_list_returns_equivalent_dag():
    dag = make_mugicha_dag()
    result = apply_substitutions(dag, [])
    assert result.title == dag.title
    assert len(result.edges) == len(dag.edges)
    assert {e.id for e in result.edges} == {e.id for e in dag.edges}


# ---------------------------------------------------------------------------
# End-to-end: checker → proposer → selector → rewriter
# ---------------------------------------------------------------------------


def test_full_pipeline_resolves_mugicha_violations():
    """After rewrite, the DAG should be satisfiable under the demo profile."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    proposer = RecipeProposer(client=MockLLMClient(), table=table)

    dag = make_mugicha_dag()
    constraints = Constraints(profile=profile)
    edges_by_id = {e.id: e for e in dag.edges}

    # 1. Find violations
    violations = find_violations(dag, constraints)
    assert len(violations) == 2

    # 2. For each violation, pick best candidate
    chosen: list[SubstitutionCandidate] = []
    for v in violations:
        cands = proposer.propose(v, edges_by_id[v.edge_id], profile)
        best = select_best(cands)
        if best is not None:
            chosen.append(best)

    # 3. Rewrite
    new_dag = apply_substitutions(dag, chosen)

    # 4. Verify: no more violations
    assert is_satisfiable(new_dag, constraints), (
        f"Remaining violations: {find_violations(new_dag, constraints)}"
    )

    # 5. Verify: やかん eliminated everywhere
    for edge in new_dag.edges:
        names = {t.name for t in edge.tools_required}
        assert "やかん" not in names
