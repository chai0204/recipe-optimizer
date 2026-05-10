"""Tests for the checker module (pure-algorithmic, no LLM)."""

from __future__ import annotations

from pathlib import Path

from recipe_optimizer.data_io import load_profile
from recipe_optimizer.modules.checker import (
    REASON_MISSING_TOOL,
    find_violations,
    is_satisfiable,
)
from recipe_optimizer.schemas import (
    Constraints,
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    Tool,
    ToolKind,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag
from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"


def _demo_constraints() -> Constraints:
    return Constraints(profile=load_profile(PROFILE_PATH))


def _empty_constraints() -> Constraints:
    return Constraints(profile=UserProfile(user_id="empty", tools_owned=[]))


# ---------------------------------------------------------------------------
# Baseline / no-op cases
# ---------------------------------------------------------------------------


def test_empty_dag_has_no_violations():
    dag = RecipeDAG(
        title="empty",
        servings=1,
        nodes=[GoalNode(id="x", description="d", is_final=True)],
        edges=[],
        final_node_id="x",
    )
    assert find_violations(dag, _demo_constraints()) == []
    assert is_satisfiable(dag, _demo_constraints())


def test_serve_edge_with_no_tools_never_violates():
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="x"),
            GoalNode(id="b", description="y", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.SERVE,
                description="器に盛る",
                tools_required=[],
            )
        ],
        final_node_id="b",
    )
    assert find_violations(dag, _empty_constraints()) == []


# ---------------------------------------------------------------------------
# Real-data integration: oyakodon (clean) vs mugicha (yakan missing)
# ---------------------------------------------------------------------------


def test_oyakodon_under_demo_profile_is_clean():
    """Demo profile has all tools the oyakodon recipe requires
    (片手鍋, 包丁, まな板, 菜箸, ボウル, コンロ口)."""
    dag = make_oyakodon_dag()
    violations = find_violations(dag, _demo_constraints())
    assert violations == [], f"Unexpected violations: {violations}"
    assert is_satisfiable(dag, _demo_constraints())


def test_mugicha_under_demo_profile_flags_yakan_on_both_edges():
    """The demo profile has no やかん, so both mugicha edges should violate."""
    dag = make_mugicha_dag()
    violations = find_violations(dag, _demo_constraints())

    edge_ids = [v.edge_id for v in violations]
    assert "e_boil" in edge_ids
    assert "e_brew" in edge_ids

    boil_v = next(v for v in violations if v.edge_id == "e_boil")
    assert boil_v.reason == REASON_MISSING_TOOL
    assert "やかん" in boil_v.missing
    # コンロ口 is also required, but the profile has コンロ口1/2 which
    # match via substring → should NOT be in missing
    assert not any("コンロ口" in m for m in boil_v.missing)

    brew_v = next(v for v in violations if v.edge_id == "e_brew")
    assert brew_v.missing == ["やかん"]


def test_is_satisfiable_returns_false_when_violations_exist():
    dag = make_mugicha_dag()
    assert not is_satisfiable(dag, _demo_constraints())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_lists_all_missing_tools_per_edge():
    """An edge requiring multiple missing tools reports them all."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="x"),
            GoalNode(id="b", description="y", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.UNKNOWN,
                description="needs many",
                tools_required=[
                    Tool(name="圧力鍋", kind=ToolKind.CONTAINER),
                    Tool(name="ミキサー", kind=ToolKind.APPLIANCE),
                ],
            )
        ],
        final_node_id="b",
    )
    violations = find_violations(dag, _demo_constraints())
    assert len(violations) == 1
    assert sorted(violations[0].missing) == ["ミキサー", "圧力鍋"]


def test_composite_partial_owned_is_violation():
    """User has 包丁 but not まな板 → composite 包丁+まな板 fails."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="x"),
            GoalNode(id="b", description="y", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.CHOP,
                description="chop",
                tools_required=[Tool(name="包丁+まな板", kind=ToolKind.UTENSIL)],
            )
        ],
        final_node_id="b",
    )
    profile = UserProfile(
        user_id="partial",
        tools_owned=[Tool(name="包丁", kind=ToolKind.UTENSIL)],
    )
    constraints = Constraints(profile=profile)
    violations = find_violations(dag, constraints)
    assert len(violations) == 1
    assert violations[0].missing == ["包丁+まな板"]


def test_violations_returned_in_edge_order():
    e1 = ProcessEdge(
        id="e1",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.UNKNOWN,
        description="x",
        tools_required=[Tool(name="不所持A", kind=ToolKind.UTENSIL)],
    )
    e2 = ProcessEdge(
        id="e2",
        from_nodes=["b"],
        to_node="c",
        action=ProcessType.UNKNOWN,
        description="y",
        tools_required=[Tool(name="不所持B", kind=ToolKind.UTENSIL)],
    )
    e3 = ProcessEdge(
        id="e3",
        from_nodes=["c"],
        to_node="d",
        action=ProcessType.SERVE,
        description="z",
        tools_required=[],  # no violation
    )
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="r"),
            GoalNode(id="b", description="s"),
            GoalNode(id="c", description="t"),
            GoalNode(id="d", description="u", is_final=True),
        ],
        edges=[e1, e2, e3],
        final_node_id="d",
    )
    violations = find_violations(dag, _empty_constraints())
    assert [v.edge_id for v in violations] == ["e1", "e2"]
