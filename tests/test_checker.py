"""Tests for the checker module under the resource model."""

from __future__ import annotations

from pathlib import Path

from recipe_optimizer.data_io import load_profile
from recipe_optimizer.modules.checker import (
    REASON_MISSING_RESOURCE,
    find_violations,
    is_satisfiable,
)
from recipe_optimizer.schemas import (
    Constraints,
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    Resource,
    ResourceKind,
    ResourceRequirement,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag
from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"


def _demo_constraints() -> Constraints:
    return Constraints(profile=load_profile(PROFILE_PATH))


def _empty_constraints() -> Constraints:
    return Constraints(profile=UserProfile(user_id="empty"))


# ---------------------------------------------------------------------------
# Baseline / no-op
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


def test_edge_with_no_resource_uses_never_violates():
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
                duration_min=0.0,
                resource_uses=[],
            )
        ],
        final_node_id="b",
    )
    assert find_violations(dag, _empty_constraints()) == []


# ---------------------------------------------------------------------------
# Real-data fixtures
# ---------------------------------------------------------------------------


def test_oyakodon_under_demo_profile_is_clean():
    """Demo profile carries every resource the oyakodon DAG declares."""
    dag = make_oyakodon_dag()
    violations = find_violations(dag, _demo_constraints())
    assert violations == [], f"Unexpected violations: {violations}"


def test_mugicha_under_demo_profile_flags_yakan_on_both_edges():
    dag = make_mugicha_dag()
    violations = find_violations(dag, _demo_constraints())

    edge_ids = [v.edge_id for v in violations]
    assert "e_boil" in edge_ids
    assert "e_brew" in edge_ids
    for v in violations:
        assert v.reason == REASON_MISSING_RESOURCE
        assert v.missing_kind == ResourceKind.CONTAINER
        assert v.missing_name_hint == "やかん"


def test_is_satisfiable_reflects_violation_presence():
    constraints = _demo_constraints()
    assert is_satisfiable(make_oyakodon_dag(), constraints)
    assert not is_satisfiable(make_mugicha_dag(), constraints)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_multiple_missing_resources_on_one_edge_yield_one_violation_each():
    """Each unmet ResourceRequirement becomes its own violation."""
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
                description="exotic",
                duration_min=10.0,
                resource_uses=[
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="圧力鍋",
                        hold_duration_min=10.0,
                    ),
                    ResourceRequirement(
                        kind=ResourceKind.APPLIANCE,
                        name_hint="ミキサー",
                        hold_duration_min=10.0,
                    ),
                ],
            )
        ],
        final_node_id="b",
    )
    violations = find_violations(dag, _demo_constraints())
    assert len(violations) == 2
    missing = sorted(v.missing_name_hint for v in violations if v.missing_name_hint)
    assert missing == ["ミキサー", "圧力鍋"]


def test_violations_returned_in_edge_then_resource_order():
    edges = [
        ProcessEdge(
            id="e1",
            from_nodes=["a"],
            to_node="b",
            action=ProcessType.UNKNOWN,
            description="",
            duration_min=1.0,
            resource_uses=[
                ResourceRequirement(
                    kind=ResourceKind.CONTAINER,
                    name_hint="圧力鍋",
                    hold_duration_min=1.0,
                )
            ],
        ),
        ProcessEdge(
            id="e2",
            from_nodes=["b"],
            to_node="c",
            action=ProcessType.UNKNOWN,
            description="",
            duration_min=1.0,
            resource_uses=[
                ResourceRequirement(
                    kind=ResourceKind.APPLIANCE,
                    name_hint="ミキサー",
                    hold_duration_min=1.0,
                )
            ],
        ),
    ]
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="r"),
            GoalNode(id="b", description="s"),
            GoalNode(id="c", description="t", is_final=True),
        ],
        edges=edges,
        final_node_id="c",
    )
    violations = find_violations(dag, _demo_constraints())
    assert [v.edge_id for v in violations] == ["e1", "e2"]
