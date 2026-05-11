"""Tests for the scheduler module under the resource model."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile
from recipe_optimizer.modules.scheduler import (
    SchedulingError,
    compute_critical_path,
    schedule,
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

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"


def _demo_constraints() -> Constraints:
    return Constraints(profile=load_profile(PROFILE_PATH))


def _profile_with(**pools) -> UserProfile:
    return UserProfile(user_id="t", **pools)


# ---------------------------------------------------------------------------
# Critical path
# ---------------------------------------------------------------------------


def test_critical_path_empty_dag():
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[GoalNode(id="x", description="d", is_final=True)],
        edges=[],
        final_node_id="x",
    )
    assert compute_critical_path(dag) == []


def test_critical_path_oyakodon_ends_at_serve():
    dag = make_oyakodon_dag()
    cp = compute_critical_path(dag)
    assert cp[-1] == "e_serve"
    assert any(eid.startswith("e_simmer") for eid in cp)


# ---------------------------------------------------------------------------
# Schedule basics
# ---------------------------------------------------------------------------


def test_schedule_oyakodon_makespan_positive():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    assert sched.total_duration_min > 0
    assert len(sched.steps) == len(dag.edges)


def test_schedule_oyakodon_respects_predecessors():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    by_id = {s.edge_id: s for s in sched.steps}

    assert by_id["e_simmer_onion"].start_min >= by_id["e_combine_step1"].end_min
    assert by_id["e_add_chicken"].start_min >= by_id["e_chop_chicken"].end_min
    assert by_id["e_add_chicken"].start_min >= by_id["e_simmer_onion"].end_min
    assert by_id["e_serve"].end_min == sched.total_duration_min


def test_schedule_assigns_resource_ids():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    for step in sched.steps:
        # Each step has at least one resource (cook) assigned
        assert len(step.assigned_resource_ids) >= 1


# ---------------------------------------------------------------------------
# Resource contention
# ---------------------------------------------------------------------------


def test_one_cook_serializes_three_attentive_chops():
    """Three independent chop edges must run sequentially with cook=1."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="ra", description="A"),
            GoalNode(id="rb", description="B"),
            GoalNode(id="rc", description="C"),
            GoalNode(id="ca", description="cut A"),
            GoalNode(id="cb", description="cut B"),
            GoalNode(id="cc", description="cut C"),
            GoalNode(id="done", description="done", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id=f"ec_{i}",
                from_nodes=[src],
                to_node=dst,
                action=ProcessType.CHOP,
                description=f"chop {i}",
                duration_min=2.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=2.0),
                    ResourceRequirement(
                        kind=ResourceKind.WORKSTATION, hold_duration_min=2.0
                    ),
                ],
            )
            for i, (src, dst) in enumerate(
                [("ra", "ca"), ("rb", "cb"), ("rc", "cc")]
            )
        ]
        + [
            ProcessEdge(
                id="ef",
                from_nodes=["ca", "cb", "cc"],
                to_node="done",
                action=ProcessType.MIX,
                description="combine",
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0),
                ],
            )
        ],
        final_node_id="done",
    )
    profile = _profile_with(
        cooks=[Resource(id="c1", kind=ResourceKind.COOK, name="自分")],
        workstations=[Resource(id="w1", kind=ResourceKind.WORKSTATION, name="台")],
    )
    sched = schedule(dag, Constraints(profile=profile))
    assert sched.total_duration_min == pytest.approx(7.0)  # 3*2 + 1


def test_three_burner_edges_with_two_burners_serializes_third():
    """3 parallel simmer edges with 2 burners — third must wait."""
    def _edge(eid: str, src: str, dst: str, container: str) -> ProcessEdge:
        return ProcessEdge(
            id=eid,
            from_nodes=[src],
            to_node=dst,
            action=ProcessType.SIMMER,
            description=eid,
            duration_min=4.0,
            resource_uses=[
                ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=0.5),
                ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=4.0),
                ResourceRequirement(
                    kind=ResourceKind.CONTAINER,
                    name_hint=container,
                    hold_duration_min=4.0,
                ),
            ],
        )

    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id=f"r{i}", description="") for i in range(1, 4)
        ] + [
            GoalNode(id=f"b{i}", description="") for i in range(1, 4)
        ] + [GoalNode(id="done", description="", is_final=True)],
        edges=[
            _edge("e1", "r1", "b1", "片手鍋"),
            _edge("e2", "r2", "b2", "両手鍋"),
            _edge("e3", "r3", "b3", "フライパン"),
            ProcessEdge(
                id="ef",
                from_nodes=["b1", "b2", "b3"],
                to_node="done",
                action=ProcessType.MIX,
                description="combine",
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0),
                ],
            ),
        ],
        final_node_id="done",
    )
    profile = _profile_with(
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
        burners=[
            Resource(id="b1", kind=ResourceKind.BURNER, name="コンロ口1"),
            Resource(id="b2", kind=ResourceKind.BURNER, name="コンロ口2"),
        ],
        containers=[
            Resource(id="p1", kind=ResourceKind.CONTAINER, name="片手鍋"),
            Resource(id="p2", kind=ResourceKind.CONTAINER, name="両手鍋"),
            Resource(id="p3", kind=ResourceKind.CONTAINER, name="フライパン"),
        ],
    )
    sched = schedule(dag, Constraints(profile=profile))
    by_id = {s.edge_id: s for s in sched.steps}

    events: list[tuple[float, int]] = []
    for eid in ("e1", "e2", "e3"):
        events.append((by_id[eid].start_min, +1))
        events.append((by_id[eid].end_min, -1))
    events.sort(key=lambda x: (x[0], x[1]))

    in_flight = 0
    max_concurrent = 0
    for _, delta in events:
        in_flight += delta
        max_concurrent = max(max_concurrent, in_flight)
    assert max_concurrent <= 2


def test_chop_runs_during_simmer_non_attentive_tail():
    """Cook is freed after attentive part of simmer, so chop runs in parallel."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="raw_simmer", description=""),
            GoalNode(id="raw_chop", description=""),
            GoalNode(id="simmered", description=""),
            GoalNode(id="chopped", description=""),
            GoalNode(id="done", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e_simmer",
                from_nodes=["raw_simmer"],
                to_node="simmered",
                action=ProcessType.SIMMER,
                description="simmer",
                duration_min=4.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=0.5),
                    ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=4.0),
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="片手鍋",
                        hold_duration_min=4.0,
                    ),
                ],
            ),
            ProcessEdge(
                id="e_chop",
                from_nodes=["raw_chop"],
                to_node="chopped",
                action=ProcessType.CHOP,
                description="chop",
                duration_min=2.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=2.0),
                    ResourceRequirement(
                        kind=ResourceKind.WORKSTATION, hold_duration_min=2.0
                    ),
                ],
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["simmered", "chopped"],
                to_node="done",
                action=ProcessType.MIX,
                description="mix",
                duration_min=0.5,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=0.5),
                ],
            ),
        ],
        final_node_id="done",
    )
    profile = _profile_with(
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
        burners=[Resource(id="b", kind=ResourceKind.BURNER, name="コンロ口")],
        workstations=[Resource(id="w", kind=ResourceKind.WORKSTATION, name="台")],
        containers=[Resource(id="p", kind=ResourceKind.CONTAINER, name="片手鍋")],
    )
    sched = schedule(dag, Constraints(profile=profile))
    by_id = {s.edge_id: s for s in sched.steps}

    # simmer: cook[0,0.5], burner[0,4], container[0,4]
    # chop starts at 0.5 (cook freed), ends at 2.5
    # ef starts at max(4, 2.5) = 4
    assert by_id["e_chop"].start_min == pytest.approx(0.5)
    assert sched.total_duration_min == pytest.approx(4.5)


def test_missing_resource_kind_raises_scheduling_error():
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description=""),
            GoalNode(id="b", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.SIMMER,
                description="x",
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=1.0),
                ],
            )
        ],
        final_node_id="b",
    )
    profile = _profile_with(
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
    )
    with pytest.raises(SchedulingError):
        schedule(dag, Constraints(profile=profile))
