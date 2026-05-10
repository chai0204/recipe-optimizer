"""Tests for the scheduler module (pure-algorithmic resource-constrained scheduling)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile
from recipe_optimizer.modules.scheduler import (
    compute_critical_path,
    edge_requires_burner,
    edge_requires_cook,
    edge_requires_workstation,
    named_container_tools,
    schedule,
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

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"


def _demo_constraints() -> Constraints:
    return Constraints(profile=load_profile(PROFILE_PATH))


# ---------------------------------------------------------------------------
# Resource introspection helpers
# ---------------------------------------------------------------------------


def test_edge_requires_burner_true_when_heat_source_listed():
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.SIMMER,
        description="x",
        tools_required=[
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
        ],
    )
    assert edge_requires_burner(edge) is True


def test_edge_requires_burner_false_without_heat_source():
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.MIX,
        description="x",
        tools_required=[Tool(name="ボウル", kind=ToolKind.CONTAINER)],
    )
    assert edge_requires_burner(edge) is False


@pytest.mark.parametrize(
    "action,expected",
    [
        (ProcessType.CHOP, True),
        (ProcessType.SLICE, True),
        (ProcessType.MINCE, True),
        (ProcessType.PEEL, True),
        (ProcessType.KNEAD, True),
        (ProcessType.SIMMER, False),
        (ProcessType.MIX, False),
        (ProcessType.SERVE, False),
    ],
)
def test_edge_requires_workstation_by_action(action, expected):
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=action,
        description="x",
    )
    assert edge_requires_workstation(edge) is expected


def test_edge_requires_cook_iff_attentive_positive():
    free = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.REST,
        description="x",
        duration_min=10.0,
        attentive_min=0.0,
    )
    busy = ProcessEdge(
        id="e2",
        from_nodes=["a"],
        to_node="c",
        action=ProcessType.MIX,
        description="x",
        duration_min=1.0,
        attentive_min=1.0,
    )
    assert edge_requires_cook(free) is False
    assert edge_requires_cook(busy) is True


def test_named_container_tools_excludes_heat_source_and_utensil():
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.SIMMER,
        description="x",
        tools_required=[
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
            Tool(name="菜箸", kind=ToolKind.UTENSIL),
            Tool(name="電子レンジ", kind=ToolKind.APPLIANCE),
        ],
    )
    names = [t.name for t in named_container_tools(edge)]
    assert names == ["片手鍋", "電子レンジ"]


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
    # The simmer-onion → simmer-chicken chain should dominate the prep chain
    assert "e_simmer_onion" in cp or "e_simmer_chicken" in cp


# ---------------------------------------------------------------------------
# Schedule basics
# ---------------------------------------------------------------------------


def test_schedule_oyakodon_makespan_finite_and_positive():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    assert sched.total_duration_min > 0
    assert len(sched.steps) == len(dag.edges)


def test_schedule_oyakodon_respects_predecessor_completion():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    by_id = {s.edge_id: s for s in sched.steps}

    # e_simmer_onion needs n_pot_step1 → produced by e_combine_step1
    assert by_id["e_simmer_onion"].start_min >= by_id["e_combine_step1"].end_min
    # e_add_chicken needs both n_simmered_onion and n_chicken_cut
    assert by_id["e_add_chicken"].start_min >= by_id["e_chop_chicken"].end_min
    assert by_id["e_add_chicken"].start_min >= by_id["e_simmer_onion"].end_min
    # e_serve at the very end
    assert by_id["e_serve"].end_min == sched.total_duration_min


# ---------------------------------------------------------------------------
# Resource contention: cook + workstation
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
                id="ea",
                from_nodes=["ra"],
                to_node="ca",
                action=ProcessType.CHOP,
                description="chop A",
                duration_min=2.0,
                attentive_min=2.0,
            ),
            ProcessEdge(
                id="eb",
                from_nodes=["rb"],
                to_node="cb",
                action=ProcessType.CHOP,
                description="chop B",
                duration_min=2.0,
                attentive_min=2.0,
            ),
            ProcessEdge(
                id="ec",
                from_nodes=["rc"],
                to_node="cc",
                action=ProcessType.CHOP,
                description="chop C",
                duration_min=2.0,
                attentive_min=2.0,
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["ca", "cb", "cc"],
                to_node="done",
                action=ProcessType.MIX,
                description="combine",
                duration_min=1.0,
                attentive_min=1.0,
            ),
        ],
        final_node_id="done",
    )
    profile = UserProfile(
        user_id="t",
        tools_owned=[
            Tool(name="包丁", kind=ToolKind.UTENSIL),
            Tool(name="まな板", kind=ToolKind.UTENSIL),
        ],
        num_cooks=1,
        workstation_count=1,
    )
    sched = schedule(dag, Constraints(profile=profile))
    # 3 × 2-min chops + 1-min mix = 7 min if perfectly serial
    assert sched.total_duration_min == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Resource contention: burners
# ---------------------------------------------------------------------------


def test_three_burner_edges_with_two_burners_serializes_third():
    """Three parallel-able simmer edges with burners=2: the third must wait."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="r1", description=""),
            GoalNode(id="r2", description=""),
            GoalNode(id="r3", description=""),
            GoalNode(id="b1", description=""),
            GoalNode(id="b2", description=""),
            GoalNode(id="b3", description=""),
            GoalNode(id="done", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["r1"],
                to_node="b1",
                action=ProcessType.SIMMER,
                description="simmer 1",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=4.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["r2"],
                to_node="b2",
                action=ProcessType.SIMMER,
                description="simmer 2",
                tools_required=[
                    Tool(name="両手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=4.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="e3",
                from_nodes=["r3"],
                to_node="b3",
                action=ProcessType.SIMMER,
                description="simmer 3",
                tools_required=[
                    Tool(name="フライパン", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=4.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["b1", "b2", "b3"],
                to_node="done",
                action=ProcessType.MIX,
                description="combine",
                duration_min=1.0,
                attentive_min=1.0,
            ),
        ],
        final_node_id="done",
    )
    profile = UserProfile(
        user_id="t",
        tools_owned=[
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
            Tool(name="両手鍋", kind=ToolKind.CONTAINER),
            Tool(name="フライパン", kind=ToolKind.CONTAINER),
        ],
        burners_count=2,
        num_cooks=1,
    )
    sched = schedule(dag, Constraints(profile=profile))
    by_id = {s.edge_id: s for s in sched.steps}

    # Walk the timeline and verify ≤ 2 burner edges in flight at any point
    events: list[tuple[float, int]] = []
    for eid in ("e1", "e2", "e3"):
        events.append((by_id[eid].start_min, +1))
        events.append((by_id[eid].end_min, -1))
    events.sort(key=lambda x: (x[0], x[1]))  # ends before starts at same instant

    in_flight = 0
    max_concurrent = 0
    for _, delta in events:
        in_flight += delta
        max_concurrent = max(max_concurrent, in_flight)
    assert max_concurrent <= 2

    # Concrete makespan: e1[0,4], e2[0.5,4.5], e3[4,8], ef[8,9] → total 9
    assert sched.total_duration_min == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Resource contention: named containers
# ---------------------------------------------------------------------------


def test_named_container_serializes_two_edges_using_same_pot():
    """Two parallel-able edges that both list 片手鍋 cannot overlap."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="r1", description=""),
            GoalNode(id="r2", description=""),
            GoalNode(id="b1", description=""),
            GoalNode(id="b2", description=""),
            GoalNode(id="done", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["r1"],
                to_node="b1",
                action=ProcessType.SIMMER,
                description="x",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=3.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["r2"],
                to_node="b2",
                action=ProcessType.SIMMER,
                description="y",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=3.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["b1", "b2"],
                to_node="done",
                action=ProcessType.MIX,
                description="z",
                duration_min=1.0,
                attentive_min=1.0,
            ),
        ],
        final_node_id="done",
    )
    profile = UserProfile(
        user_id="t",
        tools_owned=[Tool(name="片手鍋", kind=ToolKind.CONTAINER)],
        burners_count=2,
        num_cooks=1,
    )
    sched = schedule(dag, Constraints(profile=profile))
    by_id = {s.edge_id: s for s in sched.steps}

    e1, e2 = by_id["e1"], by_id["e2"]
    no_overlap = (e1.end_min <= e2.start_min) or (e2.end_min <= e1.start_min)
    assert no_overlap, "Edges sharing 片手鍋 must not overlap"


# ---------------------------------------------------------------------------
# Parallelization: cook freed during simmer's non-attentive tail
# ---------------------------------------------------------------------------


def test_chop_runs_during_simmer_non_attentive_tail():
    """Cook is freed after attentive_min, so a chop can run while a long
    simmer is still cooking. Total time should be dominated by the simmer,
    not be the sum of simmer + chop."""
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
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=4.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="e_chop",
                from_nodes=["raw_chop"],
                to_node="chopped",
                action=ProcessType.CHOP,
                description="chop",
                duration_min=2.0,
                attentive_min=2.0,
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["simmered", "chopped"],
                to_node="done",
                action=ProcessType.MIX,
                description="mix",
                duration_min=0.5,
                attentive_min=0.5,
            ),
        ],
        final_node_id="done",
    )
    profile = UserProfile(
        user_id="t",
        tools_owned=[
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
            Tool(name="包丁", kind=ToolKind.UTENSIL),
        ],
        num_cooks=1,
        burners_count=2,
        workstation_count=1,
    )
    sched = schedule(dag, Constraints(profile=profile))
    by_id = {s.edge_id: s for s in sched.steps}

    # e_simmer ends at 4.0; e_chop runs [0.5, 2.5] (cook freed at 0.5).
    # ef cannot start until both end → max(4.0, 2.5) = 4.0; ef[4.0, 4.5].
    assert by_id["e_chop"].start_min == pytest.approx(0.5)
    assert by_id["e_chop"].end_min == pytest.approx(2.5)
    assert sched.total_duration_min == pytest.approx(4.5)

    # Verify they're flagged as parallel
    chop_step = by_id["e_chop"]
    assert "e_simmer" in chop_step.parallel_with


# ---------------------------------------------------------------------------
# Smoke: oyakodon end-to-end shape checks
# ---------------------------------------------------------------------------


def test_schedule_oyakodon_no_attentive_overlap_with_one_cook():
    """At any time at most ``num_cooks`` attentive windows are active."""
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    by_id = {s.edge_id: s for s in sched.steps}
    edges_by_id = {e.id: e for e in dag.edges}

    # Build (start, +1)/(start+attentive, -1) events
    events: list[tuple[float, int]] = []
    for eid, step in by_id.items():
        edge = edges_by_id[eid]
        if edge.attentive_min <= 0:
            continue
        events.append((step.start_min, +1))
        events.append((step.start_min + edge.attentive_min, -1))
    events.sort(key=lambda x: (x[0], x[1]))

    in_flight = 0
    for _, delta in events:
        in_flight += delta
        assert in_flight <= 1  # num_cooks=1


def test_schedule_oyakodon_no_burner_over_allocation():
    dag = make_oyakodon_dag()
    sched = schedule(dag, _demo_constraints())
    by_id = {s.edge_id: s for s in sched.steps}
    edges_by_id = {e.id: e for e in dag.edges}

    events: list[tuple[float, int]] = []
    for eid, step in by_id.items():
        if not edge_requires_burner(edges_by_id[eid]):
            continue
        events.append((step.start_min, +1))
        events.append((step.end_min, -1))
    events.sort(key=lambda x: (x[0], x[1]))

    in_flight = 0
    for _, delta in events:
        in_flight += delta
        assert in_flight <= 2  # burners_count=2
