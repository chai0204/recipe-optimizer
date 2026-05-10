"""Resource-constrained scheduler for ``RecipeDAG``.

Pure-algorithmic module (no LLM). Produces a :class:`Schedule` that
respects four resource pools simultaneously:

- **Cook** (``num_cooks``): consumed for ``attentive_min`` only — frees
  up while the dish continues simmering / baking on its own.
- **Burner** (``burners_count``): consumed for ``duration_min`` whenever
  an edge lists a ``kind=heat_source`` tool.
- **Workstation** (``workstation_count``): consumed for ``attentive_min``
  for chop / slice / mince / peel / knead actions.
- **Named containers** (one slot per name): every tool with kind in
  ``{CONTAINER, APPLIANCE}`` is held for ``duration_min``.

Algorithm: classical *list scheduling* — iterate edges in topological
order, compute the earliest time at which all required resources and
all predecessor edges have completed, allocate, repeat. Greedy. Not
guaranteed optimal but converges within a few percent of optimum on
recipe-sized DAGs and runs in O(E log E).

The critical path (longest dependency chain weighted by ``duration_min``)
is computed independently of resources — it is the lower bound on
makespan if every resource were unlimited, and is what we surface to
the user as "the bottleneck steps".
"""

from __future__ import annotations

import networkx as nx

from ..schemas import (
    Constraints,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    Schedule,
    ScheduledStep,
    Tool,
    ToolKind,
)

# ---------------------------------------------------------------------------
# Resource introspection
# ---------------------------------------------------------------------------


WORKSTATION_ACTIONS: frozenset[ProcessType] = frozenset(
    {
        ProcessType.CHOP,
        ProcessType.SLICE,
        ProcessType.MINCE,
        ProcessType.PEEL,
        ProcessType.KNEAD,
    }
)


def edge_requires_burner(edge: ProcessEdge) -> bool:
    """An edge consumes a burner iff it lists a heat-source tool.

    The parser/LLM is responsible for adding the heat source to
    ``tools_required`` whenever heat is involved. This keeps the
    scheduler purely declarative and avoids inferring resources from
    action names alone.
    """
    return any(t.kind == ToolKind.HEAT_SOURCE for t in edge.tools_required)


def edge_requires_workstation(edge: ProcessEdge) -> bool:
    """Knife / prep counter actions require the workstation."""
    return edge.action in WORKSTATION_ACTIONS


def edge_requires_cook(edge: ProcessEdge) -> bool:
    """Any edge with non-zero attention demand consumes the cook briefly."""
    return edge.attentive_min > 0.0


def named_container_tools(edge: ProcessEdge) -> list[Tool]:
    """Return tools that must be reserved exclusively for ``duration_min``.

    Heat sources are excluded — they are tracked via the burner pool.
    Utensils (knives, chopsticks) are excluded because their use is too
    short to meaningfully contend in PoC-scale recipes.
    """
    return [
        t
        for t in edge.tools_required
        if t.kind in {ToolKind.CONTAINER, ToolKind.APPLIANCE}
    ]


# ---------------------------------------------------------------------------
# Edge dependency graph
# ---------------------------------------------------------------------------


def _build_edge_dag(dag: RecipeDAG) -> nx.DiGraph:
    """Edge-level DAG: ``A -> B`` iff ``A.to_node ∈ B.from_nodes``.

    Nodes carry ``duration`` for downstream critical-path calculations.
    """
    edges_producing: dict[str, str] = {e.to_node: e.id for e in dag.edges}
    g: nx.DiGraph = nx.DiGraph()
    for e in dag.edges:
        g.add_node(e.id, duration=e.duration_min)
    for e in dag.edges:
        for from_node in e.from_nodes:
            pred = edges_producing.get(from_node)
            if pred is not None:
                g.add_edge(pred, e.id)
    return g


# ---------------------------------------------------------------------------
# Critical path (resource-independent)
# ---------------------------------------------------------------------------


def compute_critical_path(dag: RecipeDAG) -> list[str]:
    """Edges on the longest dependency chain weighted by ``duration_min``.

    Returns the chain as ordered edge IDs, source-first. Independent of
    resource constraints: this is the inherent recipe bottleneck.
    """
    g = _build_edge_dag(dag)
    if g.number_of_nodes() == 0:
        return []

    edges_by_id = {e.id: e for e in dag.edges}
    longest: dict[str, float] = {}
    parent: dict[str, str | None] = {}

    for eid in nx.topological_sort(g):
        edge = edges_by_id[eid]
        preds = list(g.predecessors(eid))
        if preds:
            best = max(preds, key=lambda p: longest[p])
            longest[eid] = longest[best] + edge.duration_min
            parent[eid] = best
        else:
            longest[eid] = edge.duration_min
            parent[eid] = None

    last = max(longest, key=lambda eid: longest[eid])
    path: list[str] = []
    cur: str | None = last
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def schedule(dag: RecipeDAG, constraints: Constraints) -> Schedule:
    """Greedy list scheduling against the four resource pools.

    Returns a Schedule whose ``steps`` are ordered topologically. Each
    ``ScheduledStep.parallel_with`` lists the IDs of edges that overlap
    in wall-clock time with this one.
    """
    profile = constraints.profile
    g = _build_edge_dag(dag)
    edges_by_id = {e.id: e for e in dag.edges}

    # Resource pools — one float per slot, holding "busy until" time
    cook_pool: list[float] = [0.0] * max(profile.num_cooks, 1)
    burner_pool: list[float] = [0.0] * max(profile.burners_count, 0)
    workstation_pool: list[float] = [0.0] * max(profile.workstation_count, 0)
    container_busy: dict[str, float] = {}

    edge_start: dict[str, float] = {}
    edge_end: dict[str, float] = {}
    topo_order = list(nx.topological_sort(g))

    for eid in topo_order:
        edge = edges_by_id[eid]

        # 1. Predecessor edges must have completed
        prereq_end = max(
            (edge_end[p] for p in g.predecessors(eid)),
            default=0.0,
        )

        # 2. Each required resource pool must have a free slot
        candidate_starts: list[float] = [prereq_end]

        if edge_requires_cook(edge) and cook_pool:
            candidate_starts.append(min(cook_pool))
        if edge_requires_burner(edge):
            if not burner_pool:
                raise ValueError(
                    f"Edge {edge.id!r} requires a burner but the profile has none "
                    f"(burners_count=0)"
                )
            candidate_starts.append(min(burner_pool))
        if edge_requires_workstation(edge) and workstation_pool:
            candidate_starts.append(min(workstation_pool))
        for tool in named_container_tools(edge):
            candidate_starts.append(container_busy.get(tool.name, 0.0))

        start = max(candidate_starts)
        end = start + edge.duration_min

        # 3. Allocate
        if edge_requires_cook(edge) and cook_pool:
            slot = cook_pool.index(min(cook_pool))
            cook_pool[slot] = start + edge.attentive_min
        if edge_requires_burner(edge):
            slot = burner_pool.index(min(burner_pool))
            burner_pool[slot] = end
        if edge_requires_workstation(edge) and workstation_pool:
            slot = workstation_pool.index(min(workstation_pool))
            workstation_pool[slot] = start + edge.attentive_min
        for tool in named_container_tools(edge):
            container_busy[tool.name] = end

        edge_start[eid] = start
        edge_end[eid] = end

    # 4. Build ScheduledStep records, computing parallel_with
    intervals = [(eid, edge_start[eid], edge_end[eid]) for eid in topo_order]

    def overlapping_with(eid: str, s: float, e: float) -> list[str]:
        return [
            other
            for other, os_, oe_ in intervals
            if other != eid and not (e <= os_ or oe_ <= s)
        ]

    steps = [
        ScheduledStep(
            edge_id=eid,
            start_min=edge_start[eid],
            end_min=edge_end[eid],
            parallel_with=overlapping_with(eid, edge_start[eid], edge_end[eid]),
        )
        for eid in topo_order
    ]

    total = max(edge_end.values(), default=0.0)
    cp = compute_critical_path(dag)

    return Schedule(
        steps=steps,
        total_duration_min=total,
        critical_path_edge_ids=cp,
    )
