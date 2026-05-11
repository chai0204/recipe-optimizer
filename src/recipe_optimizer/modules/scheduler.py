"""Resource-constrained scheduler against explicit ``resource_uses``.

Pure-algorithmic module. The substitution layer (proposer + rewriter)
ensures every edge in the input DAG carries a ``resource_uses`` list
whose entries reference resource kinds (and optional name hints) the
user actually owns. The scheduler then assigns each requirement to a
concrete ``Resource`` instance and walks the timeline.

Algorithm:

1. Build the edge dependency graph (A → B iff A.to_node ∈ B.from_nodes).
2. Topologically iterate edges.
3. For each edge, compute the earliest start such that
   - all predecessor edges have finished, AND
   - every ``ResourceRequirement`` can be matched to a ``Resource`` of
     the right kind+name_hint whose busy-until time is ≤ the candidate
     start (taking start_offset_min into account).
4. Allocate each chosen resource for ``[start+offset, start+offset+hold]``.

Critical path is computed independently of resources (longest chain
weighted by ``duration_min``).
"""

from __future__ import annotations

import networkx as nx

from ..data_io.tool_use_table import _matches_name_hint
from ..schemas import (
    Constraints,
    ProcessEdge,
    RecipeDAG,
    Resource,
    ResourceRequirement,
    Schedule,
    ScheduledStep,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Resource pool with busy-until per slot
# ---------------------------------------------------------------------------


class _ResourcePool:
    """In-memory busy-until tracker keyed by resource id."""

    def __init__(self, profile: UserProfile) -> None:
        self._by_id: dict[str, Resource] = {}
        for kind in (
            "cook",
            "burner",
            "workstation",
            "container",
            "appliance",
            "utensil",
        ):
            for r in getattr(profile, f"{kind}s" if kind != "cook" else "cooks"):
                self._by_id[r.id] = r
        self._busy_until: dict[str, float] = {rid: 0.0 for rid in self._by_id}

    def candidates_for(self, req: ResourceRequirement) -> list[Resource]:
        """All resources matching ``req.kind`` and optional ``name_hint``."""
        out: list[Resource] = []
        for r in self._by_id.values():
            if r.kind != req.kind:
                continue
            if req.name_hint is not None and not _matches_name_hint(
                r.name, req.name_hint
            ):
                continue
            out.append(r)
        return out

    def earliest_slot(
        self, req: ResourceRequirement, candidates: list[Resource]
    ) -> tuple[Resource, float] | None:
        """Pick the resource whose busy_until is smallest. Returns
        ``(resource, earliest_start)`` or ``None`` if no candidate exists."""
        if not candidates:
            return None
        best = min(candidates, key=lambda r: self._busy_until[r.id])
        return best, self._busy_until[best.id]

    def reserve(self, resource: Resource, start: float, end: float) -> None:
        # busy_until tracks the latest end across reservations
        self._busy_until[resource.id] = max(self._busy_until[resource.id], end)


# ---------------------------------------------------------------------------
# Edge dependency graph
# ---------------------------------------------------------------------------


def _build_edge_dag(dag: RecipeDAG) -> nx.DiGraph:
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


class SchedulingError(RuntimeError):
    """Raised when an edge demands a resource kind/name the profile lacks."""


def schedule(dag: RecipeDAG, constraints: Constraints) -> Schedule:
    """Allocate each edge's resource_uses against the profile's pools."""
    profile = constraints.profile
    g = _build_edge_dag(dag)
    edges_by_id = {e.id: e for e in dag.edges}
    pool = _ResourcePool(profile)

    edge_start: dict[str, float] = {}
    edge_end: dict[str, float] = {}
    edge_assignments: dict[str, list[str]] = {}
    topo_order = list(nx.topological_sort(g))

    for eid in topo_order:
        edge = edges_by_id[eid]
        prereq_end = max(
            (edge_end[p] for p in g.predecessors(eid)),
            default=0.0,
        )

        # For each requirement, find the earliest slot. Constrain the
        # edge's start so that *every* requirement can begin at its
        # ``start_offset_min`` after the edge's start.
        candidates_per_req: list[tuple[ResourceRequirement, Resource, float]] = []
        for req in edge.resource_uses:
            cands = pool.candidates_for(req)
            if not cands:
                raise SchedulingError(
                    f"Edge {edge.id!r}: no resource of kind={req.kind.value} "
                    f"name_hint={req.name_hint!r} available in profile"
                )
            chosen, free_at = pool.earliest_slot(req, cands)
            # The edge start must be such that free_at ≤ start + start_offset
            min_start_for_req = free_at - req.start_offset_min
            candidates_per_req.append((req, chosen, min_start_for_req))

        start = max([prereq_end] + [s for _, _, s in candidates_per_req])
        end = start + edge.duration_min

        # Reserve
        assignments: list[str] = []
        for req, chosen, _ in candidates_per_req:
            r_start = start + req.start_offset_min
            r_end = r_start + req.hold_duration_min
            pool.reserve(chosen, r_start, r_end)
            assignments.append(chosen.id)

        edge_start[eid] = start
        edge_end[eid] = end
        edge_assignments[eid] = assignments

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
            assigned_resource_ids=edge_assignments[eid],
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


# ---------------------------------------------------------------------------
# Backwards-compat shim for legacy module imports (now no-op)
# ---------------------------------------------------------------------------


def edge_requires_burner(edge: ProcessEdge) -> bool:
    from ..schemas import ResourceKind  # noqa: PLC0415

    return any(u.kind == ResourceKind.BURNER for u in edge.resource_uses)
