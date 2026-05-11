"""Numbered-list recipe view (output ①).

Renders ``Schedule.steps`` in chronological start-time order. This is
the user-friendly "do A, then B" sequence that mirrors a traditional
printed recipe.

Edges that the scheduler placed in parallel are still listed
sequentially here, but flagged with a ``(並行)`` marker so the user
knows they can be done concurrently. A future LLM ``renderer`` can
polish this into more natural Japanese prose; the structure provided
here is the deterministic fallback.
"""

from __future__ import annotations

from ..schemas import RecipeDAG, Schedule


def _format_minutes(minutes: float) -> str:
    if minutes < 1.0:
        seconds = round(minutes * 60)
        return f"{seconds}秒"
    if minutes == int(minutes):
        return f"{int(minutes)}分"
    return f"{minutes:.1f}分"


def to_numbered_steps(dag: RecipeDAG, schedule: Schedule) -> list[str]:
    """Return one string per step, in chronological order.

    Format: ``"<n>. <description> (<duration>)"`` with optional
    ``(並行可能)`` suffix when ``parallel_with`` is non-empty.
    """
    edges_by_id = {e.id: e for e in dag.edges}
    chronological = sorted(
        schedule.steps, key=lambda s: (s.start_min, s.edge_id)
    )

    lines: list[str] = []
    for i, step in enumerate(chronological, start=1):
        edge = edges_by_id[step.edge_id]
        duration_str = _format_minutes(edge.duration_min)
        # Edges with non-empty parallel_with that are not pure serve actions
        # benefit from a parallel marker.
        parallel_marker = " (並行可能)" if step.parallel_with else ""
        lines.append(f"{i}. {edge.description} ({duration_str}){parallel_marker}")
    return lines
