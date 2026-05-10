"""Structured DAG visualization (output ②) in Mermaid syntax.

Mermaid is a text-based diagram language that GitHub, Notion, VSCode,
and many markdown viewers render natively. This means the same string
can be used as:

- a debug print on the terminal (still readable as text)
- an embedded diagram in README / works/ files
- live-rendered preview in editors

The output is a ``flowchart TD`` (top-down) with:

- one labeled box per ``GoalNode`` (final node highlighted)
- one directed arrow per ``(from_node, edge)`` pair, labeled with the
  action and duration

Critical-path edges receive a thicker stroke when a Schedule with
``critical_path_edge_ids`` is provided.
"""

from __future__ import annotations

from ..schemas import RecipeDAG, Schedule

_FINAL_CLASS = "finalNode"
_CRITICAL_CLASS = "criticalEdge"


def _safe_label(text: str) -> str:
    """Mermaid edge/node labels live inside double-quoted strings.

    Replace inner double quotes with single quotes and collapse
    newlines to spaces so the diagram parses cleanly.
    """
    return text.replace('"', "'").replace("\n", " ").strip()


def _format_minutes(minutes: float) -> str:
    if minutes < 1.0:
        return f"{int(round(minutes * 60))}秒"
    if minutes == int(minutes):
        return f"{int(minutes)}分"
    return f"{minutes:.1f}分"


def to_mermaid(dag: RecipeDAG, schedule: Schedule | None = None) -> str:
    """Render the DAG as a Mermaid flowchart.

    Args:
        dag: the (optimized) recipe DAG.
        schedule: optional; when provided, edges on the critical path
            are emphasized via the ``criticalEdge`` linkStyle.
    """
    critical_edges: set[str] = (
        set(schedule.critical_path_edge_ids) if schedule else set()
    )

    lines: list[str] = ["flowchart TD"]

    for node in dag.nodes:
        label = _safe_label(node.description) or node.id
        if node.is_final:
            lines.append(f'  {node.id}["{label}"]:::{_FINAL_CLASS}')
        else:
            lines.append(f'  {node.id}["{label}"]')

    edge_index = 0
    critical_indices: list[int] = []
    for edge in dag.edges:
        action = edge.action.value
        duration = _format_minutes(edge.duration_min)
        edge_label = _safe_label(f"{action} ({duration})")
        for from_node in edge.from_nodes:
            lines.append(f'  {from_node} -->|"{edge_label}"| {edge.to_node}')
            if edge.id in critical_edges:
                critical_indices.append(edge_index)
            edge_index += 1

    # Styles
    lines.append("")
    lines.append(f"  classDef {_FINAL_CLASS} fill:#aef,stroke:#06a,stroke-width:2px")
    if critical_indices:
        idx_csv = ",".join(str(i) for i in critical_indices)
        lines.append(f"  linkStyle {idx_csv} stroke:#d40,stroke-width:3px")

    return "\n".join(lines)
