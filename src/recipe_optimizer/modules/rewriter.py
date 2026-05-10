"""Rewriter: apply chosen substitutions to a RecipeDAG.

Pure-algorithmic module (no LLM). Given one or more selected
:class:`SubstitutionCandidate` objects, produces a new
:class:`RecipeDAG` with each target edge replaced.

Design contract:

- ``replacement_edge`` is constructed by the proposer to preserve the
  original edge's ``id``, ``from_nodes``, and ``to_node``. This means
  rewriting **never changes DAG topology** — only ``tools_required``,
  ``duration_min``, ``attentive_min``, and ``description`` change.
  Therefore subsequent substitutions still find their target edges by
  ID, and the final node / dependency chain stay intact.

- The output is a freshly constructed ``RecipeDAG``, which forces
  Pydantic's ``model_validator`` to re-run. Any inadvertent breakage
  (a malformed replacement_edge slipping through) surfaces immediately
  as a ``ValidationError`` rather than silently corrupting downstream
  scheduling.

- Inputs are not mutated. Functional style; pipeline orchestrators get
  a clean diff of (before, after) DAGs.
"""

from __future__ import annotations

from ..schemas import RecipeDAG, SubstitutionCandidate


def apply_substitution(
    dag: RecipeDAG,
    candidate: SubstitutionCandidate,
) -> RecipeDAG:
    """Replace one edge with the candidate's ``replacement_edge``.

    The target edge is identified by ``candidate.original_edge_id``.
    Returns a new ``RecipeDAG`` (the input is not mutated).

    Raises:
        ValueError: if no edge with the candidate's ID exists in ``dag``.
    """
    target_id = candidate.original_edge_id
    target_index: int | None = None
    for i, edge in enumerate(dag.edges):
        if edge.id == target_id:
            target_index = i
            break
    if target_index is None:
        raise ValueError(
            f"Cannot apply substitution: edge {target_id!r} not found in DAG"
        )

    new_edges = list(dag.edges)
    new_edges[target_index] = candidate.replacement_edge

    # Reconstruct (not model_copy) so Pydantic re-runs validation.
    return RecipeDAG(
        title=dag.title,
        servings=dag.servings,
        nodes=dag.nodes,
        edges=new_edges,
        final_node_id=dag.final_node_id,
        raw_text=dag.raw_text,
    )


def apply_substitutions(
    dag: RecipeDAG,
    candidates: list[SubstitutionCandidate],
) -> RecipeDAG:
    """Apply each candidate sequentially. Order is preserved.

    Because substitutions never alter edge IDs or topology, applying
    them in any order produces an equivalent result — but we honour the
    given order for determinism in logs.
    """
    for cand in candidates:
        dag = apply_substitution(dag, cand)
    return dag
