"""Constraint checker: identify edges that cannot be executed as-is.

Pure-algorithmic module (no LLM). Given a RecipeDAG and the user's
constraints (profile + session state), returns a list of
ConstraintViolation objects, one per edge whose required tools are not
all satisfied by the user's owned tools.

Scope decisions for this PoC:

- **Tool checking**: every tool listed in ``edge.tools_required`` must
  be satisfied by ``profile.tools_owned`` via
  :func:`recipe_optimizer.data_io.tool_use_table.is_tool_owned`
  (exact match, composite ``"a+b"`` decomposition, or one-way substring
  containment).
- **Ingredient checking**: currently a no-op. Static profiles do not
  enumerate the user's pantry, and dynamic constraint updates are
  deferred to a later iteration.
- **Burner / heat-source contention**: a *temporal* concern handled by
  the scheduler. The checker only reports static infeasibility, never
  resource conflicts that depend on parallel execution.
"""

from __future__ import annotations

from ..data_io.tool_use_table import is_tool_owned
from ..schemas import ConstraintViolation, Constraints, RecipeDAG


REASON_MISSING_TOOL = "missing_tool"


def find_violations(
    dag: RecipeDAG, constraints: Constraints
) -> list[ConstraintViolation]:
    """Identify edges in ``dag`` whose required tools are not all owned.

    Returns the violations in the same order as ``dag.edges`` so callers
    can iterate deterministically. Edges with empty ``tools_required``
    (e.g., a ``serve`` action) never produce a violation.
    """
    owned = constraints.profile.tools_owned
    violations: list[ConstraintViolation] = []

    for edge in dag.edges:
        missing = [tool.name for tool in edge.tools_required if not is_tool_owned(tool, owned)]
        if missing:
            violations.append(
                ConstraintViolation(
                    edge_id=edge.id,
                    reason=REASON_MISSING_TOOL,
                    missing=missing,
                )
            )

    return violations


def is_satisfiable(dag: RecipeDAG, constraints: Constraints) -> bool:
    """True iff ``find_violations`` returns no violations."""
    return not find_violations(dag, constraints)
