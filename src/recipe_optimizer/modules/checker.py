"""Constraint checker: identify resource requirements the profile cannot satisfy.

Pure-algorithmic. For every ``ResourceRequirement`` on every edge,
look for a matching resource in the user's pool (kind + name_hint).
Any unmet requirement becomes a ``ConstraintViolation``.

The scheduler enforces temporal contention separately — the checker
only sees *static* infeasibility (no compatible resource exists at all).
"""

from __future__ import annotations

from ..data_io.tool_use_table import can_satisfy_spec
from ..schemas import (
    ConstraintViolation,
    Constraints,
    RecipeDAG,
    ResourceSpec,
)


REASON_MISSING_RESOURCE = "missing_resource"


def find_violations(
    dag: RecipeDAG, constraints: Constraints
) -> list[ConstraintViolation]:
    """One violation per unsatisfiable ``ResourceRequirement``.

    Order follows ``dag.edges`` then the order of ``resource_uses``
    inside each edge for determinism.
    """
    profile = constraints.profile
    owned = (
        profile.cooks
        + profile.burners
        + profile.workstations
        + profile.containers
        + profile.appliances
        + profile.utensils
    )

    violations: list[ConstraintViolation] = []
    for edge in dag.edges:
        for use in edge.resource_uses:
            spec = ResourceSpec(
                kind=use.kind,
                name_hint=use.name_hint,
                relative_duration=1.0,
            )
            if not can_satisfy_spec(spec, owned):
                violations.append(
                    ConstraintViolation(
                        edge_id=edge.id,
                        reason=REASON_MISSING_RESOURCE,
                        missing_kind=use.kind,
                        missing_name_hint=use.name_hint,
                    )
                )
    return violations


def is_satisfiable(dag: RecipeDAG, constraints: Constraints) -> bool:
    return not find_violations(dag, constraints)
