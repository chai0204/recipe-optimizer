"""Selector: choose the best SubstitutionCandidate from a list.

Pure-algorithmic module (no LLM). Scores each candidate by a weighted
combination of ``quality_delta`` (higher is better) and
``time_delta_min`` (lower is better), then picks the maximum.

The default weighting prefers quality preservation over speed: a 1-unit
loss in ``quality_factor`` requires the substitute to be 20 minutes
faster to break even (``quality:time = 1.0 : 0.05``). This matches the
intuition that home cooks generally prefer the dish to taste right
even at the cost of a few extra minutes.

Output: the chosen candidate is returned with its ``score`` field
populated (immutable: ``model_copy(update={...})`` is used). The other
candidates remain unscored unless ``rank_candidates`` is used.
"""

from __future__ import annotations

from typing import TypedDict

from ..schemas import SubstitutionCandidate


class Weights(TypedDict):
    """Coefficients in the linear score function."""

    quality: float
    time_min: float


DEFAULT_WEIGHTS: Weights = {
    "quality": 1.0,
    "time_min": 0.05,
}


def score_candidate(
    candidate: SubstitutionCandidate,
    weights: Weights | None = None,
) -> float:
    """Linear combination of quality and time deltas.

    Higher is better. Negative ``quality_delta`` (worse quality) and
    positive ``time_delta_min`` (slower) both push the score down.
    """
    w = weights or DEFAULT_WEIGHTS
    return (
        candidate.quality_delta * w["quality"]
        - candidate.time_delta_min * w["time_min"]
    )


def rank_candidates(
    candidates: list[SubstitutionCandidate],
    weights: Weights | None = None,
) -> list[SubstitutionCandidate]:
    """Return all candidates sorted best-first, each with ``score`` populated.

    Stable: ties resolve in the input order (which is typically the
    proposer's table-order, i.e. quality-descending).
    """
    scored: list[tuple[float, int, SubstitutionCandidate]] = [
        (score_candidate(c, weights), idx, c) for idx, c in enumerate(candidates)
    ]
    # Sort by score desc, then by original index asc (stable tiebreaker)
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [c.model_copy(update={"score": s}) for s, _, c in scored]


def select_best(
    candidates: list[SubstitutionCandidate],
    weights: Weights | None = None,
) -> SubstitutionCandidate | None:
    """Return the highest-scoring candidate, or ``None`` if list is empty."""
    if not candidates:
        return None
    ranked = rank_candidates(candidates, weights)
    return ranked[0]
