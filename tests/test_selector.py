"""Tests for the selector module (pure-algorithmic candidate scoring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile, load_table
from recipe_optimizer.llm import MockLLMClient
from recipe_optimizer.modules.checker import find_violations
from recipe_optimizer.modules.proposer import RecipeProposer
from recipe_optimizer.modules.selector import (
    DEFAULT_WEIGHTS,
    Weights,
    rank_candidates,
    score_candidate,
    select_best,
)
from recipe_optimizer.schemas import (
    Constraints,
    ProcessEdge,
    ProcessType,
    SubstitutionCandidate,
    Tool,
    ToolKind,
)

from .fixtures.mugicha_dag import make_mugicha_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


def _candidate(
    *,
    edge_id: str = "e",
    tool_name: str = "x",
    quality_delta: float = 0.0,
    time_delta_min: float = 0.0,
) -> SubstitutionCandidate:
    """Build a minimal SubstitutionCandidate fixture."""
    replacement = ProcessEdge(
        id=edge_id,
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.MIX,
        description="r",
        tools_required=[Tool(name=tool_name, kind=ToolKind.CONTAINER)],
    )
    return SubstitutionCandidate(
        original_edge_id=edge_id,
        replacement_edge=replacement,
        rationale="",
        quality_delta=quality_delta,
        time_delta_min=time_delta_min,
    )


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


def test_score_zero_when_no_delta():
    c = _candidate()
    assert score_candidate(c) == pytest.approx(0.0)


def test_score_penalizes_time_delta():
    c = _candidate(time_delta_min=2.0)
    assert score_candidate(c) == pytest.approx(-2.0 * DEFAULT_WEIGHTS["time_min"])


def test_score_penalizes_negative_quality_delta():
    c = _candidate(quality_delta=-0.1)
    assert score_candidate(c) == pytest.approx(-0.1 * DEFAULT_WEIGHTS["quality"])


def test_score_rewards_positive_quality_delta():
    c = _candidate(quality_delta=+0.05)
    assert score_candidate(c) == pytest.approx(0.05)


def test_score_combines_quality_and_time():
    c = _candidate(quality_delta=-0.05, time_delta_min=1.0)
    expected = -0.05 - 1.0 * DEFAULT_WEIGHTS["time_min"]
    assert score_candidate(c) == pytest.approx(expected)


def test_score_with_custom_weights():
    c = _candidate(quality_delta=-0.1, time_delta_min=2.0)
    weights: Weights = {"quality": 2.0, "time_min": 0.0}  # quality-only
    assert score_candidate(c, weights) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


def test_rank_returns_empty_for_empty_input():
    assert rank_candidates([]) == []


def test_rank_orders_best_first():
    # Pre-computed scores under DEFAULT_WEIGHTS (quality=1.0, time_min=0.05):
    fast_low_quality = _candidate(tool_name="fast", quality_delta=-0.2, time_delta_min=-1.0)
    # → -0.2 - (-1.0)*0.05 = -0.15
    same_quality_slow = _candidate(tool_name="slow", quality_delta=0.0, time_delta_min=+5.0)
    # → 0 - 5*0.05 = -0.25
    high_quality_fast = _candidate(tool_name="best", quality_delta=+0.05, time_delta_min=-0.5)
    # → 0.05 - (-0.5)*0.05 = +0.075

    ranked = rank_candidates([fast_low_quality, same_quality_slow, high_quality_fast])
    names = [c.replacement_edge.tools_required[0].name for c in ranked]
    assert names == ["best", "fast", "slow"]
    # Verify scores monotonically non-increasing
    assert ranked[0].score >= ranked[1].score >= ranked[2].score


def test_rank_populates_score_field():
    c = _candidate(quality_delta=-0.05, time_delta_min=1.0)
    [out] = rank_candidates([c])
    assert out.score == pytest.approx(score_candidate(c))


def test_rank_is_stable_on_ties():
    """Ties should resolve in input order — important for determinism."""
    a = _candidate(tool_name="A", quality_delta=0.0, time_delta_min=1.0)
    b = _candidate(tool_name="B", quality_delta=0.0, time_delta_min=1.0)
    c = _candidate(tool_name="C", quality_delta=0.0, time_delta_min=1.0)
    ranked = rank_candidates([a, b, c])
    assert [r.replacement_edge.tools_required[0].name for r in ranked] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------


def test_select_best_returns_none_for_empty():
    assert select_best([]) is None


def test_select_best_picks_max_score():
    poor = _candidate(tool_name="poor", quality_delta=-0.2, time_delta_min=0.0)
    good = _candidate(tool_name="good", quality_delta=+0.05, time_delta_min=+0.5)
    chosen = select_best([poor, good])
    assert chosen is not None
    assert chosen.replacement_edge.tools_required[0].name == "good"
    assert chosen.score == pytest.approx(0.05 - 0.5 * DEFAULT_WEIGHTS["time_min"])


def test_select_best_quality_outweighs_small_time_penalty():
    """A 0.05 quality gain should outweigh a 0.5 min slower."""
    fast_worse = _candidate(tool_name="fast", quality_delta=-0.05, time_delta_min=-0.5)
    slow_better = _candidate(tool_name="slow", quality_delta=+0.05, time_delta_min=+0.5)
    chosen = select_best([fast_worse, slow_better])
    assert chosen is not None
    assert chosen.replacement_edge.tools_required[0].name == "slow"


def test_select_best_picks_fastest_when_quality_neutral():
    a = _candidate(tool_name="a", quality_delta=0.0, time_delta_min=+2.0)
    b = _candidate(tool_name="b", quality_delta=0.0, time_delta_min=+1.0)
    c = _candidate(tool_name="c", quality_delta=0.0, time_delta_min=+0.5)
    chosen = select_best([a, b, c])
    assert chosen is not None
    assert chosen.replacement_edge.tools_required[0].name == "c"


# ---------------------------------------------------------------------------
# End-to-end: mugicha boil_water
# ---------------------------------------------------------------------------


def test_select_picks_pot_for_mugicha_boil_water():
    """Integration: proposer → selector for the canonical 麦茶 case.

    With default weights, 片手鍋 (q=0, t=+0.8) should beat 両手鍋
    (q=0, t=+1.2) and 電子レンジ (q=-0.1, t=+0.5).
    """
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    proposer = RecipeProposer(client=MockLLMClient(), table=table)

    dag = make_mugicha_dag()
    edges_by_id = {e.id: e for e in dag.edges}
    violations = find_violations(dag, Constraints(profile=profile))
    boil_v = next(v for v in violations if v.edge_id == "e_boil")

    candidates = proposer.propose(boil_v, edges_by_id["e_boil"], profile)
    chosen = select_best(candidates)

    assert chosen is not None
    container_names = {
        t.name
        for t in chosen.replacement_edge.tools_required
        if t.kind in {ToolKind.CONTAINER, ToolKind.APPLIANCE}
    }
    assert "片手鍋" in container_names
    assert chosen.score is not None


def test_select_with_speed_priority_picks_microwave():
    """If we override weights to value time over quality, 電子レンジ wins
    on this small quality cost (q=-0.1) because it's the fastest."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    proposer = RecipeProposer(client=MockLLMClient(), table=table)

    dag = make_mugicha_dag()
    edges_by_id = {e.id: e for e in dag.edges}
    boil_v = next(
        v
        for v in find_violations(dag, Constraints(profile=profile))
        if v.edge_id == "e_boil"
    )
    candidates = proposer.propose(boil_v, edges_by_id["e_boil"], profile)

    # Speed-first: time penalty dominates, quality almost ignored
    speed_weights: Weights = {"quality": 0.05, "time_min": 1.0}
    chosen = select_best(candidates, weights=speed_weights)

    assert chosen is not None
    container_names = {
        t.name
        for t in chosen.replacement_edge.tools_required
        if t.kind in {ToolKind.CONTAINER, ToolKind.APPLIANCE}
    }
    assert "電子レンジ" in container_names
