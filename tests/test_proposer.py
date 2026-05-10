"""Tests for the proposer module (table-hit path + LLM fallback)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile, load_table
from recipe_optimizer.llm import MockLLMClient
from recipe_optimizer.modules.checker import find_violations
from recipe_optimizer.modules.proposer import (
    MAX_CANDIDATES,
    PROPOSER_LABEL,
    LLMProposal,
    RecipeProposer,
    _build_replacement_edge,
    _option_to_candidate,
)
from recipe_optimizer.schemas import (
    ConstraintViolation,
    Constraints,
    ProcessEdge,
    ProcessType,
    Tool,
    ToolKind,
    ToolOption,
    ToolUseTable,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


# ---------------------------------------------------------------------------
# _build_replacement_edge
# ---------------------------------------------------------------------------


def test_build_replacement_edge_swaps_missing_with_option_tool():
    original = ProcessEdge(
        id="e_boil",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="やかんで湯を沸かす",
        tools_required=[
            Tool(name="やかん", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
        ],
        duration_min=5.0,
        attentive_min=1.0,
    )
    opt = ToolOption(
        tool=Tool(name="片手鍋", kind=ToolKind.CONTAINER),
        time_factor=1.15,
        quality_factor=1.0,
    )
    new_edge = _build_replacement_edge(original, opt, ["やかん"])

    names = [t.name for t in new_edge.tools_required]
    assert "コンロ口" in names  # preserved
    assert "片手鍋" in names  # added
    assert "やかん" not in names  # removed

    # Duration scales by time_factor
    assert new_edge.duration_min == pytest.approx(5.0 * 1.15)
    assert new_edge.attentive_min == pytest.approx(1.0 * 1.15)

    # Description rewrites yakan → 片手鍋
    assert "片手鍋" in new_edge.description
    assert "やかん" not in new_edge.description

    # Identity preserved
    assert new_edge.id == "e_boil"
    assert new_edge.action == ProcessType.BOIL_WATER
    assert new_edge.from_nodes == ["a"]
    assert new_edge.to_node == "b"


def test_build_replacement_edge_appliance_drops_external_heat_source():
    """Substituting with an appliance (e.g. microwave) drops the burner
    from preserved tools — the appliance has its own heat."""
    original = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="やかんで湯を沸かす",
        tools_required=[
            Tool(name="やかん", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
        ],
        duration_min=5.0,
        attentive_min=1.0,
    )
    opt = ToolOption(
        tool=Tool(name="電子レンジ", kind=ToolKind.APPLIANCE),
        time_factor=1.0,
        quality_factor=0.9,
    )
    new_edge = _build_replacement_edge(original, opt, ["やかん"])

    kinds = {t.kind for t in new_edge.tools_required}
    assert ToolKind.HEAT_SOURCE not in kinds  # コンロ口 dropped
    assert ToolKind.APPLIANCE in kinds  # 電子レンジ added


def test_build_replacement_edge_attentive_capped_at_duration():
    original = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL,
        description="x",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
        duration_min=2.0,
        attentive_min=2.0,
    )
    # time_factor < 1 (faster); attentive should not exceed new duration
    opt = ToolOption(
        tool=Tool(name="電気ケトル", kind=ToolKind.APPLIANCE),
        time_factor=0.5,
        quality_factor=1.0,
    )
    new_edge = _build_replacement_edge(original, opt, ["やかん"])
    assert new_edge.duration_min == pytest.approx(1.0)
    assert new_edge.attentive_min <= new_edge.duration_min


# ---------------------------------------------------------------------------
# _option_to_candidate
# ---------------------------------------------------------------------------


def test_option_to_candidate_computes_deltas_and_rationale():
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="x",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
        duration_min=10.0,
        attentive_min=2.0,
    )
    opt = ToolOption(
        tool=Tool(name="片手鍋", kind=ToolKind.CONTAINER),
        time_factor=1.2,
        quality_factor=0.95,
    )
    cand = _option_to_candidate(opt, edge, ["やかん"])

    assert cand.original_edge_id == "e"
    assert cand.time_delta_min == pytest.approx(2.0)  # 10*1.2 - 10
    assert cand.quality_delta == pytest.approx(-0.05)
    assert "片手鍋" in cand.rationale
    assert "やかん" in cand.rationale


# ---------------------------------------------------------------------------
# Table-hit path: no LLM call
# ---------------------------------------------------------------------------


def test_proposer_table_hit_for_mugicha_boil_water():
    """User lacks やかん but has 片手鍋/両手鍋/電子レンジ → table hit, no LLM."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    client = MockLLMClient()  # not registered for "proposer"
    proposer = RecipeProposer(client=client, table=table)

    edge = ProcessEdge(
        id="e_boil",
        from_nodes=["water"],
        to_node="boiling",
        action=ProcessType.BOIL_WATER,
        description="やかんで湯を沸かす",
        tools_required=[
            Tool(name="やかん", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
        ],
        duration_min=5.0,
        attentive_min=1.0,
    )
    violation = ConstraintViolation(
        edge_id="e_boil", reason="missing_tool", missing=["やかん"]
    )

    candidates = proposer.propose(violation, edge, profile)

    assert 0 < len(candidates) <= MAX_CANDIDATES

    # No やかん in any replacement
    for c in candidates:
        names = {t.name for t in c.replacement_edge.tools_required}
        assert "やかん" not in names

    # Mock LLM was NOT called
    assert len(client.call_logs) == 0


def test_proposer_proposes_pot_for_mugicha_via_full_pipeline():
    """End-to-end: checker → proposer for the 麦茶 example.

    Validates that 片手鍋 (or 両手鍋) appears among proposed alternatives —
    the canonical case discussed in the design conversation.
    """
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    client = MockLLMClient()
    proposer = RecipeProposer(client=client, table=table)

    dag = make_mugicha_dag()
    violations = find_violations(dag, Constraints(profile=profile))
    edges_by_id = {e.id: e for e in dag.edges}

    boil_violation = next(v for v in violations if v.edge_id == "e_boil")
    candidates = proposer.propose(boil_violation, edges_by_id["e_boil"], profile)

    proposed_tool_names = set()
    for c in candidates:
        for t in c.replacement_edge.tools_required:
            proposed_tool_names.add(t.name)

    # At least one of these owned containers should be proposed
    assert proposed_tool_names & {"片手鍋", "両手鍋", "電子レンジ"}


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


def test_proposer_falls_back_to_llm_when_table_empty():
    """Empty table → LLM is invoked, candidates filtered + persisted."""
    profile = UserProfile(
        user_id="t",
        tools_owned=[
            Tool(name="片手鍋", kind=ToolKind.CONTAINER),
            Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
        ],
    )
    empty_table = ToolUseTable(entries={})

    canned = LLMProposal(
        candidates=[
            ToolOption(
                tool=Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                time_factor=1.15,
                quality_factor=1.0,
            )
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)

    proposer = RecipeProposer(client=client, table=empty_table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="やかんで沸かす",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
        duration_min=5.0,
        attentive_min=1.0,
    )
    violation = ConstraintViolation(
        edge_id="e", reason="missing_tool", missing=["やかん"]
    )

    candidates = proposer.propose(violation, edge, profile)

    # 片手鍋 candidate produced
    assert len(candidates) == 1
    assert any(
        t.name == "片手鍋" for t in candidates[0].replacement_edge.tools_required
    )

    # LLM call was logged
    assert len(client.call_logs) == 1
    assert client.call_logs[0].label == PROPOSER_LABEL
    assert client.call_logs[0].success

    # Table updated with the LLM-discovered combination, marked source=llm
    assert "boil_water" in empty_table.entries
    pot_entry = next(
        opt for opt in empty_table.entries["boil_water"] if opt.tool.name == "片手鍋"
    )
    assert pot_entry.source == "llm"


def test_proposer_filters_unowned_llm_proposals():
    """LLM proposals for tools the user lacks must be dropped."""
    profile = UserProfile(
        user_id="t",
        tools_owned=[Tool(name="片手鍋", kind=ToolKind.CONTAINER)],
    )
    empty_table = ToolUseTable(entries={})

    canned = LLMProposal(
        candidates=[
            ToolOption(
                tool=Tool(name="圧力鍋", kind=ToolKind.CONTAINER),  # not owned
                time_factor=0.5,
                quality_factor=1.0,
            ),
            ToolOption(
                tool=Tool(name="片手鍋", kind=ToolKind.CONTAINER),  # owned
                time_factor=1.15,
                quality_factor=1.0,
            ),
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)

    proposer = RecipeProposer(client=client, table=empty_table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL,
        description="x",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
        duration_min=5.0,
        attentive_min=1.0,
    )
    violation = ConstraintViolation(
        edge_id="e", reason="missing_tool", missing=["やかん"]
    )

    candidates = proposer.propose(violation, edge, profile)

    assert len(candidates) == 1
    new_names = {t.name for t in candidates[0].replacement_edge.tools_required}
    assert "片手鍋" in new_names
    assert "圧力鍋" not in new_names

    # Only the validated candidate persists to the table
    boil_options = empty_table.entries.get("boil", [])
    assert all(opt.tool.name != "圧力鍋" for opt in boil_options)


def test_proposer_returns_empty_when_no_valid_alternative():
    """Table empty + LLM proposes only unowned tools → no candidates."""
    profile = UserProfile(
        user_id="t",
        tools_owned=[Tool(name="包丁", kind=ToolKind.UTENSIL)],
    )
    empty_table = ToolUseTable(entries={})

    canned = LLMProposal(
        candidates=[
            ToolOption(
                tool=Tool(name="圧力鍋", kind=ToolKind.CONTAINER),  # not owned
                time_factor=0.5,
                quality_factor=1.0,
            )
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)

    proposer = RecipeProposer(client=client, table=empty_table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL,
        description="x",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
    )
    violation = ConstraintViolation(
        edge_id="e", reason="missing_tool", missing=["やかん"]
    )

    candidates = proposer.propose(violation, edge, profile)
    assert candidates == []


def test_proposer_caps_at_max_candidates():
    """If table has more options than MAX_CANDIDATES, only top N are returned."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    client = MockLLMClient()
    proposer = RecipeProposer(client=client, table=table)

    # Use BOIL_WATER which has multiple compatible options under demo profile
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="x",
        tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
        duration_min=5.0,
        attentive_min=1.0,
    )
    violation = ConstraintViolation(
        edge_id="e", reason="missing_tool", missing=["やかん"]
    )
    candidates = proposer.propose(violation, edge, profile)
    assert len(candidates) <= MAX_CANDIDATES
