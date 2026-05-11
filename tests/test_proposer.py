"""Tests for the proposer under the resource model."""

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
)
from recipe_optimizer.schemas import (
    ConstraintViolation,
    Constraints,
    ProcessEdge,
    ProcessType,
    Resource,
    ResourceKind,
    ResourceRequirement,
    ResourceSpec,
    ToolOption,
    ToolUseTable,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


def _profile_with(**pools) -> UserProfile:
    return UserProfile(user_id="t", **pools)


# ---------------------------------------------------------------------------
# Table-hit path
# ---------------------------------------------------------------------------


def test_proposer_table_hit_for_mugicha_boil_water():
    """Demo profile lacks やかん → table proposes 片手鍋 / 両手鍋 / 電子レンジ."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    client = MockLLMClient()
    proposer = RecipeProposer(client=client, table=table)

    dag = make_mugicha_dag()
    edges_by_id = {e.id: e for e in dag.edges}
    violations = find_violations(dag, Constraints(profile=profile))
    boil_v = next(v for v in violations if v.edge_id == "e_boil")

    candidates = proposer.propose(boil_v, edges_by_id["e_boil"], profile)

    assert 0 < len(candidates) <= MAX_CANDIDATES
    for c in candidates:
        # No やかん anywhere in the replacement
        hints = {u.name_hint for u in c.replacement_edge.resource_uses}
        assert "やかん" not in hints

    # Mock LLM was NOT called
    assert len(client.call_logs) == 0


def test_proposer_microwave_alternative_drops_burner_automatically():
    """The 電子レンジ option in the table has no BURNER spec, so the
    replacement_edge built from it also has no BURNER use — no need
    for the old appliance-heuristic in proposer code."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    proposer = RecipeProposer(client=MockLLMClient(), table=table)

    dag = make_mugicha_dag()
    edges_by_id = {e.id: e for e in dag.edges}
    boil_v = next(
        v for v in find_violations(dag, Constraints(profile=profile))
        if v.edge_id == "e_boil"
    )
    candidates = proposer.propose(boil_v, edges_by_id["e_boil"], profile)

    microwave = next(
        c for c in candidates
        if any(
            u.kind == ResourceKind.APPLIANCE and u.name_hint == "電子レンジ"
            for u in c.replacement_edge.resource_uses
        )
    )
    assert all(
        u.kind != ResourceKind.BURNER
        for u in microwave.replacement_edge.resource_uses
    )


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


def test_proposer_falls_back_to_llm_when_table_empty():
    profile = _profile_with(
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
        containers=[Resource(id="p", kind=ResourceKind.CONTAINER, name="片手鍋")],
        burners=[Resource(id="b", kind=ResourceKind.BURNER, name="コンロ口")],
    )
    table = ToolUseTable(entries={})

    canned = LLMProposal(
        candidates=[
            ToolOption(
                label="片手鍋+コンロ",
                resources=[
                    ResourceSpec(kind=ResourceKind.COOK, relative_duration=0.2),
                    ResourceSpec(kind=ResourceKind.BURNER, relative_duration=1.0),
                    ResourceSpec(
                        kind=ResourceKind.CONTAINER, name_hint="片手鍋"
                    ),
                ],
                time_factor=1.15,
            )
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)

    proposer = RecipeProposer(client=client, table=table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL_WATER,
        description="やかんで沸かす",
        duration_min=5.0,
        resource_uses=[
            ResourceRequirement(
                kind=ResourceKind.CONTAINER,
                name_hint="やかん",
                hold_duration_min=5.0,
            )
        ],
    )
    violation = ConstraintViolation(
        edge_id="e",
        reason="missing_resource",
        missing_kind=ResourceKind.CONTAINER,
        missing_name_hint="やかん",
    )

    candidates = proposer.propose(violation, edge, profile)
    assert len(candidates) == 1
    assert len(client.call_logs) == 1

    # Table updated with the LLM-discovered option, tagged source=llm
    assert "boil_water" in table.entries
    assert any(opt.source == "llm" for opt in table.entries["boil_water"])


def test_proposer_filters_unowned_llm_options():
    profile = _profile_with(
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
        containers=[Resource(id="p", kind=ResourceKind.CONTAINER, name="片手鍋")],
        burners=[Resource(id="b", kind=ResourceKind.BURNER, name="コンロ口")],
    )
    table = ToolUseTable(entries={})

    canned = LLMProposal(
        candidates=[
            ToolOption(
                label="圧力鍋",
                resources=[
                    ResourceSpec(kind=ResourceKind.COOK),
                    ResourceSpec(kind=ResourceKind.BURNER),
                    ResourceSpec(
                        kind=ResourceKind.CONTAINER, name_hint="圧力鍋"
                    ),
                ],
            ),
            ToolOption(
                label="片手鍋+コンロ",
                resources=[
                    ResourceSpec(kind=ResourceKind.COOK),
                    ResourceSpec(kind=ResourceKind.BURNER),
                    ResourceSpec(
                        kind=ResourceKind.CONTAINER, name_hint="片手鍋"
                    ),
                ],
            ),
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)

    proposer = RecipeProposer(client=client, table=table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL,
        description="x",
        duration_min=5.0,
        resource_uses=[
            ResourceRequirement(
                kind=ResourceKind.CONTAINER,
                name_hint="やかん",
                hold_duration_min=5.0,
            )
        ],
    )
    violation = ConstraintViolation(
        edge_id="e",
        reason="missing_resource",
        missing_kind=ResourceKind.CONTAINER,
        missing_name_hint="やかん",
    )

    candidates = proposer.propose(violation, edge, profile)
    assert len(candidates) == 1
    labels_kept = [
        u.name_hint for u in candidates[0].replacement_edge.resource_uses
    ]
    assert "片手鍋" in labels_kept
    assert "圧力鍋" not in labels_kept

    boil_options = table.entries.get("boil", [])
    assert all(opt.label != "圧力鍋" for opt in boil_options)


def test_proposer_returns_empty_when_no_valid_alternative():
    profile = _profile_with(
        utensils=[Resource(id="k", kind=ResourceKind.UTENSIL, name="包丁")]
    )
    table = ToolUseTable(entries={})
    canned = LLMProposal(
        candidates=[
            ToolOption(
                label="圧力鍋",
                resources=[
                    ResourceSpec(
                        kind=ResourceKind.CONTAINER, name_hint="圧力鍋"
                    )
                ],
            )
        ]
    )
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, canned)
    proposer = RecipeProposer(client=client, table=table)
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.BOIL,
        description="x",
        duration_min=5.0,
        resource_uses=[
            ResourceRequirement(
                kind=ResourceKind.CONTAINER,
                name_hint="やかん",
                hold_duration_min=5.0,
            )
        ],
    )
    violation = ConstraintViolation(
        edge_id="e",
        reason="missing_resource",
        missing_kind=ResourceKind.CONTAINER,
        missing_name_hint="やかん",
    )
    assert proposer.propose(violation, edge, profile) == []
