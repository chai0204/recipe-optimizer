"""Tests for the RecipeParser module (Mock client path)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recipe_optimizer.data_io import load_raw_recipe
from recipe_optimizer.llm import LLMOutputError, MockLLMClient
from recipe_optimizer.modules.parser import PARSER_LABEL, RecipeParser
from recipe_optimizer.schemas import (
    GoalNode,
    ProcessEdge,
    ProcessType,
    RawRecipe,
    RecipeDAG,
    ResourceKind,
    ResourceRequirement,
)

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = REPO_ROOT / "data" / "recipes"


# ---------------------------------------------------------------------------
# RecipeDAG validators
# ---------------------------------------------------------------------------


def _trivial(**overrides):
    base = {
        "title": "t",
        "servings": 1,
        "nodes": [
            GoalNode(id="a", description="raw"),
            GoalNode(id="b", description="done", is_final=True),
        ],
        "edges": [
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.MIX,
                description="combine",
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0)
                ],
            )
        ],
        "final_node_id": "b",
    }
    base.update(overrides)
    return RecipeDAG(**base)


def test_recipe_dag_accepts_valid():
    dag = _trivial()
    assert dag.final_node_id == "b"


def test_recipe_dag_rejects_dangling_to_node():
    with pytest.raises(ValidationError):
        _trivial(
            edges=[
                ProcessEdge(
                    id="e1",
                    from_nodes=["a"],
                    to_node="missing",
                    action=ProcessType.MIX,
                    description="x",
                    duration_min=1.0,
                )
            ]
        )


def test_recipe_dag_rejects_missing_final_node():
    with pytest.raises(ValidationError):
        _trivial(final_node_id="z")


def test_recipe_dag_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError):
        _trivial(
            nodes=[
                GoalNode(id="a", description="x"),
                GoalNode(id="a", description="y", is_final=True),
            ],
            final_node_id="a",
        )


def test_process_edge_rejects_resource_use_exceeding_duration():
    with pytest.raises(ValidationError):
        ProcessEdge(
            id="e",
            from_nodes=["a"],
            to_node="b",
            action=ProcessType.SIMMER,
            description="x",
            duration_min=1.0,
            resource_uses=[
                ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=2.0)
            ],
        )


def test_process_edge_allows_partial_hold():
    """Resource held for less than the edge duration is fine."""
    edge = ProcessEdge(
        id="e",
        from_nodes=["a"],
        to_node="b",
        action=ProcessType.SIMMER,
        description="x",
        duration_min=5.0,
        resource_uses=[
            ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=0.5),
            ResourceRequirement(kind=ResourceKind.BURNER, hold_duration_min=5.0),
        ],
    )
    assert edge.duration_min == 5.0


# ---------------------------------------------------------------------------
# Oyakodon fixture sanity
# ---------------------------------------------------------------------------


def test_oyakodon_fixture_is_valid():
    dag = make_oyakodon_dag()
    assert dag.title == "親子丼"
    assert dag.final_node_id == "n_done"


def test_oyakodon_has_implicit_prep_edges():
    dag = make_oyakodon_dag()
    actions = {e.action for e in dag.edges}
    assert ProcessType.CHOP in actions
    assert ProcessType.SLICE in actions
    assert ProcessType.BEAT in actions


# ---------------------------------------------------------------------------
# Parser via Mock
# ---------------------------------------------------------------------------


def test_parser_returns_canned_dag_via_mock():
    canned = make_oyakodon_dag()
    client = MockLLMClient()
    client.register(PARSER_LABEL, canned)

    parser = RecipeParser(client=client)
    raw = load_raw_recipe(RECIPE_DIR / "oyakodon.json")
    dag = parser.parse(raw)

    assert dag.title == "親子丼"
    assert len(dag.nodes) == len(canned.nodes)
    assert len(dag.edges) == len(canned.edges)


def test_parser_logs_call():
    canned = make_oyakodon_dag()
    client = MockLLMClient()
    client.register(PARSER_LABEL, canned)
    parser = RecipeParser(client=client)
    parser.parse(load_raw_recipe(RECIPE_DIR / "oyakodon.json"))
    assert len(client.call_logs) == 1
    assert client.call_logs[0].label == PARSER_LABEL


def test_parser_propagates_validation_error_on_malformed_dag():
    client = MockLLMClient()
    client.register(
        PARSER_LABEL,
        {
            "title": "broken",
            "servings": 1,
            "nodes": [{"id": "a", "description": "x"}],
            "edges": [
                {
                    "id": "e1",
                    "from_nodes": ["a"],
                    "to_node": "ghost",
                    "action": "mix",
                    "description": "broken",
                    "duration_min": 1.0,
                    "resource_uses": [],
                }
            ],
            "final_node_id": "a",
        },
    )
    parser = RecipeParser(client=client)
    raw = load_raw_recipe(RECIPE_DIR / "oyakodon.json")
    with pytest.raises(LLMOutputError):
        parser.parse(raw)
