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
)

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = REPO_ROOT / "data" / "recipes"


# ---------------------------------------------------------------------------
# Schema-level: RecipeDAG integrity validator
# ---------------------------------------------------------------------------


def _trivial_recipe_dag(**overrides):
    """Build a minimal valid DAG, optionally overriding fields for failure tests."""
    base = {
        "title": "test",
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
                attentive_min=1.0,
            )
        ],
        "final_node_id": "b",
    }
    base.update(overrides)
    return RecipeDAG(**base)


def test_recipe_dag_accepts_valid():
    dag = _trivial_recipe_dag()
    assert dag.final_node_id == "b"


def test_recipe_dag_rejects_dangling_to_node():
    with pytest.raises(ValidationError, match="to_node 'missing' not found"):
        _trivial_recipe_dag(
            edges=[
                ProcessEdge(
                    id="e1",
                    from_nodes=["a"],
                    to_node="missing",
                    action=ProcessType.MIX,
                    description="x",
                )
            ]
        )


def test_recipe_dag_rejects_dangling_from_node():
    with pytest.raises(ValidationError, match="from_node 'ghost' not found"):
        _trivial_recipe_dag(
            edges=[
                ProcessEdge(
                    id="e1",
                    from_nodes=["ghost"],
                    to_node="b",
                    action=ProcessType.MIX,
                    description="x",
                )
            ]
        )


def test_recipe_dag_rejects_missing_final_node():
    with pytest.raises(ValidationError, match="final_node_id 'z' not found"):
        _trivial_recipe_dag(final_node_id="z")


def test_recipe_dag_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError, match="Duplicate node IDs"):
        _trivial_recipe_dag(
            nodes=[
                GoalNode(id="a", description="x"),
                GoalNode(id="a", description="y", is_final=True),
            ],
            final_node_id="a",
        )


def test_recipe_dag_rejects_attentive_exceeding_duration():
    with pytest.raises(ValidationError, match="attentive_min .* exceeds duration_min"):
        _trivial_recipe_dag(
            edges=[
                ProcessEdge(
                    id="e1",
                    from_nodes=["a"],
                    to_node="b",
                    action=ProcessType.SIMMER,
                    description="x",
                    duration_min=1.0,
                    attentive_min=2.0,
                )
            ]
        )


# ---------------------------------------------------------------------------
# Hand-crafted oyakodon fixture
# ---------------------------------------------------------------------------


def test_oyakodon_fixture_is_valid_dag():
    """The hand-crafted fixture must itself satisfy DAG integrity."""
    dag = make_oyakodon_dag()
    assert dag.title == "親子丼"
    assert dag.final_node_id == "n_done"
    assert any(n.is_final for n in dag.nodes)
    # Every edge resolves
    node_ids = {n.id for n in dag.nodes}
    for e in dag.edges:
        assert e.to_node in node_ids
        assert all(f in node_ids for f in e.from_nodes)


def test_oyakodon_fixture_has_implicit_prep_edges():
    """The fixture must include implicit prep (e.g., chopping chicken),
    which is the parser behaviour we'll require from the real LLM."""
    dag = make_oyakodon_dag()
    actions = {e.action for e in dag.edges}
    assert ProcessType.CHOP in actions  # chicken
    assert ProcessType.SLICE in actions  # onion
    assert ProcessType.BEAT in actions  # eggs


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
    assert dag.final_node_id == "n_done"
    assert len(dag.nodes) == len(canned.nodes)
    assert len(dag.edges) == len(canned.edges)


def test_parser_populates_raw_text_from_prompt():
    canned = make_oyakodon_dag()
    client = MockLLMClient()
    client.register(PARSER_LABEL, canned)

    parser = RecipeParser(client=client)
    raw = load_raw_recipe(RECIPE_DIR / "oyakodon.json")
    dag = parser.parse(raw)

    assert dag.raw_text != ""
    assert "親子丼" in dag.raw_text
    # Ingredient lines and steps should appear in the prompt
    assert "鶏肉（もも肉）" in dag.raw_text
    assert "蓋をして中火で30秒" in dag.raw_text


def test_parser_logs_call_with_label():
    canned = make_oyakodon_dag()
    client = MockLLMClient()
    client.register(PARSER_LABEL, canned)

    parser = RecipeParser(client=client)
    parser.parse(load_raw_recipe(RECIPE_DIR / "oyakodon.json"))

    assert len(client.call_logs) == 1
    log = client.call_logs[0]
    assert log.label == PARSER_LABEL
    assert log.success
    assert log.output_schema_name == "RecipeDAG"


def test_parser_propagates_validation_error_on_malformed_dag():
    """If the mock returns a dict that doesn't pass RecipeDAG validation,
    LLMOutputError should propagate (and on real LLM the retry loop kicks in)."""
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
                    "to_node": "ghost",  # dangling reference
                    "action": "mix",
                    "description": "broken",
                }
            ],
            "final_node_id": "a",
        },
    )

    parser = RecipeParser(client=client)
    raw = load_raw_recipe(RECIPE_DIR / "oyakodon.json")

    with pytest.raises(LLMOutputError, match="failed validation"):
        parser.parse(raw)


def test_parser_works_for_nikujaga_input_format():
    """Smoke: nikujaga.json loads as RawRecipe and parser flows."""
    client = MockLLMClient()
    # For nikujaga, just register the same fixture (we're testing flow, not content)
    client.register(PARSER_LABEL, make_oyakodon_dag())

    parser = RecipeParser(client=client)
    raw = load_raw_recipe(RECIPE_DIR / "nikujaga.json")
    assert isinstance(raw, RawRecipe)
    assert raw.title == "肉じゃが"

    dag = parser.parse(raw)
    assert dag.title == "親子丼"  # because we registered oyakodon as the canned response
