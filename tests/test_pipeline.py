"""Tests for pipeline.optimize_from_dag under the resource model."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile, load_raw_recipe, load_table
from recipe_optimizer.llm import MockLLMClient
from recipe_optimizer.modules.proposer import PROPOSER_LABEL, LLMProposal
from recipe_optimizer.pipeline import (
    UnresolvableRecipeError,
    optimize_from_dag,
)
from recipe_optimizer.schemas import (
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    Resource,
    ResourceKind,
    ResourceRequirement,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag
from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"
RECIPES_DIR = REPO_ROOT / "data" / "recipes"


def test_optimize_oyakodon_no_substitutions_needed():
    dag = make_oyakodon_dag()
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    raw = load_raw_recipe(RECIPES_DIR / "oyakodon.json")

    rendered = optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=MockLLMClient(),
        raw_recipe=raw,
    )

    assert rendered.title == "親子丼"
    assert rendered.substitutions_made == []
    assert len(rendered.numbered_steps) == len(dag.edges)
    assert rendered.schedule.total_duration_min > 0
    assert rendered.mermaid_dag.startswith("flowchart TD")
    assert len(rendered.shopping_list.ingredients) > 0
    assert len(rendered.shopping_list.resources_needed) > 0


def test_optimize_mugicha_resolves_yakan_violations():
    dag = make_mugicha_dag()
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    rendered = optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=MockLLMClient(),
    )

    assert len(rendered.substitutions_made) >= 1
    # No やかん in any resulting resource hint
    for edge in rendered.optimized_dag.edges:
        hints = {u.name_hint for u in edge.resource_uses}
        assert "やかん" not in hints


def test_optimize_unresolvable_raises_when_flag_set():
    """Empty profile, edge requires container — no candidates → unresolvable."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="水"),
            GoalNode(id="b", description="湯", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.BOIL_WATER,
                description="湯を沸かす",
                duration_min=5.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0),
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="やかん",
                        hold_duration_min=5.0,
                    ),
                ],
            )
        ],
        final_node_id="b",
    )
    minimal_profile = UserProfile(
        user_id="bare",
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
        utensils=[Resource(id="k", kind=ResourceKind.UTENSIL, name="包丁")],
    )
    table = load_table(TABLE_PATH)
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, LLMProposal(candidates=[]))

    with pytest.raises(UnresolvableRecipeError):
        optimize_from_dag(
            dag=dag,
            profile=minimal_profile,
            table=table,
            client=client,
        )


def test_optimize_unresolvable_returns_partial_when_flag_off():
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description="水"),
            GoalNode(id="b", description="湯", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.BOIL_WATER,
                description="湯を沸かす",
                duration_min=5.0,
                resource_uses=[
                    ResourceRequirement(kind=ResourceKind.COOK, hold_duration_min=1.0),
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="やかん",
                        hold_duration_min=5.0,
                    ),
                ],
            )
        ],
        final_node_id="b",
    )
    minimal_profile = UserProfile(
        user_id="bare",
        cooks=[Resource(id="c", kind=ResourceKind.COOK, name="自分")],
    )
    table = load_table(TABLE_PATH)
    client = MockLLMClient()
    client.register(PROPOSER_LABEL, LLMProposal(candidates=[]))

    rendered = optimize_from_dag(
        dag=dag,
        profile=minimal_profile,
        table=table,
        client=client,
        raise_on_unresolvable=False,
    )
    assert rendered.substitutions_made == []


def test_optimize_critical_path_present():
    dag = make_oyakodon_dag()
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    rendered = optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=MockLLMClient(),
    )
    assert len(rendered.schedule.critical_path_edge_ids) > 0
    assert "linkStyle" in rendered.mermaid_dag
