"""Tests for the end-to-end pipeline orchestrator."""

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
    Tool,
    ToolKind,
    UserProfile,
)

from .fixtures.mugicha_dag import make_mugicha_dag
from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"
RECIPES_DIR = REPO_ROOT / "data" / "recipes"


def test_optimize_oyakodon_no_substitutions_needed():
    """Demo profile satisfies oyakodon as-is — no substitutions."""
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
    assert rendered.substitutions_made == []  # no violations
    assert len(rendered.numbered_steps) == len(dag.edges)
    assert rendered.schedule.total_duration_min > 0
    assert rendered.mermaid_dag.startswith("flowchart TD")
    assert len(rendered.shopping_list.ingredients) > 0
    assert len(rendered.shopping_list.tools_needed) > 0


def test_optimize_mugicha_resolves_yakan_violations():
    """Demo profile lacks やかん → substitutions applied, recipe satisfiable."""
    dag = make_mugicha_dag()
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    rendered = optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=MockLLMClient(),
    )

    # Both edges had violations → both got substitutions
    assert len(rendered.substitutions_made) == 2
    sub_ids = {s.original_edge_id for s in rendered.substitutions_made}
    assert sub_ids == {"e_boil", "e_brew"}

    # No やかん anywhere in the resulting tools
    all_tool_names = {t.name for t in rendered.shopping_list.tools_needed}
    assert "やかん" not in all_tool_names


def test_optimize_unresolvable_raises_when_flag_set():
    """A user with no usable container should fail on a boil-water DAG."""
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
                tools_required=[
                    Tool(name="やかん", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=5.0,
                attentive_min=1.0,
            )
        ],
        final_node_id="b",
    )
    minimal_profile = UserProfile(
        user_id="bare",
        tools_owned=[Tool(name="包丁", kind=ToolKind.UTENSIL)],
        burners_count=0,  # no burners → cannot satisfy heat_source
    )
    table = load_table(TABLE_PATH)

    # LLM has no useful proposal — proposer falls back to LLM, which returns empty
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
    """raise_on_unresolvable=False returns whatever could be optimized."""
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
                tools_required=[Tool(name="やかん", kind=ToolKind.CONTAINER)],
                duration_min=5.0,
                attentive_min=1.0,
            )
        ],
        final_node_id="b",
    )
    minimal_profile = UserProfile(
        user_id="bare",
        tools_owned=[Tool(name="包丁", kind=ToolKind.UTENSIL)],
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
    assert rendered.title == "t"
    # No successful substitution found
    assert rendered.substitutions_made == []


def test_optimize_with_speed_priority_weights_picks_microwave_for_mugicha():
    """Custom selector weights flow through pipeline."""
    dag = make_mugicha_dag()
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    rendered = optimize_from_dag(
        dag=dag,
        profile=profile,
        table=table,
        client=MockLLMClient(),
        selector_weights={"quality": 0.05, "time_min": 1.0},
    )

    # Look for the boil_water substitution
    boil_sub = next(
        s for s in rendered.substitutions_made if s.original_edge_id == "e_boil"
    )
    new_tools = {t.name for t in boil_sub.replacement_edge.tools_required}
    assert "電子レンジ" in new_tools


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
    # Mermaid output should highlight the critical path
    assert "linkStyle" in rendered.mermaid_dag
