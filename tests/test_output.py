"""Tests for the three output formatters."""

from __future__ import annotations

from pathlib import Path

from recipe_optimizer.data_io import load_profile, load_raw_recipe, load_table
from recipe_optimizer.modules.scheduler import schedule
from recipe_optimizer.output import (
    aggregate_tools,
    build_shopping_list,
    to_mermaid,
    to_numbered_steps,
)
from recipe_optimizer.schemas import (
    Constraints,
    GoalNode,
    Ingredient,
    ProcessEdge,
    ProcessType,
    RawRecipe,
    RecipeDAG,
    Schedule,
    ScheduledStep,
    Tool,
    ToolKind,
)

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"
RECIPES_DIR = REPO_ROOT / "data" / "recipes"


# ---------------------------------------------------------------------------
# numbered_list
# ---------------------------------------------------------------------------


def test_numbered_steps_chronological_order():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))

    lines = to_numbered_steps(dag, sched)

    # One line per edge, numbered 1..N
    assert len(lines) == len(dag.edges)
    for i, line in enumerate(lines, start=1):
        assert line.startswith(f"{i}.")

    # The first line should describe whatever starts at t=0
    first_step = min(sched.steps, key=lambda s: s.start_min)
    first_edge = next(e for e in dag.edges if e.id == first_step.edge_id)
    assert first_edge.description in lines[0]


def test_numbered_steps_includes_duration():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    lines = to_numbered_steps(dag, sched)

    # Each line should embed a duration string ending in 分 or 秒
    assert all(("分" in line) or ("秒" in line) for line in lines)


def test_numbered_steps_marks_parallel():
    """An edge that overlaps with another in time gets a (並行可能) suffix."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="raw1", description="raw1"),
            GoalNode(id="raw2", description="raw2"),
            GoalNode(id="cooked1", description="cooked1"),
            GoalNode(id="cooked2", description="cooked2"),
            GoalNode(id="done", description="done", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["raw1"],
                to_node="cooked1",
                action=ProcessType.SIMMER,
                description="長い煮込み",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
                duration_min=10.0,
                attentive_min=0.5,
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["raw2"],
                to_node="cooked2",
                action=ProcessType.CHOP,
                description="切る",
                duration_min=2.0,
                attentive_min=2.0,
            ),
            ProcessEdge(
                id="ef",
                from_nodes=["cooked1", "cooked2"],
                to_node="done",
                action=ProcessType.MIX,
                description="混ぜる",
                duration_min=0.5,
                attentive_min=0.5,
            ),
        ],
        final_node_id="done",
    )
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    lines = to_numbered_steps(dag, sched)
    # The chop runs during the simmer's non-attentive tail → marked parallel
    assert any("並行可能" in line for line in lines)


# ---------------------------------------------------------------------------
# Mermaid (dag_viz)
# ---------------------------------------------------------------------------


def test_mermaid_starts_with_flowchart_header():
    out = to_mermaid(make_oyakodon_dag())
    assert out.splitlines()[0] == "flowchart TD"


def test_mermaid_includes_every_node():
    dag = make_oyakodon_dag()
    out = to_mermaid(dag)
    for node in dag.nodes:
        # Each node id should appear at least once (definition)
        assert node.id in out


def test_mermaid_marks_final_node_with_class():
    dag = make_oyakodon_dag()
    out = to_mermaid(dag)
    assert ":::finalNode" in out
    assert "classDef finalNode" in out


def test_mermaid_emphasizes_critical_path_when_schedule_provided():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    out = to_mermaid(dag, schedule=sched)
    assert "linkStyle" in out
    assert "stroke:#d40" in out


def test_mermaid_no_critical_styling_without_schedule():
    out = to_mermaid(make_oyakodon_dag())
    assert "linkStyle" not in out


def test_mermaid_escapes_inner_double_quotes():
    """Node descriptions with quotes should not break the mermaid parser."""
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description='文字 "with" 引用'),
            GoalNode(id="b", description="完成", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.MIX,
                description="r",
            )
        ],
        final_node_id="b",
    )
    out = to_mermaid(dag)
    # The inner " in 'with' should have been replaced with single quotes
    assert '"with"' not in out
    assert "'with'" in out


# ---------------------------------------------------------------------------
# Shopping list
# ---------------------------------------------------------------------------


def test_aggregate_tools_deduplicates_and_sorts_by_kind():
    dag = RecipeDAG(
        title="t",
        servings=1,
        nodes=[
            GoalNode(id="a", description=""),
            GoalNode(id="b", description=""),
            GoalNode(id="c", description="", is_final=True),
        ],
        edges=[
            ProcessEdge(
                id="e1",
                from_nodes=["a"],
                to_node="b",
                action=ProcessType.SIMMER,
                description="x",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),
                    Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE),
                ],
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["b"],
                to_node="c",
                action=ProcessType.MIX,
                description="y",
                tools_required=[
                    Tool(name="片手鍋", kind=ToolKind.CONTAINER),  # dup
                    Tool(name="菜箸", kind=ToolKind.UTENSIL),
                ],
            ),
        ],
        final_node_id="c",
    )
    tools = aggregate_tools(dag)
    names = [t.name for t in tools]
    assert names.count("片手鍋") == 1  # deduplicated
    # Heat source first, then container, then utensil
    assert names.index("コンロ口") < names.index("片手鍋") < names.index("菜箸")


def test_build_shopping_list_with_raw_recipe():
    dag = make_oyakodon_dag()
    raw = load_raw_recipe(RECIPES_DIR / "oyakodon.json")
    sl = build_shopping_list(dag, raw_recipe=raw)

    # ingredients_text appears verbatim as ingredient names
    names = [i.name for i in sl.ingredients]
    assert "鶏肉（もも肉） 1/2枚" in names
    assert "卵 2個" in names
    assert "玉ねぎ 1/4個" in names

    # Tools come from the DAG
    assert len(sl.tools_needed) > 0


def test_build_shopping_list_without_raw_recipe_yields_empty_ingredients():
    dag = make_oyakodon_dag()
    sl = build_shopping_list(dag, raw_recipe=None)
    assert sl.ingredients == []
    assert len(sl.tools_needed) > 0
