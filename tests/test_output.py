"""Tests for the three output formatters (resource model)."""

from __future__ import annotations

from pathlib import Path

from recipe_optimizer.data_io import load_profile, load_raw_recipe
from recipe_optimizer.modules.scheduler import schedule
from recipe_optimizer.output import (
    aggregate_resources,
    build_shopping_list,
    to_mermaid,
    to_numbered_steps,
)
from recipe_optimizer.schemas import (
    Constraints,
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    ResourceKind,
    ResourceRequirement,
)

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
RECIPES_DIR = REPO_ROOT / "data" / "recipes"


# ---------------------------------------------------------------------------
# numbered_list
# ---------------------------------------------------------------------------


def test_numbered_steps_chronological_order():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    lines = to_numbered_steps(dag, sched)

    assert len(lines) == len(dag.edges)
    for i, line in enumerate(lines, start=1):
        assert line.startswith(f"{i}.")


def test_numbered_steps_includes_duration():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    lines = to_numbered_steps(dag, sched)
    assert all(("分" in line) or ("秒" in line) for line in lines)


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def test_mermaid_starts_with_flowchart_header():
    out = to_mermaid(make_oyakodon_dag())
    assert out.splitlines()[0] == "flowchart TD"


def test_mermaid_includes_every_node():
    dag = make_oyakodon_dag()
    out = to_mermaid(dag)
    for node in dag.nodes:
        assert node.id in out


def test_mermaid_marks_final_node():
    out = to_mermaid(make_oyakodon_dag())
    assert ":::finalNode" in out


def test_mermaid_emphasises_critical_path_when_schedule_given():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    out = to_mermaid(dag, schedule=sched)
    assert "linkStyle" in out


# ---------------------------------------------------------------------------
# Shopping list
# ---------------------------------------------------------------------------


def test_aggregate_resources_excludes_cook_workstation():
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
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(
                        kind=ResourceKind.COOK, hold_duration_min=1.0
                    ),
                    ResourceRequirement(
                        kind=ResourceKind.WORKSTATION, hold_duration_min=1.0
                    ),
                    ResourceRequirement(
                        kind=ResourceKind.BURNER, hold_duration_min=1.0
                    ),
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="片手鍋",
                        hold_duration_min=1.0,
                    ),
                ],
            ),
            ProcessEdge(
                id="e2",
                from_nodes=["b"],
                to_node="c",
                action=ProcessType.MIX,
                description="y",
                duration_min=1.0,
                resource_uses=[
                    ResourceRequirement(
                        kind=ResourceKind.CONTAINER,
                        name_hint="片手鍋",
                        hold_duration_min=1.0,
                    )  # duplicate name should dedupe
                ],
            ),
        ],
        final_node_id="c",
    )
    out = aggregate_resources(dag)
    kinds = [r.kind for r in out]
    assert ResourceKind.COOK not in kinds
    assert ResourceKind.WORKSTATION not in kinds
    names = [r.name for r in out]
    assert names.count("片手鍋") == 1


def test_build_shopping_list_with_raw_recipe():
    dag = make_oyakodon_dag()
    raw = load_raw_recipe(RECIPES_DIR / "oyakodon.json")
    sl = build_shopping_list(dag, raw_recipe=raw)

    names = [i.name for i in sl.ingredients]
    assert "鶏肉（もも肉） 1/2枚" in names

    assert len(sl.resources_needed) > 0


def test_build_shopping_list_without_raw_recipe_has_empty_ingredients():
    sl = build_shopping_list(make_oyakodon_dag(), raw_recipe=None)
    assert sl.ingredients == []
    assert len(sl.resources_needed) > 0
