"""Hand-crafted RecipeDAG for the 麦茶 (barley tea) example.

This is the canonical case discussed in the design conversation:
boiling water "with やかん" — when the user lacks やかん, the
substitution layer must replace this with 鍋 (or 電子レンジ).

Used by checker tests to verify violation detection, and by later
proposer/rewriter tests as the input that triggers substitution.
"""

from __future__ import annotations

from recipe_optimizer.schemas import (
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    Tool,
    ToolKind,
)


def make_mugicha_dag() -> RecipeDAG:
    yakan = Tool(name="やかん", kind=ToolKind.CONTAINER)
    burner = Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE)

    nodes = [
        GoalNode(id="n_water", description="水 1L"),
        GoalNode(id="n_barley_pack", description="麦茶パック 1個"),
        GoalNode(id="n_boiling_water", description="沸騰した湯"),
        GoalNode(id="n_brewed", description="完成した麦茶", is_final=True),
    ]

    edges = [
        ProcessEdge(
            id="e_boil",
            from_nodes=["n_water"],
            to_node="n_boiling_water",
            action=ProcessType.BOIL_WATER,
            description="やかんで湯を沸かす",
            tools_required=[yakan, burner],
            duration_min=5.0,
            attentive_min=1.0,
            parameters={"volume_ml": 1000},
        ),
        ProcessEdge(
            id="e_brew",
            from_nodes=["n_boiling_water", "n_barley_pack"],
            to_node="n_brewed",
            action=ProcessType.MIX,
            description="やかんに麦茶パックを入れ抽出する",
            tools_required=[yakan],
            duration_min=10.0,
            attentive_min=0.5,
        ),
    ]

    return RecipeDAG(
        title="麦茶",
        servings=4,
        nodes=nodes,
        edges=edges,
        final_node_id="n_brewed",
    )
