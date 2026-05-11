"""Hand-crafted RecipeDAG for 麦茶 under the explicit resource model.

The canonical "やかんがない人向け" substitution case: the boil edge
demands ``container/やかん`` which the demo profile lacks, forcing the
proposer to swap in 片手鍋 / 両手鍋 / 電子レンジ from the table.
"""

from __future__ import annotations

from recipe_optimizer.schemas import (
    GoalNode,
    ProcessEdge,
    ProcessType,
    RecipeDAG,
    ResourceKind,
    ResourceRequirement,
)


def _req(
    kind: ResourceKind,
    hold: float,
    *,
    name_hint: str | None = None,
) -> ResourceRequirement:
    return ResourceRequirement(
        kind=kind, name_hint=name_hint, hold_duration_min=hold
    )


def make_mugicha_dag() -> RecipeDAG:
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
            duration_min=5.0,
            resource_uses=[
                _req(ResourceKind.COOK, 1.0),  # 火加減確認だけ最初の1分
                _req(ResourceKind.BURNER, 5.0),
                _req(ResourceKind.CONTAINER, 5.0, name_hint="やかん"),
            ],
            parameters={"volume_ml": 1000},
        ),
        ProcessEdge(
            id="e_brew",
            from_nodes=["n_boiling_water", "n_barley_pack"],
            to_node="n_brewed",
            action=ProcessType.MIX,
            description="やかんに麦茶パックを入れ抽出する",
            duration_min=10.0,
            resource_uses=[
                _req(ResourceKind.COOK, 0.5),  # 入れて蓋する程度
                _req(ResourceKind.CONTAINER, 10.0, name_hint="やかん"),
            ],
        ),
    ]

    return RecipeDAG(
        title="麦茶",
        servings=4,
        nodes=nodes,
        edges=edges,
        final_node_id="n_brewed",
    )
