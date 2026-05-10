"""Hand-crafted RecipeDAG for the 親子丼 sample.

Used by tests as the canned response from MockLLMClient(parser, ...).
The structure mirrors what we expect a real LLM to produce given the
data/recipes/oyakodon.json input — we keep it consistent so downstream
modules can be tested deterministically.
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


def make_oyakodon_dag() -> RecipeDAG:
    knife_board = Tool(name="包丁+まな板", kind=ToolKind.UTENSIL)
    bowl_chopstick = Tool(name="ボウル+菜箸", kind=ToolKind.CONTAINER)
    pot = Tool(name="片手鍋", kind=ToolKind.CONTAINER)
    burner = Tool(name="コンロ口", kind=ToolKind.HEAT_SOURCE)

    nodes = [
        # roots: raw ingredients / pre-mixed seasoning
        GoalNode(id="n_chicken_raw", description="生の鶏もも肉 1/2枚"),
        GoalNode(id="n_onion_raw", description="生の玉ねぎ 1/4個"),
        GoalNode(id="n_eggs_raw", description="生卵 2個"),
        GoalNode(id="n_seasoning", description="合わせ調味料 (●しょうゆ/みりん/酒/砂糖/だし/水)"),
        # prepped intermediates
        GoalNode(id="n_chicken_cut", description="一口大に切った鶏肉"),
        GoalNode(id="n_onion_sliced", description="薄切りにした玉ねぎ"),
        GoalNode(id="n_eggs_beaten", description="溶き卵"),
        # cooking states
        GoalNode(id="n_pot_step1", description="鍋に調味料と玉ねぎを入れた状態"),
        GoalNode(id="n_simmered_onion", description="玉ねぎが煮えた状態"),
        GoalNode(id="n_with_chicken", description="鶏肉を加えた鍋"),
        GoalNode(id="n_chicken_done", description="鶏肉に火が通った煮汁"),
        GoalNode(id="n_first_egg", description="溶き卵2/3を加え混ぜた状態"),
        GoalNode(id="n_lidded", description="蓋をして30秒煮た状態"),
        GoalNode(id="n_almost_done", description="残りの卵を加えた半熟状態"),
        # final
        GoalNode(id="n_done", description="完成した親子丼", is_final=True),
    ]

    edges = [
        ProcessEdge(
            id="e_slice_onion",
            from_nodes=["n_onion_raw"],
            to_node="n_onion_sliced",
            action=ProcessType.SLICE,
            description="玉ねぎを薄切りにする",
            tools_required=[knife_board],
            duration_min=2.0,
            attentive_min=2.0,
        ),
        ProcessEdge(
            id="e_chop_chicken",
            from_nodes=["n_chicken_raw"],
            to_node="n_chicken_cut",
            action=ProcessType.CHOP,
            description="鶏肉を一口大に切る",
            tools_required=[knife_board],
            duration_min=2.0,
            attentive_min=2.0,
        ),
        ProcessEdge(
            id="e_beat_eggs",
            from_nodes=["n_eggs_raw"],
            to_node="n_eggs_beaten",
            action=ProcessType.BEAT,
            description="卵を溶く",
            tools_required=[bowl_chopstick],
            duration_min=1.0,
            attentive_min=1.0,
        ),
        ProcessEdge(
            id="e_combine_step1",
            from_nodes=["n_seasoning", "n_onion_sliced"],
            to_node="n_pot_step1",
            action=ProcessType.MIX,
            description="鍋に●の調味料と玉ねぎを入れる",
            tools_required=[pot],
            duration_min=0.5,
            attentive_min=0.5,
        ),
        ProcessEdge(
            id="e_simmer_onion",
            from_nodes=["n_pot_step1"],
            to_node="n_simmered_onion",
            action=ProcessType.SIMMER,
            description="強めの中火で2分煮る",
            tools_required=[pot, burner],
            duration_min=2.0,
            attentive_min=0.5,
            parameters={"heat_level": "medium-high"},
        ),
        ProcessEdge(
            id="e_add_chicken",
            from_nodes=["n_simmered_onion", "n_chicken_cut"],
            to_node="n_with_chicken",
            action=ProcessType.MIX,
            description="切った鶏肉を加える",
            tools_required=[pot],
            duration_min=0.3,
            attentive_min=0.3,
        ),
        ProcessEdge(
            id="e_simmer_chicken",
            from_nodes=["n_with_chicken"],
            to_node="n_chicken_done",
            action=ProcessType.SIMMER,
            description="中火で3分煮て鶏肉に火を通す",
            tools_required=[pot, burner],
            duration_min=3.0,
            attentive_min=0.5,
            parameters={"heat_level": "medium"},
        ),
        ProcessEdge(
            id="e_add_first_egg",
            from_nodes=["n_chicken_done", "n_eggs_beaten"],
            to_node="n_first_egg",
            action=ProcessType.MIX,
            description="溶き卵2/3を加え菜箸で全体に混ぜる",
            tools_required=[pot, Tool(name="菜箸", kind=ToolKind.UTENSIL)],
            duration_min=0.3,
            attentive_min=0.3,
        ),
        ProcessEdge(
            id="e_lid",
            from_nodes=["n_first_egg"],
            to_node="n_lidded",
            action=ProcessType.SIMMER,
            description="蓋をして中火で30秒",
            tools_required=[pot, burner],
            duration_min=0.5,
            attentive_min=0.5,
            parameters={"requires_lid": True, "heat_level": "medium"},
        ),
        ProcessEdge(
            id="e_finish_egg",
            from_nodes=["n_lidded"],
            to_node="n_almost_done",
            action=ProcessType.SIMMER,
            description="残りの溶き卵を加えて半熟になるまで煮る",
            tools_required=[pot, burner],
            duration_min=1.0,
            attentive_min=1.0,
            parameters={"heat_level": "medium"},
        ),
        ProcessEdge(
            id="e_serve",
            from_nodes=["n_almost_done"],
            to_node="n_done",
            action=ProcessType.SERVE,
            description="器に盛って出来上がり",
            tools_required=[],
            duration_min=0.3,
            attentive_min=0.3,
        ),
    ]

    return RecipeDAG(
        title="親子丼",
        servings=1,
        nodes=nodes,
        edges=edges,
        final_node_id="n_done",
    )
