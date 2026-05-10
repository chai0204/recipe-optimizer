"""Pydantic data models that define module boundaries.

Every LLM/algorithm module accepts and returns these typed objects.
This is the central discipline that controls LLM output variance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Quantity(BaseModel):
    """A measured amount of an ingredient."""

    amount: float | None = None  # None means "to taste" / "適量"
    unit: str = ""  # "g", "ml", "個", "大さじ", ...
    note: str = ""  # free-form qualifier ("好みで")


class Ingredient(BaseModel):
    name: str
    quantity: Quantity = Field(default_factory=Quantity)
    prep_state: str = "raw"  # raw | washed | cut | cooked | ...


class ToolKind(str, Enum):
    """Coarse classification used by the substitution layer."""

    HEAT_SOURCE = "heat_source"  # コンロ口、IH、電子レンジ、オーブン
    CONTAINER = "container"  # 鍋、フライパン、ボウル、ケトル
    UTENSIL = "utensil"  # 包丁、菜箸、おたま
    APPLIANCE = "appliance"  # 炊飯器、ミキサー、トースター


class Tool(BaseModel):
    name: str
    kind: ToolKind
    attributes: dict[str, Any] = Field(default_factory=dict)
    # 例: {"capacity_ml": 2000, "has_lid": true, "burners": 2}


# ---------------------------------------------------------------------------
# Process taxonomy
# ---------------------------------------------------------------------------


class ProcessType(str, Enum):
    """Canonical action vocabulary. The tool_use_table is keyed on these.

    The parser maps natural language verbs to these. New types are added
    when LLM proposer discovers them and we promote them to the canonical set.
    """

    # 切る系
    CHOP = "chop"
    SLICE = "slice"
    MINCE = "mince"
    PEEL = "peel"
    # 加熱系
    BOIL = "boil"
    BOIL_WATER = "boil_water"
    SIMMER = "simmer"
    SAUTE = "saute"
    FRY = "fry"
    BAKE = "bake"
    STEAM = "steam"
    GRILL = "grill"
    MICROWAVE = "microwave"
    # 混合系
    MIX = "mix"
    BEAT = "beat"
    KNEAD = "knead"
    # その他
    REST = "rest"  # 待つ・冷ます
    SERVE = "serve"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Recipe DAG
# ---------------------------------------------------------------------------


class GoalNode(BaseModel):
    """A goal state: an intermediate or final result.

    Examples:
        - "切られた玉ねぎ"
        - "沸騰した湯500ml"
        - "完成した親子丼"
    """

    id: str
    description: str
    ingredients_present: list[Ingredient] = Field(default_factory=list)
    is_final: bool = False


class ProcessEdge(BaseModel):
    """A process: an action that transforms input nodes into an output node.

    The directed-acyclic structure is stored in networkx; this object is
    the edge attribute payload.
    """

    id: str
    from_nodes: list[str] = Field(default_factory=list)
    to_node: str

    action: ProcessType
    description: str  # human-readable step text

    tools_required: list[Tool] = Field(default_factory=list)
    duration_min: float = 1.0  # total wall time
    attentive_min: float = 1.0  # time the cook cannot leave (<= duration_min)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # 例: {"heat_level": "medium", "temperature_c": 180, "volume_ml": 500}


class RecipeDAG(BaseModel):
    """Graph representation. The actual networkx.DiGraph lives in a wrapper class."""

    title: str
    servings: int = 1
    nodes: list[GoalNode]
    edges: list[ProcessEdge]
    final_node_id: str
    raw_text: str = ""  # original recipe text for traceability

    @model_validator(mode="after")
    def _validate_dag_integrity(self) -> "RecipeDAG":
        """Cross-field referential integrity.

        - Node IDs must be unique
        - Every edge.from_nodes / edge.to_node must reference an existing node
        - final_node_id must be one of the nodes
        - Edge IDs must be unique

        Raising here causes ``LLMClient.generate_structured`` to retry the
        LLM call with the validation error in the feedback turn.
        """
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs in DAG")

        edge_ids = {e.id for e in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("Duplicate edge IDs in DAG")

        if self.final_node_id not in node_ids:
            raise ValueError(
                f"final_node_id '{self.final_node_id}' not found in nodes"
            )

        for edge in self.edges:
            for from_id in edge.from_nodes:
                if from_id not in node_ids:
                    raise ValueError(
                        f"Edge {edge.id!r}: from_node '{from_id}' not found in nodes"
                    )
            if edge.to_node not in node_ids:
                raise ValueError(
                    f"Edge {edge.id!r}: to_node '{edge.to_node}' not found in nodes"
                )
            if edge.attentive_min > edge.duration_min:
                raise ValueError(
                    f"Edge {edge.id!r}: attentive_min ({edge.attentive_min}) "
                    f"exceeds duration_min ({edge.duration_min})"
                )

        return self


class RawRecipe(BaseModel):
    """Unparsed recipe loaded from JSON (e.g. scraped from Cookpad).

    This is the parser's input. Already lightly structured into ingredients
    and steps lists, so the parser focuses on extracting cooking semantics
    rather than handling free-form prose.
    """

    id: str
    title: str
    servings: int = 1
    source_url: str = ""
    source_attribution: str = ""
    ingredients_text: list[str]
    steps_text: list[str]
    tips: str = ""


# ---------------------------------------------------------------------------
# Constraints (user profile + session state)
# ---------------------------------------------------------------------------


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class UserProfile(BaseModel):
    """Static-ish constraints captured at onboarding and refined over sessions."""

    user_id: str
    tools_owned: list[Tool]
    burners_count: int = 2
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    skill_factors: dict[str, float] = Field(default_factory=dict)
    # 例: {"chop": 1.3, "saute": 1.0}  # 平均比、>1 で遅い
    preferences: dict[str, Any] = Field(default_factory=dict)
    # 例: {"avoid_deep_fry": true, "max_cleanup_items": 5}


class SessionState(BaseModel):
    """Dynamic state at the moment of recipe generation.

    For PoC we only consume the static portion (ingredients_available)
    and assume kitchen is otherwise idle.
    """

    ingredients_available: list[Ingredient] = Field(default_factory=list)
    tools_in_use: list[str] = Field(default_factory=list)  # tool names currently busy
    completed_edge_ids: list[str] = Field(default_factory=list)
    elapsed_min: float = 0.0


class Constraints(BaseModel):
    profile: UserProfile
    session: SessionState = Field(default_factory=SessionState)


# ---------------------------------------------------------------------------
# Tool substitution table
# ---------------------------------------------------------------------------


class ToolOption(BaseModel):
    """One way to perform a given ProcessType."""

    tool: Tool
    time_factor: float = 1.0  # relative to the canonical tool (1.0 = baseline)
    quality_factor: float = 1.0  # 1.0 = no degradation
    constraints: dict[str, Any] = Field(default_factory=dict)
    # 例: {"max_volume_ml": 500, "requires_lid": true}
    confidence: float = 1.0
    source: str = "seed"  # seed | llm | user_feedback


class ToolUseTable(BaseModel):
    """Maps each ProcessType to a ranked list of compatible tools.

    Persisted as JSON. The proposer module updates this when LLM
    discovers new combinations not yet in the table.
    """

    entries: dict[str, list[ToolOption]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Substitution / violation types
# ---------------------------------------------------------------------------


class ConstraintViolation(BaseModel):
    """Marks a process edge that the user cannot execute as-is."""

    edge_id: str
    reason: str  # "missing_tool" | "missing_ingredient" | "burner_unavailable"
    missing: list[str] = Field(default_factory=list)  # tool names or ingredient names


class SubstitutionCandidate(BaseModel):
    """A proposed replacement for a violation."""

    original_edge_id: str
    replacement_edge: ProcessEdge
    rationale: str
    quality_delta: float = 0.0  # negative = degraded
    time_delta_min: float = 0.0
    score: float = 0.0  # selector fills this in


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class ScheduledStep(BaseModel):
    edge_id: str
    start_min: float
    end_min: float
    parallel_with: list[str] = Field(default_factory=list)  # other edge_ids


class Schedule(BaseModel):
    steps: list[ScheduledStep]
    total_duration_min: float
    critical_path_edge_ids: list[str]


# ---------------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------------


class ShoppingList(BaseModel):
    ingredients: list[Ingredient]
    tools_needed: list[Tool]


class RenderedRecipe(BaseModel):
    """Bundle of all three output formats for the PoC."""

    title: str
    numbered_steps: list[str]  # ① 一般的な数字リスト
    mermaid_dag: str  # ② 構造化DAG (mermaid記法)
    shopping_list: ShoppingList  # ③ 使うものリスト
    schedule: Schedule
    substitutions_made: list[SubstitutionCandidate] = Field(default_factory=list)
