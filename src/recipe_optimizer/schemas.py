"""Pydantic data models that define module boundaries.

The data layer is centred on **explicit resource allocation**:

- ``Resource`` is something the user owns with identity (a specific
  burner, a specific pot). Resources are organised in kind-based pools
  on ``UserProfile``.
- ``ResourceRequirement`` is an edge's reservation on a resource slot
  for a portion of its execution. Edges that simmer for 2 minutes but
  only need the cook's attention for 30 seconds say so via two
  requirements with different ``hold_duration_min`` values.
- Critical path, scheduler, checker, and proposer all read the same
  ``resource_uses`` list — no implicit per-kind logic is sprinkled
  across modules anymore.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Resource model
# ---------------------------------------------------------------------------


class ResourceKind(str, Enum):
    """Coarse pool the resource belongs to.

    Each pool is managed independently by the scheduler.
    """

    COOK = "cook"
    BURNER = "burner"  # コンロ・IH（実体は1口ずつ独立）
    WORKSTATION = "workstation"  # 包丁作業台
    CONTAINER = "container"  # 鍋・フライパン・ボウル
    APPLIANCE = "appliance"  # 電子レンジ・オーブン・炊飯器
    UTENSIL = "utensil"  # 包丁・菜箸（PoC ではプール非考慮）


class Resource(BaseModel):
    """A finite, identity-bearing resource owned by the user."""

    id: str  # globally unique within a profile
    kind: ResourceKind
    name: str  # human-readable (片手鍋, コンロ口1, 電子レンジ, ...)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ResourceRequirement(BaseModel):
    """A reservation on one resource slot during an edge's execution.

    ``hold_duration_min`` is how long this slot stays busy; it must not
    exceed the parent edge's ``duration_min``. ``start_offset_min``
    allows resources to be acquired or released part-way through (e.g.,
    a cook is needed only at the start of a long simmer).
    """

    kind: ResourceKind
    name_hint: str | None = None  # 表記揺れ吸収用（"片手鍋" 等）
    hold_duration_min: float
    start_offset_min: float = 0.0


# ---------------------------------------------------------------------------
# Process taxonomy
# ---------------------------------------------------------------------------


class ProcessType(str, Enum):
    """Canonical action vocabulary keyed in ``ToolUseTable``."""

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
    REST = "rest"
    SERVE = "serve"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Recipe DAG
# ---------------------------------------------------------------------------


class Ingredient(BaseModel):
    name: str
    quantity_text: str = ""  # raw text from the recipe ("1/2枚", "大さじ1")


class GoalNode(BaseModel):
    """A state of ingredients reached at some point in the recipe."""

    id: str
    description: str
    is_final: bool = False


class ProcessEdge(BaseModel):
    """A cooking action transforming input states into a single output state.

    Resource needs are declared in ``resource_uses``. Each entry pins a
    specific kind of resource (and optionally a name hint) busy for a
    portion of the edge's wall-clock duration.
    """

    id: str
    from_nodes: list[str] = Field(default_factory=list)
    to_node: str

    action: ProcessType
    description: str

    duration_min: float = 1.0
    resource_uses: list[ResourceRequirement] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resource_uses_fit(self) -> "ProcessEdge":
        eps = 1e-6
        for use in self.resource_uses:
            end = use.start_offset_min + use.hold_duration_min
            if end > self.duration_min + eps:
                raise ValueError(
                    f"Edge {self.id!r}: resource_use ({use.kind.value}) ends at "
                    f"{end} min, exceeding edge duration_min={self.duration_min}"
                )
            if use.start_offset_min < -eps:
                raise ValueError(
                    f"Edge {self.id!r}: negative start_offset_min on {use.kind.value}"
                )
            if use.hold_duration_min < -eps:
                raise ValueError(
                    f"Edge {self.id!r}: negative hold_duration_min on {use.kind.value}"
                )
        return self


class RecipeDAG(BaseModel):
    title: str
    servings: int = 1
    nodes: list[GoalNode]
    edges: list[ProcessEdge]
    final_node_id: str
    raw_text: str = ""

    @model_validator(mode="after")
    def _validate_dag_integrity(self) -> "RecipeDAG":
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
                        f"Edge {edge.id!r}: from_node '{from_id}' not found"
                    )
            if edge.to_node not in node_ids:
                raise ValueError(
                    f"Edge {edge.id!r}: to_node '{edge.to_node}' not found"
                )
        return self


class RawRecipe(BaseModel):
    """Unparsed recipe loaded from JSON."""

    id: str
    title: str
    servings: int = 1
    source_url: str = ""
    source_attribution: str = ""
    ingredients_text: list[str]
    steps_text: list[str]
    tips: str = ""


# ---------------------------------------------------------------------------
# User profile (resource pools)
# ---------------------------------------------------------------------------


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class UserProfile(BaseModel):
    """Static constraints captured at onboarding.

    Resources live in kind-keyed pools so the scheduler can take pool
    length as capacity and individual ``Resource`` objects as slots.
    """

    user_id: str
    cooks: list[Resource] = Field(default_factory=list)
    burners: list[Resource] = Field(default_factory=list)
    workstations: list[Resource] = Field(default_factory=list)
    containers: list[Resource] = Field(default_factory=list)
    appliances: list[Resource] = Field(default_factory=list)
    utensils: list[Resource] = Field(default_factory=list)
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    skill_factors: dict[str, float] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)

    def pool(self, kind: ResourceKind) -> list[Resource]:
        return {
            ResourceKind.COOK: self.cooks,
            ResourceKind.BURNER: self.burners,
            ResourceKind.WORKSTATION: self.workstations,
            ResourceKind.CONTAINER: self.containers,
            ResourceKind.APPLIANCE: self.appliances,
            ResourceKind.UTENSIL: self.utensils,
        }[kind]


class SessionState(BaseModel):
    """Dynamic state — placeholder for the live cooking mode."""

    ingredients_available: list[str] = Field(default_factory=list)
    busy_resource_ids: list[str] = Field(default_factory=list)
    completed_edge_ids: list[str] = Field(default_factory=list)
    elapsed_min: float = 0.0


class Constraints(BaseModel):
    profile: UserProfile
    session: SessionState = Field(default_factory=SessionState)


# ---------------------------------------------------------------------------
# Tool substitution table
# ---------------------------------------------------------------------------


class ResourceSpec(BaseModel):
    """Relative-duration resource pattern stored in a ``ToolOption``.

    ``relative_duration`` is the fraction of the option's total duration
    that this resource stays busy. 1.0 means "for the whole edge".
    Held separately from concrete ``ResourceRequirement`` so that the
    same pattern can be applied to recipes with different total
    durations.
    """

    kind: ResourceKind
    name_hint: str | None = None
    relative_duration: float = 1.0


class ToolOption(BaseModel):
    """One alternative way to perform a given ProcessType."""

    label: str
    resources: list[ResourceSpec]
    time_factor: float = 1.0  # multiplier on baseline duration
    quality_factor: float = 1.0
    constraints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = "seed"  # seed | llm | user_feedback


class ToolUseTable(BaseModel):
    entries: dict[str, list[ToolOption]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Violations / substitution
# ---------------------------------------------------------------------------


class ConstraintViolation(BaseModel):
    edge_id: str
    reason: str  # "missing_resource"
    missing_kind: ResourceKind
    missing_name_hint: str | None = None


class SubstitutionCandidate(BaseModel):
    original_edge_id: str
    replacement_edge: ProcessEdge
    rationale: str
    quality_delta: float = 0.0
    time_delta_min: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class ScheduledStep(BaseModel):
    edge_id: str
    start_min: float
    end_min: float
    assigned_resource_ids: list[str] = Field(default_factory=list)
    parallel_with: list[str] = Field(default_factory=list)


class Schedule(BaseModel):
    steps: list[ScheduledStep]
    total_duration_min: float
    critical_path_edge_ids: list[str]


# ---------------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------------


class ShoppingList(BaseModel):
    ingredients: list[Ingredient] = Field(default_factory=list)
    resources_needed: list[Resource] = Field(default_factory=list)


class RenderedRecipe(BaseModel):
    title: str
    numbered_steps: list[str]
    mermaid_dag: str
    shopping_list: ShoppingList
    schedule: Schedule
    optimized_dag: "RecipeDAG | None" = None
    substitutions_made: list[SubstitutionCandidate] = Field(default_factory=list)
