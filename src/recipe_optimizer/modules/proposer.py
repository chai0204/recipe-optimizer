"""Proposer: generate substitution candidates for an edge.

Each violation refers to one ``ResourceRequirement`` on an edge that
the user cannot satisfy. The proposer looks up alternative
``ToolOption`` patterns for the edge's ``action`` and rebuilds the
edge's ``resource_uses`` with the chosen option's pattern, scaled by
its ``time_factor``.

Two paths:

1. **Table lookup** (deterministic): all alternatives in
   ``tool_use_table`` are tried, kept only if the user can satisfy
   every spec inside.

2. **LLM fallback** (only when no table hit): the LLM is asked to
   propose new options; valid ones (with all specs satisfiable) are
   written back to the table for future deterministic lookups.

One candidate per option (capped at MAX_CANDIDATES). The candidate's
``replacement_edge`` preserves the original edge's id/from_nodes/
to_node/action — only ``description``, ``duration_min``, and
``resource_uses`` change.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..data_io.tool_use_table import (
    _matches_name_hint,
    add_entry,
    find_compatible,
    is_option_compatible,
)
from ..llm import LLMClient
from ..schemas import (
    ConstraintViolation,
    ProcessEdge,
    ResourceRequirement,
    ResourceSpec,
    SubstitutionCandidate,
    ToolOption,
    ToolUseTable,
    UserProfile,
)

PROPOSER_LABEL = "proposer"
MAX_CANDIDATES = 3


class LLMProposal(BaseModel):
    """LLM output schema."""

    candidates: list[ToolOption] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You are a cooking expert. A recipe step requires a resource the user does
not have. Propose 1-3 alternative ways to perform the same action using
only resource kinds that the user has access to.

Each candidate is a ``ToolOption`` consisting of:
- label: short Japanese description (e.g., "片手鍋+コンロ")
- resources: list of {kind, name_hint, relative_duration} where:
    kind ∈ {cook, burner, workstation, container, appliance, utensil}
    name_hint: specific resource name if relevant (e.g., "片手鍋")
    relative_duration: 0.0–1.0 — what fraction of the edge's total duration
      this resource is busy. cook tends to be fraction-only for simmering.
- time_factor: multiplier on the recipe's baseline duration_min (1.0 = same)
- quality_factor: relative quality (1.0 = same, 0.9 = slightly worse)

Output JSON only: {"candidates": [...]}.
"""


def _build_prompt(
    edge: ProcessEdge,
    violation: ConstraintViolation,
    profile: UserProfile,
) -> str:
    owned_summary = "\n".join(
        f"- {kind}: "
        + ", ".join(r.name for r in pool)
        for kind, pool in [
            ("cook", profile.cooks),
            ("burner", profile.burners),
            ("workstation", profile.workstations),
            ("container", profile.containers),
            ("appliance", profile.appliances),
            ("utensil", profile.utensils),
        ]
        if pool
    )
    current_uses = ", ".join(
        f"{u.kind.value}({u.name_hint or '*'},{u.hold_duration_min}m)"
        for u in edge.resource_uses
    )
    missing_str = f"{violation.missing_kind.value}/{violation.missing_name_hint or '*'}"
    return f"""\
Recipe step: {edge.description}
Action: {edge.action.value}
Current resource uses: {current_uses}
Missing resource (user lacks): {missing_str}
Original duration: {edge.duration_min} min

User's available resources:
{owned_summary}

Propose 1-3 alternative ToolOptions using only resources the user has."""


# ---------------------------------------------------------------------------
# Replacement-edge construction
# ---------------------------------------------------------------------------


def _spec_to_requirement(
    spec: ResourceSpec, new_duration: float
) -> ResourceRequirement:
    hold = new_duration * spec.relative_duration
    return ResourceRequirement(
        kind=spec.kind,
        name_hint=spec.name_hint,
        hold_duration_min=hold,
        start_offset_min=0.0,
    )


def _build_replacement_edge(
    original: ProcessEdge,
    option: ToolOption,
) -> ProcessEdge:
    new_duration = original.duration_min * option.time_factor
    new_uses = [_spec_to_requirement(spec, new_duration) for spec in option.resources]

    new_description = original.description
    # Replace any explicit ``やかん`` etc. in the description with the
    # primary container name from the new option, if obvious.
    if option.label:
        new_description = f"{original.description.split('（')[0]}（{option.label}で代替）"

    return original.model_copy(
        update={
            "duration_min": new_duration,
            "resource_uses": new_uses,
            "description": new_description,
        }
    )


CONTINUITY_QUALITY_BONUS = 0.25
"""Per-shared-name bonus applied to ``quality_delta`` when a candidate
reuses a container name that an upstream edge already occupies. This
encodes the strong "use the same pot for the next step" preference
that home cooking has by default — moving food across containers
mid-recipe is unusual unless there is a specific reason. The bonus is
sized to outrun typical option-level perks (e.g. a whisk being 5%
better quality + 20% faster than chopsticks) so that continuity wins
absent a clearly superior alternative."""


def _continuity_bonus(
    option: ToolOption,
    preferred_resource_names: set[str] | None,
) -> float:
    """Bonus for each option spec whose ``name_hint`` matches a name in
    ``preferred_resource_names`` under the same fuzzy policy used by
    ``is_tool_owned`` (exact or one-way substring containment). The
    fuzziness lets an option labelled "鍋で混ぜる" with ``name_hint="鍋"``
    still continue from an upstream edge that committed to "片手鍋"."""
    if not preferred_resource_names:
        return 0.0
    matches = 0
    for spec in option.resources:
        if not spec.name_hint:
            continue
        if any(
            _matches_name_hint(pref, spec.name_hint)
            for pref in preferred_resource_names
        ):
            matches += 1
    return CONTINUITY_QUALITY_BONUS * matches


def _option_to_candidate(
    option: ToolOption,
    edge: ProcessEdge,
    violation: ConstraintViolation,
    preferred_resource_names: set[str] | None = None,
) -> SubstitutionCandidate:
    replacement = _build_replacement_edge(edge, option)
    time_delta = replacement.duration_min - edge.duration_min
    bonus = _continuity_bonus(option, preferred_resource_names)
    quality_delta = (option.quality_factor - 1.0) + bonus

    sign_t = "+" if time_delta >= 0 else ""
    sign_q = "+" if quality_delta >= 0 else ""
    missing_repr = (
        f"{violation.missing_kind.value}/{violation.missing_name_hint or '*'}"
    )
    continuity_note = (
        f"、容器継続ボーナス +{bonus:.2f}" if bonus > 0 else ""
    )
    rationale = (
        f"{missing_repr} の代替として {option.label} を使用"
        f"（時間 {sign_t}{time_delta:.1f}分、品質 {sign_q}{quality_delta:.2f}"
        f"{continuity_note}）"
    )

    return SubstitutionCandidate(
        original_edge_id=edge.id,
        replacement_edge=replacement,
        rationale=rationale,
        quality_delta=quality_delta,
        time_delta_min=time_delta,
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class RecipeProposer:
    """Generate substitution candidates. Mutates the table on LLM discovery."""

    def __init__(
        self,
        client: LLMClient,
        table: ToolUseTable,
        max_retries: int = 1,
    ) -> None:
        self.client = client
        self.table = table
        self.max_retries = max_retries

    def propose(
        self,
        violation: ConstraintViolation,
        edge: ProcessEdge,
        profile: UserProfile,
        preferred_resource_names: set[str] | None = None,
    ) -> list[SubstitutionCandidate]:
        """Generate substitution candidates for one violation.

        ``preferred_resource_names`` lets the pipeline hint at containers
        that upstream edges have already committed to (e.g. the boiling
        pot continues into the steeping step). Candidates whose resource
        ``name_hint`` overlaps the preferred set receive a small quality
        bonus, breaking ties toward continuity.
        """
        owned = (
            profile.cooks
            + profile.burners
            + profile.workstations
            + profile.containers
            + profile.appliances
            + profile.utensils
        )

        # 1. Table lookup
        existing = find_compatible(self.table, edge.action, owned)
        if existing:
            return [
                _option_to_candidate(opt, edge, violation, preferred_resource_names)
                for opt in existing[:MAX_CANDIDATES]
            ]

        # 2. LLM fallback
        proposal = self._ask_llm(edge, violation, profile)

        candidates: list[SubstitutionCandidate] = []
        for opt in proposal.candidates:
            if not is_option_compatible(opt, owned):
                continue
            tagged = opt.model_copy(
                update={"source": opt.source if opt.source != "seed" else "llm"}
            )
            add_entry(self.table, edge.action, tagged)
            candidates.append(
                _option_to_candidate(tagged, edge, violation, preferred_resource_names)
            )

        return candidates[:MAX_CANDIDATES]

    def _ask_llm(
        self,
        edge: ProcessEdge,
        violation: ConstraintViolation,
        profile: UserProfile,
    ) -> LLMProposal:
        return self.client.generate_structured(
            prompt=_build_prompt(edge, violation, profile),
            output_schema=LLMProposal,
            system=SYSTEM_PROMPT,
            max_retries=self.max_retries,
            label=PROPOSER_LABEL,
        )
