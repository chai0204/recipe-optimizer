"""Proposer: generate substitution candidates for violating edges.

Hybrid module that combines two paths:

1. **Table lookup** (fast, deterministic): query ``tool_use_table`` for
   the edge's ``ProcessType``, filter by what the user owns, and return
   each compatible option as a SubstitutionCandidate.

2. **LLM fallback** (slow, exploratory): only invoked when the table
   has no compatible options. The LLM proposes alternatives, which are
   filtered by ownership and **written back to the table** so future
   lookups for the same action become deterministic. This is the data-
   layer learning mechanism described in
   ``knowledge/llm-as-bounded-module.md``.

Output is a list of :class:`SubstitutionCandidate`, each containing a
``replacement_edge`` that preserves DAG topology (same ``id``,
``from_nodes``, ``to_node``, ``action``) but updates ``tools_required``
and durations to reflect the substitute tool's profile.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..data_io.tool_use_table import add_entry, find_compatible, is_tool_owned
from ..llm import LLMClient
from ..schemas import (
    ConstraintViolation,
    ProcessEdge,
    SubstitutionCandidate,
    ToolKind,
    ToolOption,
    ToolUseTable,
    UserProfile,
)

PROPOSER_LABEL = "proposer"
MAX_CANDIDATES = 3


# ---------------------------------------------------------------------------
# LLM I/O schema
# ---------------------------------------------------------------------------


class LLMProposal(BaseModel):
    """LLM output schema: a small set of tool substitution proposals."""

    candidates: list[ToolOption] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You are a cooking expert proposing tool substitutions.

A recipe step requires a tool the user does not have. Propose 1-3
alternative ways to perform the same action using only the user's
available tools.

For each candidate:
- tool: { name (Japanese), kind (heat_source|container|utensil|appliance), attributes }
- time_factor: relative time vs the original tool (1.0 = same, 1.2 = 20% slower)
- quality_factor: relative quality (1.0 = same, 0.9 = slightly worse)
- constraints: optional limits (e.g., {"max_volume_ml": 500})
- confidence: 0.0-1.0

Constraints:
- Prefer the user's existing tools (their exact names if possible).
- Prefer minimal time and quality degradation.
- Output JSON only, conforming to {"candidates": [...]}.
"""


def _build_prompt(
    edge: ProcessEdge,
    violation: ConstraintViolation,
    profile: UserProfile,
) -> str:
    owned = ", ".join(f"{t.name}({t.kind.value})" for t in profile.tools_owned)
    missing = ", ".join(violation.missing) if violation.missing else "(none)"
    return f"""\
Recipe step: {edge.description}
Action: {edge.action.value}
Original tools: {[t.name for t in edge.tools_required]}
Missing tools (user lacks): {missing}
User's available tools: {owned}
Original duration: {edge.duration_min} min, attentive: {edge.attentive_min} min

Propose 1-3 alternative ways to perform this step using only the user's tools."""


# ---------------------------------------------------------------------------
# Replacement-edge construction
# ---------------------------------------------------------------------------


def _build_replacement_edge(
    original: ProcessEdge,
    opt: ToolOption,
    missing_names: list[str],
) -> ProcessEdge:
    """Construct a replacement edge that swaps missing tools for ``opt.tool``.

    Tools listed in ``original.tools_required`` whose name appears in
    ``missing_names`` are dropped; ``opt.tool`` is appended. Other tools
    are preserved by default.

    Heuristic: if the substitute is a self-contained appliance (e.g.
    microwave, electric kettle), external heat sources are dropped from
    the preserved set — using a microwave does not also occupy a burner.
    """
    missing_set = set(missing_names)
    preserved = [t for t in original.tools_required if t.name not in missing_set]

    if opt.tool.kind == ToolKind.APPLIANCE:
        preserved = [t for t in preserved if t.kind != ToolKind.HEAT_SOURCE]

    new_tools = preserved + [opt.tool]
    new_duration = original.duration_min * opt.time_factor
    # Scale attentive proportionally but never above the new total duration
    new_attentive = min(original.attentive_min * opt.time_factor, new_duration)

    new_description = original.description
    for missing in missing_names:
        if missing in new_description:
            new_description = new_description.replace(missing, opt.tool.name)
            break  # only first occurrence to avoid weird substitutions

    return original.model_copy(
        update={
            "tools_required": new_tools,
            "duration_min": new_duration,
            "attentive_min": new_attentive,
            "description": new_description,
        }
    )


def _option_to_candidate(
    opt: ToolOption,
    edge: ProcessEdge,
    missing_names: list[str],
) -> SubstitutionCandidate:
    replacement = _build_replacement_edge(edge, opt, missing_names)
    time_delta = replacement.duration_min - edge.duration_min
    quality_delta = opt.quality_factor - 1.0

    sign_t = "+" if time_delta >= 0 else ""
    sign_q = "+" if quality_delta >= 0 else ""
    missing_str = "・".join(missing_names) if missing_names else "原器具"
    rationale = (
        f"{missing_str} の代替として {opt.tool.name} を使用"
        f"（時間 {sign_t}{time_delta:.1f}分、品質 {sign_q}{quality_delta:.2f}）"
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
    """Generate substitution candidates for a violating edge.

    The provided ``table`` is **mutated** when LLM discovers a new
    compatible combination — this is the data-layer learning mechanism.
    """

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
    ) -> list[SubstitutionCandidate]:
        # 1. Table lookup
        existing = find_compatible(self.table, edge.action, profile.tools_owned)
        existing = [opt for opt in existing if opt.tool.name not in set(violation.missing)]

        if existing:
            return [
                _option_to_candidate(opt, edge, violation.missing)
                for opt in existing[:MAX_CANDIDATES]
            ]

        # 2. LLM fallback
        proposal = self._ask_llm(edge, violation, profile)

        candidates: list[SubstitutionCandidate] = []
        for opt in proposal.candidates:
            if not is_tool_owned(opt.tool, profile.tools_owned):
                continue
            # Persist the discovery to the table for future deterministic lookups
            opt_with_source = opt.model_copy(
                update={"source": opt.source if opt.source != "seed" else "llm"}
            )
            add_entry(self.table, edge.action, opt_with_source)
            candidates.append(_option_to_candidate(opt_with_source, edge, violation.missing))

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
