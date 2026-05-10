"""Load, persist, and operate on the tool-use table.

The table is the data-layer learning store: at startup it holds the
seeded knowledge; the proposer module appends new combinations
discovered by the LLM; over time the schedule/usage records can adjust
``time_factor`` per user.

Behaviour stays in this module as free functions; ``ToolUseTable`` in
schemas.py is a pure data model.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import ProcessType, Tool, ToolOption, ToolUseTable


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_table(path: str | Path) -> ToolUseTable:
    """Load a ToolUseTable from JSON.

    Top-level cosmetic keys (e.g. ``_schema_note``) are ignored by
    Pydantic's default ``extra="ignore"``.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return ToolUseTable.model_validate(data)


def save_table(table: ToolUseTable, path: str | Path) -> None:
    """Save a ToolUseTable to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        json.dump(table.model_dump(), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def _key(process: ProcessType | str) -> str:
    return process.value if isinstance(process, ProcessType) else process


def lookup(table: ToolUseTable, process: ProcessType | str) -> list[ToolOption]:
    """All registered options for a given process type."""
    return list(table.entries.get(_key(process), []))


def is_tool_owned(option_tool: Tool, owned: list[Tool]) -> bool:
    """Whether ``option_tool`` is satisfied by the user's owned tools.

    Composite tools written as ``"包丁+まな板"`` require **all** parts
    to be owned. Each part matches by:

    1. Exact name equality.
    2. One-way substring containment (e.g., owned ``"フライパン"`` matches
       option spec ``"深型フライパン"``, and vice versa). This handles
       small variations between seed-table names and concrete user
       inventory names without resorting to embedding similarity.
    """
    owned_names = {t.name for t in owned}
    parts = [p.strip() for p in option_tool.name.split("+")]
    return all(_part_matches_owned(part, owned_names) for part in parts)


def _part_matches_owned(part: str, owned_names: set[str]) -> bool:
    if part in owned_names:
        return True
    for owned_name in owned_names:
        if part in owned_name or owned_name in part:
            return True
    return False


def filter_by_owned(
    options: list[ToolOption], owned_tools: list[Tool]
) -> list[ToolOption]:
    """Keep only options whose required tool is satisfied by ``owned_tools``."""
    return [opt for opt in options if is_tool_owned(opt.tool, owned_tools)]


def find_compatible(
    table: ToolUseTable,
    process: ProcessType | str,
    owned_tools: list[Tool],
) -> list[ToolOption]:
    """``lookup`` ∘ ``filter_by_owned``, sorted by quality_factor desc.

    This is the primary read API used by the proposer module.
    """
    options = lookup(table, process)
    compatible = filter_by_owned(options, owned_tools)
    compatible.sort(
        key=lambda o: (o.quality_factor, -o.time_factor, o.confidence),
        reverse=True,
    )
    return compatible


# ---------------------------------------------------------------------------
# Mutations (used by proposer when LLM discovers new combinations)
# ---------------------------------------------------------------------------


def add_entry(
    table: ToolUseTable,
    process: ProcessType | str,
    option: ToolOption,
) -> bool:
    """Append a new ToolOption.

    Returns:
        True if appended, False if a duplicate (same tool name) already exists.
    """
    key = _key(process)
    if key not in table.entries:
        table.entries[key] = []
    existing_names = {opt.tool.name for opt in table.entries[key]}
    if option.tool.name in existing_names:
        return False
    table.entries[key].append(option)
    return True


def update_time_factor(
    table: ToolUseTable,
    process: ProcessType | str,
    tool_name: str,
    new_factor: float,
) -> bool:
    """Adjust the time_factor of an existing entry (used after observing
    real cooking durations).

    Returns:
        True if the entry was found and updated, False otherwise.
    """
    key = _key(process)
    for opt in table.entries.get(key, []):
        if opt.tool.name == tool_name:
            opt.time_factor = new_factor
            return True
    return False
