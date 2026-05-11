"""Load, persist, and operate on the tool-use table.

The table is the data-layer learning store: at startup it holds seeded
knowledge; the proposer module appends new combinations discovered by
the LLM; over time, usage logs can adjust ``time_factor`` per user.

Behaviour stays here as free functions; ``ToolUseTable`` is a pure data
model in ``schemas.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import (
    ProcessType,
    Resource,
    ResourceKind,
    ResourceSpec,
    ToolOption,
    ToolUseTable,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_table(path: str | Path) -> ToolUseTable:
    p = Path(path)
    with p.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return ToolUseTable.model_validate(data)


def save_table(table: ToolUseTable, path: str | Path) -> None:
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


def _matches_name_hint(resource_name: str, name_hint: str) -> bool:
    """Same matching policy as the old tool-name matcher.

    Exact equality, or one-way substring containment ("片手鍋" matches
    "鍋" if the user owns "片手鍋", and "片手鍋" satisfies a hint of
    "鍋"). This absorbs minor wording variation between the seed table
    and concrete user inventory names.
    """
    if resource_name == name_hint:
        return True
    return resource_name in name_hint or name_hint in resource_name


def can_satisfy_spec(spec: ResourceSpec, owned: list[Resource]) -> bool:
    """Whether the user owns at least one resource matching ``spec``."""
    pool = [r for r in owned if r.kind == spec.kind]
    if not pool:
        return False
    if spec.name_hint is None:
        return True
    return any(_matches_name_hint(r.name, spec.name_hint) for r in pool)


def is_option_compatible(option: ToolOption, owned: list[Resource]) -> bool:
    """All ``ResourceSpec`` entries in the option must be satisfiable."""
    return all(can_satisfy_spec(spec, owned) for spec in option.resources)


def find_compatible(
    table: ToolUseTable,
    process: ProcessType | str,
    owned: list[Resource],
) -> list[ToolOption]:
    """``lookup`` + filter to options the user can perform, ranked by quality."""
    options = lookup(table, process)
    compatible = [opt for opt in options if is_option_compatible(opt, owned)]
    compatible.sort(
        key=lambda o: (o.quality_factor, -o.time_factor, o.confidence),
        reverse=True,
    )
    return compatible


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def add_entry(
    table: ToolUseTable,
    process: ProcessType | str,
    option: ToolOption,
) -> bool:
    """Append a new option. Returns False if an option with the same
    label already exists for this process."""
    key = _key(process)
    if key not in table.entries:
        table.entries[key] = []
    existing_labels = {opt.label for opt in table.entries[key]}
    if option.label in existing_labels:
        return False
    table.entries[key].append(option)
    return True


def update_time_factor(
    table: ToolUseTable,
    process: ProcessType | str,
    option_label: str,
    new_factor: float,
) -> bool:
    """Adjust the time_factor of an existing option (post-cook calibration)."""
    key = _key(process)
    for opt in table.entries.get(key, []):
        if opt.label == option_label:
            opt.time_factor = new_factor
            return True
    return False
