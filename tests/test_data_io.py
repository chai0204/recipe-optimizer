"""Tests for data_io.profile and data_io.tool_use_table."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import (
    add_entry,
    filter_by_owned,
    find_compatible,
    is_tool_owned,
    load_profile,
    load_table,
    lookup,
    save_profile,
    save_table,
    update_time_factor,
)
from recipe_optimizer.schemas import (
    ProcessType,
    SkillLevel,
    Tool,
    ToolKind,
    ToolOption,
    ToolUseTable,
    UserProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"
TABLE_PATH = REPO_ROOT / "data" / "tool_use_table.json"


# ---------------------------------------------------------------------------
# Profile I/O
# ---------------------------------------------------------------------------


def test_load_profile_real_data():
    profile = load_profile(PROFILE_PATH)
    assert profile.user_id == "demo_user_01"
    assert profile.burners_count == 2
    assert profile.skill_level == SkillLevel.INTERMEDIATE

    names = {t.name for t in profile.tools_owned}
    assert "片手鍋" in names
    assert "両手鍋" in names
    assert "フライパン" in names
    assert "電子レンジ" in names
    # Intentionally absent (used to trigger substitution logic in tests)
    assert "やかん" not in names
    assert "オーブン" not in names
    assert "圧力鍋" not in names


def test_save_then_load_profile_round_trip(tmp_path):
    profile = UserProfile(
        user_id="round_trip",
        tools_owned=[Tool(name="包丁", kind=ToolKind.UTENSIL)],
        burners_count=3,
    )
    out = tmp_path / "p.json"
    save_profile(profile, out)
    loaded = load_profile(out)
    assert loaded.user_id == "round_trip"
    assert loaded.burners_count == 3
    assert loaded.tools_owned[0].name == "包丁"


# ---------------------------------------------------------------------------
# Table I/O + lookup
# ---------------------------------------------------------------------------


def test_load_table_real_data():
    table = load_table(TABLE_PATH)
    assert "boil_water" in table.entries
    assert "saute" in table.entries
    assert "chop" in table.entries

    bw = table.entries["boil_water"]
    bw_names = {opt.tool.name for opt in bw}
    assert "やかん" in bw_names
    assert "電子レンジ" in bw_names
    assert "片手鍋" in bw_names


def test_lookup_known_and_unknown():
    table = load_table(TABLE_PATH)
    assert len(lookup(table, ProcessType.BOIL_WATER)) > 0
    assert lookup(table, ProcessType.UNKNOWN) == []


# ---------------------------------------------------------------------------
# is_tool_owned (matching logic)
# ---------------------------------------------------------------------------


def test_is_tool_owned_exact_match():
    owned = [Tool(name="片手鍋", kind=ToolKind.CONTAINER)]
    assert is_tool_owned(Tool(name="片手鍋", kind=ToolKind.CONTAINER), owned)
    assert not is_tool_owned(Tool(name="両手鍋", kind=ToolKind.CONTAINER), owned)


def test_is_tool_owned_substring_match():
    """Owned 'フライパン' should satisfy option spec '深型フライパン' and vice versa."""
    owned = [Tool(name="フライパン", kind=ToolKind.CONTAINER)]
    assert is_tool_owned(Tool(name="深型フライパン", kind=ToolKind.CONTAINER), owned)
    assert is_tool_owned(Tool(name="フライパン", kind=ToolKind.CONTAINER), owned)


def test_is_tool_owned_composite_all_required():
    owned = [
        Tool(name="包丁", kind=ToolKind.UTENSIL),
        Tool(name="まな板", kind=ToolKind.UTENSIL),
    ]
    composite = Tool(name="包丁+まな板", kind=ToolKind.UTENSIL)
    assert is_tool_owned(composite, owned)

    owned_partial = [Tool(name="包丁", kind=ToolKind.UTENSIL)]
    assert not is_tool_owned(composite, owned_partial)


# ---------------------------------------------------------------------------
# filter_by_owned + find_compatible
# ---------------------------------------------------------------------------


def test_find_compatible_filters_unowned_for_boil_water():
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    options = find_compatible(table, ProcessType.BOIL_WATER, profile.tools_owned)
    names = {opt.tool.name for opt in options}

    # User has no やかん nor 電気ケトル → must be filtered out
    assert "やかん" not in names
    assert "電気ケトル" not in names
    # User has 片手鍋, 両手鍋, 電子レンジ → must remain
    assert "片手鍋" in names
    assert "両手鍋" in names
    assert "電子レンジ" in names


def test_find_compatible_chop_uses_composite_match():
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    options = find_compatible(table, ProcessType.CHOP, profile.tools_owned)
    names = {opt.tool.name for opt in options}

    # 包丁 and まな板 are both owned
    assert "包丁+まな板" in names
    # User does not own these
    assert "フードプロセッサ" not in names
    assert "キッチンばさみ" not in names


def test_find_compatible_bake_yields_none_for_user_without_oven():
    """User profile intentionally lacks oven, toaster, and grill —
    find_compatible should return [] for ProcessType.BAKE."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    options = find_compatible(table, ProcessType.BAKE, profile.tools_owned)
    assert options == []


def test_find_compatible_sorts_by_quality_desc():
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    options = find_compatible(table, ProcessType.SAUTE, profile.tools_owned)

    qualities = [opt.quality_factor for opt in options]
    assert qualities == sorted(qualities, reverse=True)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def test_add_entry_appends_new():
    table = ToolUseTable(entries={})
    new_opt = ToolOption(
        tool=Tool(name="ホットプレート", kind=ToolKind.APPLIANCE),
        time_factor=1.1,
        source="llm",
    )
    appended = add_entry(table, ProcessType.SAUTE, new_opt)
    assert appended is True
    assert len(table.entries["saute"]) == 1
    assert table.entries["saute"][0].tool.name == "ホットプレート"
    assert table.entries["saute"][0].source == "llm"


def test_add_entry_skips_duplicate_by_name():
    table = ToolUseTable(
        entries={
            "saute": [
                ToolOption(tool=Tool(name="フライパン", kind=ToolKind.CONTAINER))
            ]
        }
    )
    dup = ToolOption(
        tool=Tool(name="フライパン", kind=ToolKind.CONTAINER),
        time_factor=2.0,  # different params, same name
    )
    appended = add_entry(table, ProcessType.SAUTE, dup)
    assert appended is False
    assert len(table.entries["saute"]) == 1


def test_update_time_factor():
    table = ToolUseTable(
        entries={
            "boil": [
                ToolOption(
                    tool=Tool(name="片手鍋", kind=ToolKind.CONTAINER), time_factor=1.0
                )
            ]
        }
    )
    assert update_time_factor(table, ProcessType.BOIL, "片手鍋", 1.3) is True
    assert table.entries["boil"][0].time_factor == 1.3
    assert update_time_factor(table, ProcessType.BOIL, "missing", 0.5) is False


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_table_round_trip(tmp_path):
    table = ToolUseTable(
        entries={
            "boil": [
                ToolOption(
                    tool=Tool(name="鍋", kind=ToolKind.CONTAINER),
                    time_factor=1.5,
                    quality_factor=0.9,
                    source="seed",
                )
            ]
        }
    )
    out = tmp_path / "table.json"
    save_table(table, out)
    loaded = load_table(out)
    assert loaded.entries["boil"][0].tool.name == "鍋"
    assert loaded.entries["boil"][0].time_factor == 1.5
    assert loaded.entries["boil"][0].quality_factor == 0.9
