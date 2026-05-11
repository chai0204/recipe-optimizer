"""Tests for data_io.profile and data_io.tool_use_table (resource model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import (
    add_entry,
    can_satisfy_spec,
    find_compatible,
    is_option_compatible,
    load_profile,
    load_table,
    lookup,
    save_profile,
    save_table,
    update_time_factor,
)
from recipe_optimizer.schemas import (
    ProcessType,
    Resource,
    ResourceKind,
    ResourceSpec,
    SkillLevel,
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
    assert profile.skill_level == SkillLevel.INTERMEDIATE
    assert len(profile.cooks) == 1
    assert len(profile.burners) == 2
    assert len(profile.workstations) == 1

    container_names = {r.name for r in profile.containers}
    assert "片手鍋" in container_names
    assert "両手鍋" in container_names
    assert "フライパン" in container_names
    # Intentionally absent
    assert "やかん" not in container_names
    assert "圧力鍋" not in container_names


def test_save_then_load_profile_round_trip(tmp_path):
    profile = UserProfile(
        user_id="round_trip",
        cooks=[Resource(id="c1", kind=ResourceKind.COOK, name="自分")],
        containers=[Resource(id="p", kind=ResourceKind.CONTAINER, name="片手鍋")],
    )
    out = tmp_path / "p.json"
    save_profile(profile, out)
    loaded = load_profile(out)
    assert loaded.user_id == "round_trip"
    assert len(loaded.cooks) == 1
    assert loaded.containers[0].name == "片手鍋"


# ---------------------------------------------------------------------------
# Table I/O
# ---------------------------------------------------------------------------


def test_load_table_real_data():
    table = load_table(TABLE_PATH)
    assert "boil_water" in table.entries
    assert "saute" in table.entries
    assert "chop" in table.entries

    bw_labels = {opt.label for opt in table.entries["boil_water"]}
    assert "やかん+コンロ" in bw_labels
    assert "電子レンジ" in bw_labels
    assert "片手鍋+コンロ" in bw_labels


def test_lookup_known_and_unknown():
    table = load_table(TABLE_PATH)
    assert len(lookup(table, ProcessType.BOIL_WATER)) > 0
    assert lookup(table, ProcessType.UNKNOWN) == []


# ---------------------------------------------------------------------------
# can_satisfy_spec
# ---------------------------------------------------------------------------


def test_can_satisfy_spec_exact_match():
    owned = [Resource(id="p", kind=ResourceKind.CONTAINER, name="片手鍋")]
    assert can_satisfy_spec(
        ResourceSpec(kind=ResourceKind.CONTAINER, name_hint="片手鍋"), owned
    )
    assert not can_satisfy_spec(
        ResourceSpec(kind=ResourceKind.CONTAINER, name_hint="両手鍋"), owned
    )


def test_can_satisfy_spec_substring_match():
    """Owned 'フライパン' should satisfy 'フライパン' or '深型フライパン'."""
    owned = [Resource(id="p", kind=ResourceKind.CONTAINER, name="フライパン")]
    assert can_satisfy_spec(
        ResourceSpec(kind=ResourceKind.CONTAINER, name_hint="深型フライパン"), owned
    )


def test_can_satisfy_spec_no_hint_any_kind_works():
    """If name_hint is None, any resource of the right kind suffices."""
    owned = [Resource(id="b1", kind=ResourceKind.BURNER, name="コンロ口1")]
    assert can_satisfy_spec(ResourceSpec(kind=ResourceKind.BURNER), owned)
    assert not can_satisfy_spec(ResourceSpec(kind=ResourceKind.APPLIANCE), owned)


def test_can_satisfy_spec_returns_false_when_pool_empty():
    assert not can_satisfy_spec(
        ResourceSpec(kind=ResourceKind.APPLIANCE, name_hint="電子レンジ"), []
    )


# ---------------------------------------------------------------------------
# is_option_compatible + find_compatible
# ---------------------------------------------------------------------------


def test_find_compatible_filters_unowned_resources():
    """User lacks やかん and 電気ケトル → those options dropped."""
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)

    all_owned = (
        profile.cooks
        + profile.burners
        + profile.workstations
        + profile.containers
        + profile.appliances
        + profile.utensils
    )
    options = find_compatible(table, ProcessType.BOIL_WATER, all_owned)
    labels = {opt.label for opt in options}

    assert "やかん+コンロ" not in labels  # no やかん
    assert "電気ケトル" not in labels  # no 電気ケトル
    assert "片手鍋+コンロ" in labels
    assert "両手鍋+コンロ" in labels
    assert "電子レンジ" in labels


def test_find_compatible_bake_returns_empty_for_user_without_oven():
    profile = load_profile(PROFILE_PATH)
    table = load_table(TABLE_PATH)
    all_owned = (
        profile.cooks + profile.burners + profile.workstations
        + profile.containers + profile.appliances + profile.utensils
    )
    assert find_compatible(table, ProcessType.BAKE, all_owned) == []


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def test_add_entry_appends_new():
    table = ToolUseTable(entries={})
    opt = ToolOption(
        label="ホットプレート",
        resources=[
            ResourceSpec(kind=ResourceKind.COOK),
            ResourceSpec(kind=ResourceKind.APPLIANCE, name_hint="ホットプレート"),
        ],
        time_factor=1.1,
        source="llm",
    )
    appended = add_entry(table, ProcessType.SAUTE, opt)
    assert appended is True
    assert len(table.entries["saute"]) == 1
    assert table.entries["saute"][0].source == "llm"


def test_add_entry_skips_duplicate_label():
    table = ToolUseTable(
        entries={
            "saute": [
                ToolOption(
                    label="フライパン+コンロ",
                    resources=[
                        ResourceSpec(kind=ResourceKind.COOK),
                        ResourceSpec(kind=ResourceKind.BURNER),
                    ],
                )
            ]
        }
    )
    dup = ToolOption(
        label="フライパン+コンロ",
        resources=[ResourceSpec(kind=ResourceKind.COOK)],
        time_factor=2.0,
    )
    appended = add_entry(table, ProcessType.SAUTE, dup)
    assert appended is False
    assert len(table.entries["saute"]) == 1


def test_update_time_factor():
    table = ToolUseTable(
        entries={
            "boil": [
                ToolOption(
                    label="片手鍋+コンロ",
                    resources=[ResourceSpec(kind=ResourceKind.COOK)],
                    time_factor=1.0,
                )
            ]
        }
    )
    assert update_time_factor(table, ProcessType.BOIL, "片手鍋+コンロ", 1.3) is True
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
                    label="鍋",
                    resources=[
                        ResourceSpec(kind=ResourceKind.COOK, relative_duration=0.3),
                        ResourceSpec(kind=ResourceKind.CONTAINER, name_hint="鍋"),
                    ],
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
    assert loaded.entries["boil"][0].label == "鍋"
    assert loaded.entries["boil"][0].time_factor == 1.5
    assert loaded.entries["boil"][0].resources[0].relative_duration == 0.3
