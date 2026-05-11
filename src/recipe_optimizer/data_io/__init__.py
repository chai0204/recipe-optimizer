"""Data persistence layer (JSON ↔ Pydantic)."""

from .profile import load_profile, save_profile
from .recipes import list_recipes, load_raw_recipe
from .tool_use_table import (
    add_entry,
    can_satisfy_spec,
    find_compatible,
    is_option_compatible,
    load_table,
    lookup,
    save_table,
    update_time_factor,
)

__all__ = [
    # profile
    "load_profile",
    "save_profile",
    # raw recipes
    "load_raw_recipe",
    "list_recipes",
    # tool_use_table I/O
    "load_table",
    "save_table",
    # tool_use_table operations
    "lookup",
    "can_satisfy_spec",
    "is_option_compatible",
    "find_compatible",
    "add_entry",
    "update_time_factor",
]
