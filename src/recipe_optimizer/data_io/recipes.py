"""Load RawRecipe data from JSON files in data/recipes/."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import RawRecipe


def load_raw_recipe(path: str | Path) -> RawRecipe:
    """Load a single RawRecipe JSON file."""
    p = Path(path)
    with p.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return RawRecipe.model_validate(data)


def list_recipes(directory: str | Path) -> list[RawRecipe]:
    """Load all RawRecipe JSON files in a directory (non-recursive)."""
    d = Path(directory)
    return [load_raw_recipe(p) for p in sorted(d.glob("*.json"))]
