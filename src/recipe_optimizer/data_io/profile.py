"""Load and persist UserProfile from JSON.

Pydantic v2 silently ignores extra fields by default, so cosmetic keys
like ``_notes_for_poc`` in the JSON file pose no issue. They are simply
not round-tripped on save.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import UserProfile


def load_profile(path: str | Path) -> UserProfile:
    """Load a UserProfile from a JSON file."""
    p = Path(path)
    with p.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return UserProfile.model_validate(data)


def save_profile(profile: UserProfile, path: str | Path) -> None:
    """Save a UserProfile to a JSON file (UTF-8, indented, non-ASCII preserved)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        json.dump(profile.model_dump(), fp, ensure_ascii=False, indent=2)
        fp.write("\n")
