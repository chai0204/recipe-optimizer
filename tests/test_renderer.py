"""Tests for the renderer module (Mock client path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe_optimizer.data_io import load_profile
from recipe_optimizer.llm import LLMOutputError, MockLLMClient
from recipe_optimizer.modules.renderer import (
    RENDERER_LABEL,
    RecipeRenderer,
    RenderedSteps,
)
from recipe_optimizer.modules.scheduler import schedule
from recipe_optimizer.schemas import Constraints

from .fixtures.oyakodon_dag import make_oyakodon_dag

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "data" / "user_profile.json"


def _scheduled_oyakodon():
    dag = make_oyakodon_dag()
    sched = schedule(dag, Constraints(profile=load_profile(PROFILE_PATH)))
    return dag, sched


def test_renderer_returns_canned_steps():
    dag, sched = _scheduled_oyakodon()

    canned = RenderedSteps(
        steps=[
            "玉ねぎを薄切りにする",
            "鶏肉を一口大に切る",
            "卵を溶く",
            "鍋に調味料を入れ煮始める",
            "鶏肉と卵を加え半熟まで煮る",
            "器に盛る",
        ]
    )
    client = MockLLMClient()
    client.register(RENDERER_LABEL, canned)

    renderer = RecipeRenderer(client=client)
    out = renderer.render(dag, sched)
    assert out == canned.steps
    assert len(client.call_logs) == 1
    assert client.call_logs[0].label == RENDERER_LABEL


def test_renderer_raises_on_unregistered_mock():
    dag, sched = _scheduled_oyakodon()
    client = MockLLMClient()
    renderer = RecipeRenderer(client=client)
    with pytest.raises(LLMOutputError):
        renderer.render(dag, sched)


def test_renderer_includes_raw_recipe_context():
    """When ``raw_recipe`` is provided, its ingredients_text/steps_text
    should appear in the prompt (the LLM uses them as anchoring context)."""
    from recipe_optimizer.modules.renderer import _build_prompt
    from recipe_optimizer.schemas import RawRecipe

    dag, sched = _scheduled_oyakodon()
    raw = RawRecipe(
        id="t",
        title="親子丼",
        servings=1,
        ingredients_text=["鶏肉 1/2枚", "卵 2個"],
        steps_text=["切って煮る"],
        tips="火を通しすぎない",
    )

    prompt = _build_prompt(dag, sched, raw)
    assert "鶏肉 1/2枚" in prompt
    assert "切って煮る" in prompt
    assert "火を通しすぎない" in prompt
