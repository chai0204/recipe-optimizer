"""CLI entry point for the recipe-optimizer PoC.

Currently exposes a single ``demo`` mode that runs the optimizer on
hand-crafted DAG fixtures using a MockLLMClient — no real model needed.
This is the fastest way to see all four output views (steps, DAG,
shopping list, schedule) together.

A future ``parse`` mode will accept a recipe JSON file and run the full
parser-included pipeline against a configured LocalLLMClient. That
requires the model to be downloaded; not enabled by default.

Usage:

    recipe-optimizer demo --recipe oyakodon
    recipe-optimizer demo --recipe mugicha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data_io import load_profile, load_raw_recipe, load_table
from .llm import LLMClient, MockLLMClient
from .pipeline import (
    UnresolvableRecipeError,
    optimize_from_dag,
    optimize_from_raw,
)
from .schemas import RecipeDAG


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_PROFILE_PATH = _DATA_DIR / "user_profile.json"
_TABLE_PATH = _DATA_DIR / "tool_use_table.json"
_RECIPES_DIR = _DATA_DIR / "recipes"


def _print_kind_grouped_resources(resources: list, title: str = "## 必要なもの") -> None:
    print(f"\n{title}")
    if not resources:
        print("  (なし)")
        return
    current_kind = None
    for resource in resources:
        if resource.kind != current_kind:
            print(f"\n  ### {resource.kind.value}")
            current_kind = resource.kind
        print(f"  - {resource.name}")


def _print_rendered_recipe(rendered, profile, original_dag: RecipeDAG | None = None) -> None:
    print(f"# {rendered.title}\n")

    print("## 手順 (実行順)")
    for line in rendered.numbered_steps:
        print(f"  {line}")

    if rendered.substitutions_made:
        print("\n## 適用された代替")
        for sub in rendered.substitutions_made:
            print(f"  - {sub.original_edge_id}: {sub.rationale}")

    print(f"\n## スケジュール (合計 {rendered.schedule.total_duration_min:.1f}分)")
    # Use the post-substitution DAG for accurate step descriptions
    schedule_dag = rendered.optimized_dag or original_dag
    edges_by_id = (
        {} if schedule_dag is None else {e.id: e for e in schedule_dag.edges}
    )
    for step in sorted(rendered.schedule.steps, key=lambda s: s.start_min):
        marker = " (CP)" if step.edge_id in rendered.schedule.critical_path_edge_ids else ""
        edge = edges_by_id.get(step.edge_id)
        desc = edge.description if edge else step.edge_id
        print(f"  [{step.start_min:5.1f}-{step.end_min:5.1f}] {desc}{marker}")

    print("\n## 買い物 / 食材")
    for ing in rendered.shopping_list.ingredients:
        print(f"  - {ing.name}")

    _print_kind_grouped_resources(rendered.shopping_list.resources_needed)

    print("\n## DAG (Mermaid)")
    print("```mermaid")
    print(rendered.mermaid_dag)
    print("```")


def _cmd_demo(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    table = load_table(args.table)

    # Lazy import of test fixtures — they live under tests/fixtures/
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "tests"))
    try:
        if args.recipe == "oyakodon":
            from fixtures.oyakodon_dag import make_oyakodon_dag

            dag = make_oyakodon_dag()
            raw_path = _RECIPES_DIR / "oyakodon.json"
        elif args.recipe == "mugicha":
            from fixtures.mugicha_dag import make_mugicha_dag

            dag = make_mugicha_dag()
            raw_path = None
        else:
            print(
                f"unknown demo recipe: {args.recipe!r} (try: oyakodon, mugicha)",
                file=sys.stderr,
            )
            return 2
    finally:
        sys.path.pop(0)

    raw_recipe = load_raw_recipe(raw_path) if raw_path and raw_path.exists() else None
    client = MockLLMClient()  # demo path never invokes parser; mock is fine

    try:
        rendered = optimize_from_dag(
            dag=dag,
            profile=profile,
            table=table,
            client=client,
            raw_recipe=raw_recipe,
        )
    except UnresolvableRecipeError as exc:
        print(f"!! 最適化失敗: {exc}", file=sys.stderr)
        return 1

    _print_rendered_recipe(rendered, profile, original_dag=dag)
    return 0


def _build_llm_client(backend: str, model: str | None) -> LLMClient:
    """Instantiate a real LLM client according to ``backend``.

    Supported backends:
      - ``mock``: returns MockLLMClient (will fail on parser since
        nothing is registered — useful only when bypassing parsing).
      - ``claude_cli``: uses ``ClaudeCliClient`` (no API key needed,
        wraps ``claude -p``; default model ``haiku``).
      - ``gguf``: ``LlamaCppClient`` with local GGUF (gemma-4-E4B Q4_K_M
        by default).
    """
    if backend == "claude_cli":
        from .llm.claude_cli import ClaudeCliClient

        return ClaudeCliClient(model=model or "haiku")

    if backend == "gguf":
        from huggingface_hub import hf_hub_download

        from .llm.llamacpp import LlamaCppClient

        path = hf_hub_download(
            "unsloth/gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
        )
        return LlamaCppClient(model_path=path)

    return MockLLMClient()


def _cmd_run(args: argparse.Namespace) -> int:
    """Parse a RawRecipe JSON with a real LLM and optimise it end-to-end."""
    profile = load_profile(args.profile)
    table = load_table(args.table)
    raw = load_raw_recipe(args.recipe)

    print(
        f"<!-- source: {args.recipe} | "
        f"backend: {args.backend} (model={args.model or 'default'}) -->",
        flush=True,
    )

    client = _build_llm_client(args.backend, args.model)

    try:
        rendered = optimize_from_raw(
            raw_recipe=raw,
            profile=profile,
            table=table,
            client=client,
            raise_on_unresolvable=False,
            use_renderer=args.renderer,
        )
    except UnresolvableRecipeError as exc:
        print(f"!! 最適化失敗: {exc}", file=sys.stderr)
        return 1

    _print_rendered_recipe(rendered, profile, original_dag=rendered.optimized_dag)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recipe-optimizer",
        description="Recipe optimization PoC — adapt recipes to a cook's constraints.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="run with a built-in DAG fixture (no LLM needed)")
    demo.add_argument("--recipe", default="oyakodon", help="oyakodon | mugicha")
    demo.add_argument(
        "--profile", default=str(_PROFILE_PATH), help="path to user_profile.json"
    )
    demo.add_argument(
        "--table", default=str(_TABLE_PATH), help="path to tool_use_table.json"
    )
    demo.set_defaults(func=_cmd_demo)

    run = sub.add_parser(
        "run",
        help="parse a real RawRecipe JSON with a live LLM and optimise it",
    )
    run.add_argument("--recipe", required=True, help="path to a RawRecipe JSON file")
    run.add_argument(
        "--backend",
        default="claude_cli",
        choices=["claude_cli", "gguf", "mock"],
        help="LLM backend",
    )
    run.add_argument("--model", default=None, help="model name override (backend-specific)")
    run.add_argument(
        "--renderer",
        action="store_true",
        help="enable LLM renderer for polished natural-language steps (extra LLM call)",
    )
    run.add_argument(
        "--profile", default=str(_PROFILE_PATH), help="path to user_profile.json"
    )
    run.add_argument(
        "--table", default=str(_TABLE_PATH), help="path to tool_use_table.json"
    )
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
