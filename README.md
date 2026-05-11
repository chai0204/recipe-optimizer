# recipe-optimizer

調理者の制約条件（所有器具・スキル・現状の食材）に応じてレシピを最適化するソリューションの PoC。

> **Status**: PoC end-to-end が動作確認済み。Haiku 4.5 で実レシピのパース→代替提案→スケジュール→自然な日本語ステップ生成までの全段が成立する。実プロダクト化に向けては Issue を残し、本リポは設計検証の主目的を達成。

詳細設計は **[docs/architecture.md](docs/architecture.md)** を参照。動作サンプルは **[docs/examples/](docs/examples/)** に格納（実 LLM 出力を含む共有用デモ 2 件）。元レシピと LLM 由来の **避けられない不確実性** は **[docs/limitations.md](docs/limitations.md)** に明記。

## アーキテクチャ要旨

- レシピを **構造化DAG（GoalNode = 状態、ProcessEdge = アクション）** で表現
- 各 ProcessEdge は **`resource_uses: list[ResourceRequirement]`** で必要リソース（cook / burner / workstation / container / appliance / utensil）を明示
- LLM は **3つの小粒モジュール**（parser / proposer / renderer）に閉じ込め、入出力を Pydantic スキーマで縛って variance を抑制
- その他（制約照合・代替候補スコアリング・DAG書換・スケジューリング・出力フォーマット）は純アルゴリズム
- 個人最適化は **データ層の蓄積**（`tool_use_table.json`、ユーザープロファイル）で実現

## モジュール構成

```
src/recipe_optimizer/
├── schemas.py          # Pydantic データモデル（Resource/ProcessEdge/ToolOption 等）
├── llm/                # LLMClient 抽象 + Mock / Local (transformers) / LlamaCpp (GGUF) / ClaudeCli (claude -p)
├── modules/            # parser / checker / proposer / selector / rewriter / scheduler / renderer
├── data_io/            # JSON ↔ Pydantic（profile / tool_use_table / raw recipe）
├── output/             # 数字リスト / Mermaid DAG / 買い物リスト
├── pipeline.py         # オーケストレータ（optimize_from_dag / optimize_from_raw）
└── cli.py              # recipe-optimizer demo | run
```

## クイックスタート

### インストール

```bash
uv pip install -e ".[dev]"            # コア + テスト
uv pip install -e ".[dev,gguf]"       # + llama-cpp-python (CPU GGUF backend)
uv pip install -e ".[dev,local-llm]"  # + transformers (heavy bf16 backend)
```

### Mock 経路でデモ（LLM 不要）

```bash
recipe-optimizer demo --recipe oyakodon
recipe-optimizer demo --recipe mugicha
```

組み込みの DAG fixture を使い、checker → proposer (テーブルヒット) → selector → rewriter → scheduler → 出力フォーマッタの全段を瞬時に走らせる。

### 実 LLM 経路（Haiku 4.5 経由）

```bash
# Claude Code の認証を流用するので API キー不要
recipe-optimizer run --recipe data/recipes/mugicha.json \
  --backend claude_cli --renderer
```

`--renderer` を付けると Haiku が最終的な日本語ステップを家庭料理レシピ調に整形する（LLM 呼び出しが2回に増える）。

### GGUF（ローカル CPU）経路

```bash
# 初回は ~5GB のモデル DL（unsloth/gemma-4-E4B-it Q4_K_M）
recipe-optimizer run --recipe data/recipes/oyakodon.json --backend gguf
```

## 検証済みの動作

| シナリオ | 結果 |
|---|---|
| oyakodon (8ステップ) を Haiku で end-to-end | ✅ 15ノード DAG、makespan 14.5分 |
| mugicha (3ステップ + やかん不所持) | ✅ 片手鍋・氷水急冷へ自動代替 |
| Mock 経路全テスト | ✅ 95 tests pass (0.2s) |

詳細は [docs/architecture.md](docs/architecture.md) と Issue を参照。

## ライセンス

[MIT License](LICENSE) — Copyright (c) 2026 Shun
