# recipe-optimizer

調理者の制約条件（所有器具・スキル・現状の食材）に応じてレシピを最適化するソリューションの PoC。

## アーキテクチャの考え方

- レシピを **構造化DAG（ゴールノード + プロセスエッジ）** として表現する
- LLM は **入出力を Pydantic スキーマで縛った小粒モジュール** として使う（パース／代替提案／レンダリングの3用途のみ）
- それ以外（制約照合・DAG書換・スケジューリング）はすべて純アルゴリズム
- 個人最適化は **データ層の蓄積**（ツール代替テーブル、ユーザープロファイル、セッションログ）で実現する。モデル重みは固定

詳細設計は `docs/design.md`（後で追加）参照。

## モジュール構成

```
src/recipe_optimizer/
├── schemas.py          # Pydantic データモデル
├── llm/                # LLMクライアント抽象 + ローカル/モック実装
├── modules/            # parser, checker, proposer, selector, rewriter, scheduler, renderer
├── data_io/            # ツールテーブル・プロファイルの永続化
├── output/             # 数字リスト / DAG可視化 / 買い物リスト
├── pipeline.py         # オーケストレータ
└── cli.py
```

## 開発状況

PoC 開発中。Renderer A（事前にレシピ全体を生成）のみを対象とし、実行時対話モード（Renderer B）は後回し。

## ライセンス

Private (TBD)
