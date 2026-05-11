# Examples

実 LLM (Haiku 4.5) を使った end-to-end の出力サンプルです。設計判断やアーキテクチャの詳細は [../architecture.md](../architecture.md) を参照。

## サンプル一覧

| # | レシピ | 検証ポイント |
|---|---|---|
| 1 | [mugicha.md](mugicha.md) | 「やかんを持たない人」向けの自動代替（片手鍋・氷水急冷） |
| 2 | [oyakodon.md](oyakodon.md) | 複雑な8ステップレシピの DAG 化と並列化 |

## 共通の前提

### ユーザープロファイル（デモ用）

`data/user_profile.json` から抜粋：

- **熱源**: コンロ口 2 つ（バーナープール容量 2）
- **容器**: 片手鍋、両手鍋、フライパン、ボウル大 / 小、ザル
- **家電**: 電子レンジ、炊飯器
- **包丁系**: 包丁、まな板、菜箸、おたま、計量カップ、計量スプーン

**意図的に欠落**: やかん、オーブン、圧力鍋、ミキサー、冷蔵庫

この欠落により mugicha では substitution が、oyakodon では特定の workstation 指示で軽い代替が発火します。

### 実行コマンド

```bash
# Haiku 4.5 (Claude Code CLI 経由 — API キー不要)
recipe-optimizer run --recipe data/recipes/<name>.json --backend claude_cli --renderer

# ローカル GGUF (CPU で完結)
recipe-optimizer run --recipe data/recipes/<name>.json --backend gguf

# Mock (LLM なし、組み込み DAG fixture を使う)
recipe-optimizer demo --recipe <name>
```

## 出力の構造

3 つのビューがバンドルされて返ります。

1. **手順** — `--renderer` 付きなら家庭料理レシピ調の磨かれた日本語、なしなら parser 出力をそのまま番号付け
2. **スケジュール** — 各エッジの開始/終了時刻、リソース割当、クリティカルパス（CP）マーカー
3. **DAG** — Mermaid 記法。GitHub 上では図として描画される
4. **買い物リスト** — 元レシピの材料 + 代替後の必要器具（kind 別グルーピング）
5. **適用された代替** — 違反エッジとそれを解決した ToolOption の audit trail
