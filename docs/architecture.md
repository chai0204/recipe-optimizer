# recipe-optimizer アーキテクチャ

## 1. 設計上の核心

3つの設計原則が全体を貫いている。

### 1.1 「LLM = 制約された純粋関数モジュール」

LLM を呼び出す箇所は **3点に限定** している（parser / proposer / renderer）。各呼び出しは:

```
LLMModule(input: PydanticSchema_in) → PydanticSchema_out
```

の純粋関数として扱い、`LLMClient.generate_structured()` が以下を一括で担当する:

- システムプロンプト + few-shot examples + ユーザープロンプトの組み立て
- 出力 JSON のパース
- Pydantic スキーマでの検証（失敗時は1回まで再試行）
- 呼び出しログの記録（label 付き、後でばらつき分析可）

**スコープ最小化** が出力 variance を抑える鍵。たとえば proposer は「違反エッジ1つに対する代替候補」だけを生成し、レシピ全体を一度に最適化しようとはしない。

### 1.2 「リソース割当は明示的に declare する」

ProcessEdge は `tools_required: list[Tool]` のような曖昧な記述ではなく、`resource_uses: list[ResourceRequirement]` を持つ。各 requirement は:

| フィールド | 意味 |
|---|---|
| `kind` | `cook` / `burner` / `workstation` / `container` / `appliance` / `utensil` のいずれか |
| `name_hint` | 「片手鍋」「電子レンジ」等の具体名（任意） |
| `hold_duration_min` | このスロットを占有する時間 |
| `start_offset_min` | エッジ開始からのオフセット（既定 0） |

これにより、たとえば「2分の煮込みで人が拘束されるのは最初の30秒だけ」を:

```python
resource_uses=[
    ResourceRequirement(kind=COOK,      hold_duration_min=0.5),
    ResourceRequirement(kind=BURNER,    hold_duration_min=2.0),
    ResourceRequirement(kind=CONTAINER, hold_duration_min=2.0, name_hint="片手鍋"),
]
```

と書ける。スケジューラは `hold_duration_min` を直接消費するため、暗黙の「action から推論」ロジックがどこにも残らない。

### 1.3 「データ層が学習層、モデルは固定」

`tool_use_table.json` は LLM の探索結果のキャッシュ兼学習層として機能する:

- 初期は seed エントリ（boil_water = やかん/片手鍋/電子レンジ等）
- proposer が table にない action で LLM フォールバックを叩いたら、結果を `source="llm"` でテーブルに追記
- 同じ action で次回は決定論的にテーブルヒット

`UserProfile.skill_factors` のような個人パラメータもデータ層の更新で改善されていく（オフライン fine-tuning 不要）。

## 2. データモデル

### 2.1 主要型

| 型 | 役割 |
|---|---|
| `Resource` | ユーザーが所有する1つのリソース（`id`/`kind`/`name`/`attributes`） |
| `ResourceRequirement` | エッジが必要とするリソース1枠の予約 |
| `ResourceSpec` | テーブル内の「`relative_duration` で表現された pattern」 |
| `ToolOption` | アクションを実現する1パターン（resources 配列 + time_factor + quality_factor） |
| `GoalNode` | DAG のノード（食材の状態 / 中間状態 / 最終状態） |
| `ProcessEdge` | DAG のエッジ（action / description / duration_min / resource_uses） |
| `RecipeDAG` | レシピ全体のグラフ |
| `UserProfile` | リソースを kind 別プールで保持（`cooks`, `burners`, `workstations`, `containers`, `appliances`, `utensils`） |
| `ConstraintViolation` | 解決不能なリソース要求（`kind` + `name_hint`） |
| `SubstitutionCandidate` | 代替提案（`replacement_edge` + scoring deltas） |
| `Schedule` | スケジュール結果（`steps`, `total_duration_min`, `critical_path_edge_ids`） |
| `RenderedRecipe` | 最終出力 bundle（番号付き手順 / Mermaid DAG / 買い物リスト / Schedule） |

### 2.2 DAG の整合性

`RecipeDAG.model_validator(mode="after")` で以下を強制:

- ノード ID 重複なし / エッジ ID 重複なし
- `final_node_id` がノードに存在
- 全 `from_nodes` / `to_node` がノードに存在
- 各 `resource_use.start_offset + hold_duration ≤ edge.duration_min`

LLM が壊れた DAG を返した場合は `ValidationError` が発火し、`LLMClient` のリトライループが再試行に入る。

## 3. パイプライン

```
RawRecipe ──[parser, LLM]──▶ RecipeDAG
                                │
                  ┌─────────────┘
                  ▼
            ┌── checker ──▶ ConstraintViolation[]
            │                       │
            │                  ┌────┴────────┐
            │                  ▼             ▼
            │              proposer      (no violation)
            │           (table + LLM)        │
            │                  │             │
            │                  ▼             │
            │              selector          │
            │           (linear scoring)     │
            │                  │             │
            │                  ▼             │
            │              rewriter          │
            │            (apply subs)        │
            │                  │             │
            └──────────┐       │             │
                       ▼       ▼             ▼
                       checker (verify) ────┐
                              │              │
                              ▼              ▼
                       scheduler (RCPSP-lite over resource pools)
                              │
                              ▼
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        numbered_list    dag_viz       shopping_list
        (or renderer)   (Mermaid)
                              │
                              ▼
                       RenderedRecipe
```

各ステップの責務:

### parser (LLM)

`RawRecipe` のテキストを `RecipeDAG` に変換する。
- システムプロンプトに **出力スキーマの compact 表現** と **典型的なリソースパターン**（simmer / chop など）を埋め込む
- **Few-shot example**（ゆで卵レシピ1件）を `examples` パラメータで渡し、duration / cook hold / name_hint の具体値を学習させる
- 暗黙の prep ステップ（「切った鶏肉を加える」→ chop エッジを補う）も指示
- 出力は `inject_schema=False` だが grammar 制約バックエンド（llama.cpp の response_format）で構造強制

### checker (pure)

`find_violations(dag, constraints)` が DAG の各エッジの各 `ResourceRequirement` を `UserProfile` のプールに照合し、未充足を `ConstraintViolation` として返す。**意味的整合性ではなく、kind+name_hint の照合のみ。**

### proposer (LLM + table)

1. `find_compatible(table, edge.action, owned)` でテーブル先引き
2. ヒットなら option を `SubstitutionCandidate` に変換して返す（LLM 呼ばない）
3. ヒットゼロなら LLM に小粒呼び出し → 帰ってきた候補のうち user が所有可能なものをテーブルに `source="llm"` で書き戻し → 候補化

各 option は `resources: list[ResourceSpec]` を持つ。`time_factor` で edge duration をスケールし、`relative_duration` で各リソースの hold 時間を計算する。

```python
new_duration = original.duration_min * option.time_factor
for spec in option.resources:
    hold = new_duration * spec.relative_duration
    new_uses.append(ResourceRequirement(kind=spec.kind, name_hint=spec.name_hint, hold_duration_min=hold))
```

電子レンジでお湯を沸かす option は最初から `burner` を含まないので、自動的に「コンロ口を占有しない」substitution になる。**暗黙のヒューリスティック（電子レンジは heat_source を不要にする等）は不要**。

### selector (pure)

線形加重スコア:

```
score = w_quality × quality_delta - w_time × time_delta_min
```

既定重み `(quality=1.0, time_min=0.05)` は「品質1単位の損失 ≈ 時間20分」。`rank_candidates` は全候補をスコア降順で返し、`select_best` はトップを返す。タイブレークは入力順保持（決定論）。

### rewriter (pure)

`apply_substitution(dag, candidate)` がエッジ ID で対象を見つけ、`replacement_edge` で差し替えた新 DAG を返す。`replacement_edge` は id/from_nodes/to_node を保存するため、**トポロジーは変わらない**。バッチ版 `apply_substitutions` も提供。出力は新たな `RecipeDAG` を構築するので Pydantic の validator が再実行される。

### scheduler (pure)

**RCPSP-lite**（resource-constrained scheduling）。各 `Resource.id` ごとに `busy_until: float` を持つ `_ResourcePool` を維持し、エッジを topological 順に処理:

1. predecessor の終了時刻 = `prereq_end`
2. 各 `ResourceRequirement` に対して、kind+name_hint で候補となる Resource を列挙し、最早に空く slot を選ぶ
3. `start = max(prereq_end, max(slot earliest start for each requirement))`
4. 各 resource を `[start + offset, start + offset + hold]` で予約

makespan = 全エッジの最大 end。クリティカルパスは **リソース無視の longest dependency chain**（infinite resources 仮定の理論下限）。

### renderer (LLM, opt-in)

スケジュール済み DAG と `RawRecipe`（材料・元手順・コツ）を見せて、家庭料理レシピ調の **磨かれた日本語** 手順を生成する。

- 細粒な mix エッジを束ねる（「醤油+みりん+砂糖を混ぜる → 合わせ調味料を作る」）
- 元レシピから分量を引用（「水1L」「鶏もも肉 1/2枚」）
- 代替適用後の最終ツールを使った文章にする（「片手鍋に水を入れて〜」、「（代替）」注釈は出さない）

`use_renderer=False` のときは `output.to_numbered_steps` の deterministic 出力にフォールバック。

## 4. LLM バックエンド

`LLMClient` ABC を3つ実装:

| バックエンド | 用途 | 速度 | コスト | 構造化出力 |
|---|---|---|---|---|
| `MockLLMClient` | 単体テスト・高速開発 | 即時 | 0 | label→canned response 登録 |
| `LocalLLMClient` (transformers) | bf16 ローカル推論 | 遅い（CPU で 2-3 tok/s） | 0 | プロンプトに schema 注入 |
| `LlamaCppClient` (GGUF) | 量子化ローカル推論 | 中（CPU で 5-15 tok/s） | 0 | **grammar 制約**（response_format で sampling 時に強制） |
| `ClaudeCliClient` | `claude -p` 経由で Anthropic | 速い（数秒） | $0.15-0.30/呼び出し | プロンプト + JSON 抽出 |

`generate_structured` の `inject_schema: bool` パラメータで「テキストで schema を見せるか」を制御。grammar 制約のあるバックエンド（GGUF）では False を推奨。

### LLM 出力品質の比較（同じ minimal recipe で）

| 項目 | GGUF Q4_K_M + few-shot | Haiku 4.5 (CLI) |
|---|---|---|
| parse 時間 | 51s | **18s** |
| action 選択 | boil_water ✓ | boil_water ✓ |
| `hold_duration_min` | 5.0（cook も burner も同値） | **1.0 vs 4.0**（区別） |
| `name_hint` | None | **「やかん」抽出** |
| `is_final=True` 設定 | 未設定 | ✓ |

→ Q4_K_M は scheduling 可能な構造を出すが、cook の attentive 時間と burner の全体時間を区別できない。Haiku 4.5 は production-grade。

## 5. データ層の運用

### `data/user_profile.json`

```json
{
  "cooks":        [{"id": "cook_self", "kind": "cook", "name": "自分"}],
  "burners":      [{"id": "burner_1", "kind": "burner", "name": "コンロ口1"}, ...],
  "workstations": [...],
  "containers":   [...],
  "appliances":   [{"id": "a_microwave", "kind": "appliance", "name": "電子レンジ"}, ...],
  "utensils":     [...]
}
```

各プールの **長さ = 同時並列可能数**。たとえば burners が2つなら、スケジューラは2件まで同時に burner-using なエッジを走らせる。

### `data/tool_use_table.json`

action 別に `ToolOption` 配列。各 option は:

```json
{
  "label": "電子レンジ",
  "resources": [
    {"kind": "cook", "relative_duration": 0.2},
    {"kind": "appliance", "name_hint": "電子レンジ", "relative_duration": 1.0}
  ],
  "time_factor": 1.1,
  "quality_factor": 0.9,
  "source": "seed"
}
```

`source` は `seed` / `llm` / `user_feedback` のいずれか。LLM 発見の追記は `add_entry(table, action, option)` で行い、テーブルは持続するため次回以降は決定論的に同じ候補が出る。

## 6. 限界

本質的に避けられない不確実性（元レシピの曖昧さ・LLM の挙動・代替の semantic gap・スケジューラの greedy 性）は **[limitations.md](limitations.md)** に独立してまとめている。設計上の境界として扱う。

## 7. 既知の課題（実プロダクト化に向けて）

- **#4** proposer のコンテキスト非対応: 直前ステップで使った容器を次ステップでも継続するロジックがない（mugicha 蒸らしステップで「氷水で急冷」が選ばれる遠因）
- **#5** `skill_factors` を scheduler の duration 補正に未反映
- **#7** `UserProfile.preferences` を selector / pipeline で未活用
- **#8** 過剰リソース要求への早期警告（CP の N倍以上に膨れたら警告）
- **#10** 代替不能リソース（冷蔵庫等）の semantic filter（step 2/3: LLM verifier or substitutable flag）

参照: [GitHub Issues](https://github.com/chai0204/recipe-optimizer/issues)

## 8. リレーショナルモデルとしての制約

データ層を Pydantic の階層型ではなく **フラットなリレーションの集合** として見ると、checker と scheduler の論理が SQL（または Datalog）として宣言的に書ける。実装は手続き的だが、不変条件はリレーションレベルで定義されている。

### 8.1 ER スキーマ

```
Resource (id PK, kind, name)
GoalNode (node_id PK, description, is_final)
ProcessEdge (edge_id PK, action, duration_min, to_node FK→GoalNode)
EdgeFromNode (edge_id FK→ProcessEdge, from_node FK→GoalNode,
              PK=(edge_id, from_node))     -- 多対多: エッジは複数の入力ノードを持つ
ResourceRequirement (edge_id FK→ProcessEdge, seq,
                     kind, name_hint, hold_duration_min, start_offset_min,
                     PK=(edge_id, seq))

-- 学習層
ToolOption (action, label, time_factor, quality_factor, source,
            PK=(action, label))
ResourceSpec (action, label, seq, kind, name_hint, relative_duration,
              PK=(action, label, seq),
              FK=(action, label)→ToolOption)

-- スケジューラ出力
ScheduledStep (edge_id PK FK→ProcessEdge, start_min, end_min)
ScheduledAssignment (edge_id FK, resource_id FK→Resource,
                     hold_start_min, hold_end_min,
                     PK=(edge_id, resource_id, hold_start_min))
```

`Resource` と `ResourceRequirement` は「具体物プール」と「需要側のチケット」の関係。`kind+name_hint` のペアが両者をマッチングするキーになる。

### 8.2 checker は単純な反結合

`find_violations` は以下と等価:

```sql
-- 各リソース要求に対し、プールにマッチする Resource がない行を返す
SELECT rr.edge_id, rr.kind, rr.name_hint
FROM ResourceRequirement rr
WHERE NOT EXISTS (
  SELECT 1 FROM Resource r
  WHERE r.kind = rr.kind
    AND (rr.name_hint IS NULL OR r.name = rr.name_hint)
);
```

`name_hint IS NULL` は「kind が同じならどれでも可」のワイルドカード。`e_simmer_onion` の `(burner, null)` 要求は `burner_1` でも `burner_2` でも通る。

### 8.3 substitution は ResourceRequirement の差し替え

`apply_substitution` は SQL では:

```sql
-- 違反エッジの resource_uses を削除
DELETE FROM ResourceRequirement WHERE edge_id = :violated_edge_id;

-- 選ばれた ToolOption の ResourceSpec を resource_uses に展開して挿入
INSERT INTO ResourceRequirement (edge_id, seq, kind, name_hint, hold_duration_min)
SELECT :violated_edge_id, rs.seq, rs.kind, rs.name_hint,
       :new_duration_min * rs.relative_duration
FROM ResourceSpec rs
WHERE rs.action = :action AND rs.label = :chosen_label;

-- duration_min と action は ProcessEdge を UPDATE
UPDATE ProcessEdge
   SET duration_min = :new_duration_min, action = :new_action
 WHERE edge_id = :violated_edge_id;
```

`replacement_edge` がエッジ ID・`from_nodes`・`to_node` を保存することは「PK と外部参照を保存」と同義で、これがトポロジー不変性を保証する。

### 8.4 scheduler の妥当性は2つの NOT EXISTS で言える

任意の `Schedule` は以下を満たさなければならない:

```sql
-- (a) 依存制約: 各エッジは全 predecessor 終了後に開始
ASSERT NOT EXISTS (
  SELECT 1
  FROM ProcessEdge e
  JOIN EdgeFromNode ef    ON ef.edge_id = e.edge_id
  JOIN ProcessEdge pe     ON pe.to_node = ef.from_node
  JOIN ScheduledStep s    ON s.edge_id = e.edge_id
  JOIN ScheduledStep ps   ON ps.edge_id = pe.edge_id
  WHERE s.start_min < ps.end_min
);

-- (b) 資源排他: 同一 Resource は時間区間が重ならない
ASSERT NOT EXISTS (
  SELECT 1
  FROM ScheduledAssignment a1
  JOIN ScheduledAssignment a2
    ON a1.resource_id = a2.resource_id
   AND a1.edge_id <> a2.edge_id
   AND a1.hold_start_min < a2.hold_end_min
   AND a2.hold_start_min < a1.hold_end_min
);
```

makespan の最小化は (a)(b) を満たす中での `MAX(end_min)` 最小化（NP-hard、RCPSP）。実装は greedy で近似する。クリティカルパスは (a) のみで定義（infinite resources の理論下限）:

```sql
-- 各ノードの最早完了時刻を再帰計算
WITH RECURSIVE earliest(node_id, t) AS (
  SELECT node_id, 0.0 FROM GoalNode
   WHERE NOT EXISTS (SELECT 1 FROM ProcessEdge WHERE to_node = node_id)
  UNION ALL
  SELECT e.to_node, MAX(child.t + e.duration_min)
  FROM ProcessEdge e
  JOIN EdgeFromNode ef ON ef.edge_id = e.edge_id
  JOIN earliest child  ON child.node_id = ef.from_node
  GROUP BY e.to_node, e.duration_min
)
SELECT MAX(t) FROM earliest;
```

### 8.5 何が嬉しいか

- **論理と実装の分離**: scheduler は近似アルゴリズムだが、「正しさ」は宣言的に書ける（(a)(b) を満たすこと）。テストでこの不変条件をそのまま assert できる
- **永続化の自然な経路**: PoC は JSON だが、SQLite なら全モデルがそのままテーブルになる。`ToolUseTable` の学習データ蓄積も `INSERT INTO ResourceSpec` で済む
- **新しい問い合わせが SQL で書ける**: 「電子レンジを使ったレシピだけ列挙」「片手鍋を 5 分以上占有するエッジを集計」など、`resource_uses` を1級市民として扱える

[examples/oyakodon.md §裏側の構造](examples/oyakodon.md#裏側の構造テーブル表現) に、実際のレシピでこのリレーションがどう埋まるかの完全な dump がある。

## 9. 設計判断の根拠

これらの判断は [life](https://github.com/chai0204/life)（オーナーの記憶・知識リポ）の以下にも記録:

- `knowledge/llm-as-bounded-module.md`: LLM を引数・戻り値スキーマで縛った関数として使う設計パターンの抽象化
- `works/recipe-optimizer.md`: PoC のメタデータと進捗
- `profile/people/yasuda-kun.md`: 元アイデアの発案者
