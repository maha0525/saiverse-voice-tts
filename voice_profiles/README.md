# voice_profiles/

ペルソナ ID と参照音声(ref wav)の対応表を定義するディレクトリ。
あわせて TTS エンジンの読み方を補正するユーザー読み方辞書 (`pronunciation_dict.json`) もここに置く。

> **アドオン管理 UI からのアップロードを推奨**: SAIVerse のアドオン管理画面 → Voice TTS → ペルソナ別設定 で参照音声 wav と書き起こしをアップロードできます。UI でアップロードした値は本体側のアドオンストレージに保存され、**`registry.json` より UI 側の設定が優先**されます。UI で参照音声を指定せず `registry.json` の声を使う場合でも、UI で既定値から変えた話速などの調整値はその声に重なります (既定値のまま触っていない欄は `registry.json` の値を上書きしません)。本ファイルで記述する方式は CLI で一括管理したい場合や、UI 対応前の環境用のフォールバック手順です。

> **`registry.json` はローカルファイル**: 上流が配布するのは `registry.json.template`(git 管理)で、`registry.json` は `.gitignore` 対象のユーザー編集用です。初回 `setup.bat` 実行時(またはバックエンド初回起動時のローダ)に template から自動生成されます。`git pull` で上流の `.template` が更新されてもローカルの `registry.json` は触られず、衝突しません。新規フィールドを取り込みたい場合は `git diff registry.json.template` を確認して手動でマージしてください。

## ディレクトリ構成

```
voice_profiles/
├── README.md                          ← このファイル
├── registry.json.template             ← 上流配布(git 管理)
├── registry.json                      ← ユーザー編集ローカル(.gitignore)
├── pronunciation_dict.json.template   ← 上流配布(git 管理)、読み方辞書のひな形
├── pronunciation_dict.json            ← ユーザー編集ローカル(.gitignore)
└── samples/
    ├── _default/
    │   └── ref.wav      ← デフォルト参照音声(どのペルソナIDにも無い時のフォールバック)
    ├── <persona_id_1>/
    │   └── ref.wav
    └── <persona_id_2>/
        └── ref.wav
```

## registry.json の書き方

```json
{
    "_default": {
        "engine": "gpt_sovits",
        "ref_audio": "samples/_default/ref.wav",
        "ref_text": "参照音声の正確な書き起こしテキスト。",
        "params": {
            "speed": 1.0,
            "temperature": 1.0,
            "top_k": 15,
            "top_p": 1.0,
            "text_split_method": "cut5"
        }
    },
    "Yui_city_a": {
        "engine": "gpt_sovits",
        "ref_audio": "samples/Yui_city_a/ref.wav",
        "ref_text": "結衣ペルソナの参照音声の書き起こし。",
        "params": {
            "speed": 1.05
        }
    },
    "Eris_city_a": {
        "engine": "irodori",
        "ref_audio": "samples/Eris_city_a/ref.wav",
        "ref_text": "",
        "params": {
            "num_steps": 32,
            "seed": 42
        }
    }
}
```

### フィールド

| フィールド | 必須 | 説明 |
|---|---|---|
| `engine` | 必須 | `gpt_sovits`(既定) / `irodori` |
| `ref_audio` | ローカル合成で必須 | `samples/` 以下の相対パス(絶対パスも可)。外部 GPT-SoVITS にペルソナ ID のプロファイルが登録済みなら省略可能 |
| `ref_text` | エンジン依存 | gpt_sovits は必須(wav の**正確な**書き起こし)。irodori は無視されるので空でよい |
| `params` | 任意 | エンジン別の生成パラメータ(下記参照) |

### `params` に指定できる項目(エンジン別)

#### gpt_sovits

外部サーバー URL はアドオン管理 UI、または `config/default.json` の
`engines.gpt_sovits.gpt_sovits_remote_url` に設定します。
通信タイムアウトは同階層の `gpt_sovits_remote_timeout_sec` (既定 15 秒) です。
UI に URL を入力した場合は UI が優先されます (UI が空欄なら設定ファイルの URL を使います)。
設定ファイルの URL だけで外部サーバーを使う場合、参照音声 (`ref_audio`) は省略できません。外部サーバーに送る合成引数は
ペルソナ ID とテキストのみで、下表のローカル推論パラメータは送信しません。

| キー | 既定値 | 説明 |
|---|---|---|
| `speed` | 1.0 | 話速(0.5〜2.0) |
| `top_k` | 15 | サンプリング |
| `top_p` | 1.0 | サンプリング |
| `temperature` | 1.0 | ランダム性 |
| `text_split_method` | `cut5` | テキスト分割方式(`cut1` `cut2` `cut3` `cut4` `cut5`) |
| `overlap_length` | 2 | ストリーミング時のチャンク間オーバーラップ |
| `min_chunk_length` | 16 | ストリーミング最小チャンク長(semantic token 数) |
| `fixed_length_chunk` | false | 固定長チャンク(true で速度優先・品質低下) |

#### irodori
| キー | 既定値 | 説明 |
|---|---|---|
| `num_steps` | 24 | 拡散ステップ数。32 で高品質、16 以下で速度優先 |
| `seed` | (ランダム) | 同じ入力で同じ出力を得たい場合に固定 |
| `cfg_scale_text` | 3.0 | テキストへの忠実度(上げると丁寧だが硬くなる) |
| `cfg_scale_speaker` | 5.0 | 話者特徴への忠実度 |
| `truncation_factor` | 0.75 | ゴミ音声抑制のコア。**通常変更非推奨**(下げると途切れ、上げるとゴミ再発) |
| `seconds` | 自動算出 | チャンクごとに文字数から推定。指定すると上書き |

> Irodori の ref_text 欄は無視されるので空でよい(上流 API が ref_wav からのみ話者特徴を抽出する)。
>
> `ref_audio` は GPT-SoVITS と Irodori で同じ wav を使い回せる。要件は「3〜10 秒、日本語の肉声、背景雑音なし」で共通。

## 参照音声の準備ガイド

### 推奨スペック
- **長さ**: 3秒以上10秒以内（必須。範囲外は合成品質が著しく低下する）
- **サンプリングレート**: 16kHz 以上、24kHz/48kHz 推奨
- **チャンネル**: モノラル(ステレオは自動で mono に落ちる場合あり)
- **形式**: wav(PCM 16bit / 24bit / 32bit float)
- **内容**:
  - 目標の話者の素のトーンでの日本語発話
  - 背景雑音・BGM なし
  - 口の中の音(リップノイズ)少なめ
  - 感情の起伏が極端でない(フラット寄り推奨、感情は合成時のテキストと参照音声の相互作用で乗る)

### 書き起こしの書き方
- **正確に**: 言い間違い・助詞の脱落・語尾の曖昧さまで含めて聞こえた通りに
- **句読点を含める**: 音声のポーズに合わせて `、` `。` を入れる
- 英単語や固有名詞は原語で(「Claude」→「クロード」等、実際に音声で言っている通りに)

### 既存音源からの切り出し例(ffmpeg)

```bash
# 元動画から 8秒切り出し、24kHz mono wav に変換（3〜10秒の範囲内で）
ffmpeg -i source.mp4 -ss 00:01:23 -t 8 -ac 1 -ar 24000 samples/Yui_city_a/ref.wav
```

## ペルソナ ID の確認方法

SAIVerse の DB からペルソナ一覧を取得:

```python
import sqlite3
conn = sqlite3.connect(r'C:\Users\<user>\.saiverse\user_data\database\saiverse.db')
for row in conn.execute('SELECT PERSONA_ID, DISPLAY_NAME FROM ai'):
    print(row)
```

表示される `PERSONA_ID`(例: `Yui_city_a`)を `registry.json` のキーに使用。

## 動作確認

プロファイル追加後、SAIVerse を再起動して該当ペルソナに話しかける。正しく認識されていれば以下のログが出る:

**gpt_sovits**:
```
speak_as_persona enqueued: persona=Yui_city_a job=...
Loading GPT-SoVITS TTS pipeline     ← 初回のみ
TTS first chunk ready after 0.53s (job=...)
TTS streamed wav saved: ...\voice\out\<job_id>.wav
```

**irodori**:
```
speak_as_persona enqueued: persona=Eris_city_a job=...
Irodori-TTS checkpoint resolved: hf://Aratako/Irodori-TTS-500M-v2 -> ...
Loading Irodori-TTS runtime: Aratako/Irodori-TTS-500M-v2 (device=cuda precision=bf16)   ← 初回のみ
TTS first chunk ready after 1.44s (job=...)
TTS streamed wav saved: ...\voice\out\<job_id>.wav
```

`No voice profile for persona_id=Xxx (and no _default); skipping TTS.` が出たら registry.json のキー名を再確認(ペルソナ ID と完全一致が必要)。

## よくある失敗

- `ref_text` が wav の実際の発声と微妙に違う → 合成音声が不安定になる / 違う話者に聞こえる
- `ref_audio` のパスタイポ → `_default` にフォールバックするので気付きにくい。ログで `No voice profile` が出ていなければ読み込めている
- ステレオ wav の片チャンネルのみ使用される → mono に変換を推奨
- サンプリングレート不一致 → GPT-SoVITS は内部で resample するので問題ないが、24kHz/32kHz 推奨

---

## ユーザー読み方辞書 (`pronunciation_dict.json`)

TTS エンジンが固有名詞・専門語を誤読する場合、辞書ファイルで置換ルールを書ける。
TTS エンジンに渡す**直前**にテキストを文字列置換する仕組みで、チャット UI 表示や
SAIMemory 保存テキストには影響しない (TTS 専用フィルタ)。

### 典型的な使用例

GPT-SoVITS の MeCab 解析が「は」を助詞として誤判定して `wa` 読みしてしまうケース:

```json
{
    "_comment": "「まはー」が wa 読みされる問題への対処",
    "まはー": "マハー"
}
```

カタカナ「ハ」は助詞解釈されないため、結果として `mahā ≈ mahaa` で読まれる。

### フォーマット

```json
{
    "_comment_anything_starting_with_underscore": "コメント扱い、置換対象にならない",
    "誤読される語": "期待する読み方の表記",
    "もう一つ": "another"
}
```

- key: 誤読される元の文字列(完全一致で部分文字列置換)
- value: 置き換え後の表記(TTS エンジンに渡される)
- `_` で始まるキーはコメントとして無視される
- キーが長い順に適用される(部分一致による意図しない置換を防止)

### ファイル管理

- `pronunciation_dict.json.template` が上流配布(git 管理)
- `pronunciation_dict.json` がユーザー編集用(`.gitignore`)
- 初回起動時にローカル版が無ければ template から自動コピー
- `git pull` でユーザー編集が衝突しない仕組み(`registry.json` と同じ方式)

### グローバル辞書の編集方法

「全ペルソナ共通」の辞書は次の 2 ソースをマージして使われる:

#### 1. 推奨: アドオン管理 UI から編集

SAIVerse の **アドオン管理 → Voice TTS → 読み方辞書 (全ペルソナ共通)**
で「キー (誤読される語) / 値 (読ませたい表記)」を直接追加・編集できる。
変更は即座に DB に保存され、再起動なしで次回の発話から適用される
(UI 辞書はキャッシュせず毎回 fresh 取得する設計のため)。

> 本体側に `dict` 型 params のサポートが入っている必要あり (本体
> `feature/addon-dict-param-type` のマージ後)。未対応バージョンの本体では
> 「（未対応の型: dict）」と表示されるので、その場合はファイル辞書だけで運用。

#### 2. 代替: ファイル辞書 (`pronunciation_dict.json`) を直接編集

CLI で一括管理したい場合や、UI 対応前のバージョンの本体を使う場合は
このファイルを編集する。1 と同じ JSON フォーマット。

#### マージ規則

- 同一キーが両方にあれば **UI が優先**
- それぞれにしか無いキーは両方適用される (補完関係)

### ペルソナ別オーバーライド (registry.json)

特定ペルソナだけ別の読み方を持たせたい場合、`registry.json` の該当ペルソナ
エントリに `pronunciation_dict` キーを追加:

```json
{
    "Eris_city_a": {
        "engine": "irodori",
        "ref_audio": "samples/Eris_city_a/ref.wav",
        "ref_text": "",
        "params": {"num_steps": 32},
        "pronunciation_dict": {
            "ナチュレ": "なつる"
        }
    }
}
```

### 適用順序 (高→低)

1. ペルソナ別 (registry.json[<persona>].pronunciation_dict)
2. グローバル (UI 辞書 ∪ ファイル辞書、UI が同一キーで優先)
3. そのまま (置換ルールに該当しなければ)

### 注意点

- 平文の部分文字列置換なので、文書中の「は(助詞)」を全て置換するような書き方は避ける(固有名詞単位で登録)
- regex は非対応(v1)
- 実際の TTS 出力は MeCab 解析結果次第で変わるため、辞書を編集→再起動→聴いて確認、というイテレーションで調整する
