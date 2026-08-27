# Changelog

日付は ISO 8601 形式(JST)。バージョン採番はまだ付与していないため、日付とマージコミットハッシュで識別。

## [Unreleased]

### 音声ストリーム配信の待ちがホストの終了を止める問題を修正

- `api_routes.py` の `_stream_body` が、consumer queue を無期限の `q.get()`
  で待つ形のまま `run_in_executor` に渡していた。この worker thread は
  「作業中」のまま executor へ戻らないので、二つが同時に壊れる。
  - クライアントが切断してリクエストが cancel されても、待っている thread は
    残る (別 thread で走る同期処理に asyncio の cancel は届かない)。
  - インタプリタ終了時、`concurrent.futures` が `atexit` **より前** に走らせる
    `_python_exit` がその thread を join し続けるため、ホストが `atexit` に
    登録した後始末に到達できず、プロセスが終了しない。
- `close_stream` まで到達しなかったストリーム — 合成が始まらないまま subscribe
  された placeholder (subscribe-before-open) や、途中で失敗した合成 — が 1 本でも
  残れば終了印は永久に来ないので、これは「いつか起きる」ではなく起きる。
  2026-08-27、SAIVerse バックエンドが Ctrl+C で終了せず、この待ちが 3 本
  残っていた (py-spy で MainThread が `_WorkItem.run` 内の `queue.get` で寝る
  worker を join しているところを確認)。
- 待ちを 0.5 秒ずつに刻み、worker thread が毎周 executor へ返るようにした。
  `playback_worker` の job queue が既に取っている形と同じ。停止シグナルも
  cancel も届くようになる。
- 回帰テスト `tests/test_stream_body.py` を追加。旧実装ではこのテストの実行中に
  pytest 自身が終了できなくなる。

### 再生成ボタンを常時押せるようにする

- `addon.json`: `regenerate_audio` バブルボタンの `show_when` を
  `metadata_exists` から `always` に変更。
  - 旧仕様だと `audio_path` メタデータが無いメッセージ
    (TTS 合成失敗、auto_speak OFF 時、アドオン無効中の発話など) では
    再生成ボタンも pending スピナー扱い (disabled) になり永遠に触れない
    バグがあった。
  - メタデータが無いことは「再生成できない理由」にはならないので、
    `regenerate_audio` は常に visible とする。`play_audio` は引き続き
    `metadata_exists` (鳴らす音声が無いと意味が無いため)。

### v0.5.0: クラウド TTS エンジン (OpenAI TTS / ElevenLabs) に対応 (experiment/openai-elevenlabs-engines)

GPU を持たないユーザー向けに、API 経由で TTS を行う 2 エンジンを追加。
既存の GPT-SoVITS / Irodori-TTS と同じ抽象 (TTSEngine ABC) で並列に切替
可能。

#### 新規エンジン

**openai_tts** (`tools/speak/engine/openai_tts.py`)
- POST `https://api.openai.com/v1/audio/speech` を呼ぶ。
- ボイスクローンは無し、preset 9 voices から選択
  (alloy / echo / fable / onyx / nova / shimmer / ash / sage / coral)。
  ペルソナごとに addon UI のドロップダウンで指定。
- `response_format=pcm` (16-bit signed @ 24kHz raw PCM) を使用しデコード
  不要・即時 numpy 化。50 ms ぶんバッファして SynthesisChunk に変換。
- ストリーミング対応 (HTTP chunked)。
- `model`: `tts-1` (既定) / `tts-1-hd` / `gpt-4o-mini-tts`。
- `gpt-4o-mini-tts` のとき `instructions` でスタイル指示文を渡せる。
- ref_audio は無視 (preset voices なので不要)。

**elevenlabs** (`tools/speak/engine/elevenlabs.py`)
- POST `/v1/text-to-speech/{voice_id}/stream` を呼ぶ。
- ゼロショットボイスクローンを使えるが、v1 では voice_id をユーザーが
  ElevenLabs ダッシュボード (Voice Lab → Instant Voice Cloning) で
  作成 → addon UI に貼り付ける運用。自動クローンは v2 で検討。
- `output_format=pcm_24000` で受け取り numpy 化。
- ストリーミング対応。
- `model_id`: `eleven_turbo_v2_5` (既定) / `eleven_multilingual_v2` 等。
- `voice_settings`: stability / similarity_boost / style / use_speaker_boost
  をペルソナ別 params で調整可能 (範囲外値は既定値にフォールバック)。

#### API key 解決順位

1. addon UI param (`openai_api_key` / `elevenlabs_api_key`) — 最優先、UI 編集が即反映
2. `config/default.json` の `engines.<name>.api_key` (legacy フォールバック)
3. 環境変数 (`OPENAI_API_KEY` / `ELEVENLABS_API_KEY`)

OpenAI は本体既存の `OPENAI_API_KEY` 設定をそのまま流用できる。

#### エラー処理

- API key 未設定 / voice_id 未設定 → 即 `RuntimeError`
- 4xx (auth / 不正リクエスト 等) → 1 回でログ出力 + 例外
- 429 / 5xx → 1 回だけ短い backoff (1.5s) でリトライ
- ネットワークエラー → 1 回リトライ後に例外

#### addon.json 拡張

- `engine` ドロップダウンに `openai_tts` / `elevenlabs` を追加
- ペルソナ別: `openai_voice` (dropdown), `elevenlabs_voice_id` (text)
- グローバル: `openai_api_key` (password), `elevenlabs_api_key` (password)
  両方ともアコーディオンで折り畳み既定

#### profiles.py 改修

- `_API_ENGINES = {"openai_tts", "elevenlabs"}` を導入
- ローカルエンジンは ref_audio 必須 (従来通り)、API エンジンは ref_audio
  不要でプロファイル成立とする (registry.json fallback に飛ばずに済む)
- `_EXCLUDED_KEYS` に `openai_api_key` / `elevenlabs_api_key` を追加
  (engine が addon_config から fresh に解決するので params 経由不要)

#### 依存追加

なし。`httpx` は本体 requirements に既存。`openai` SDK は使わず REST 直叩き。

#### テスト

- `tests/test_engine_openai_tts.py` (16 件): リクエスト body / API key
  解決順位 / ストリーミング集約 / 4xx 即例外 / 5xx リトライ
- `tests/test_engine_elevenlabs.py` (20 件): 同様 + voice_settings 範囲
  バリデーション / output_format フォールバック / voice_id 未設定例外

#### 既知の制限

- ElevenLabs の自動クローン (`ref_audio` を upload して voice_id を自動
  生成) は v2 で対応予定。
- pronunciation_dict は OpenAI / ElevenLabs どちらにも適用される (合成
  直前の文字列置換なので engine 非依存)。
- バブル「音声を再生成」ボタンも問題なく動作 (engine が変わっても
  metadata 更新フローは共通)。

### v0.4.0: server_hooks 経由の TTS 発火に移行 (feature/voice-tts-speak-hook)

SAIVerse 本体の認知モデル Phase C-2/3 (2026-05-01) で発話経路が Track ベースに
刷新され、旧 `sub_speak` Playbook が共通経路でなくなった。`expansion_data/<addon>/playbooks/public/sub_speak.json`
で本体を上書きして末尾に `tts_speak` ノードを足す方式は新 Track Playbook
(`track_user_conversation` / `track_external` 等) を経由しないため、デフォルトの
ユーザー会話で TTS が完全に無音になっていた。

本体側に追加された **`server_hooks`** 機構経由で `persona_speak` イベントを
購読する形に切り替え、Playbook 構造に依存せず発話の最終共通経路 (`emit_speak` /
`emit_say`) から直接呼ばれるようにした。

#### 変更点

- **新規**: `speak_hook.py` — `on_persona_speak(persona_id, text_for_voice, message_id, **kwargs)`
  ハンドラ。本体の `<in_heart>` 除去・spell ブロック処理済みテキストを受け取り、
  `clean_text_for_tts()` を通して `enqueue_tts()` に投入する
- **`addon.json`**: `server_hooks` セクションを追加。`event: "persona_speak"` に
  `handler: "speak_hook:on_persona_speak"` を宣言
- **削除**: `playbooks/public/sub_speak.json` — 旧 override は撤去。残すと旧
  `meta_simple_speak` 経路で本体 hook と override の両方から TTS が呼ばれ
  二重発火する
- `tools/speak/schema.py` の `speak_as_persona` ツール自体は維持 (将来的な
  `/spell speak_as_persona` での明示発話制御に備える)
- `version`: `0.3.0` → `0.4.0`

#### 必要な本体バージョン

- `saiverse/addon_hooks.py` および `addon.json` の `server_hooks` セクション対応が
  入った本体ビルド (feature/voice-tts-speak-hook ブランチ以降)
- 旧本体で 0.4.0 を動かすと、`server_hooks` 宣言は無視され TTS が動かなくなる

### ユーザー読み方辞書(発音上書き)を追加(feature/pronunciation-dict)

GPT-SoVITS / Irodori-TTS の g2p (grapheme-to-phoneme) が固有名詞・専門語を
誤読する場合、ユーザーが**辞書ファイルで置換ルールを書ける**仕組みを追加。

例: 「まはー」が MeCab 解析で「ま+は(助詞)+ー」と誤分割されて `mawaa` 読み
されてしまう問題に対し、辞書で「まはー → マハー」と書けばカタカナの「ハ」
は助詞解釈されないため `mahā` ≈ `mahaa` と読まれる。

#### 配置 (テンプレート方式、ユーザーローカル)

- `voice_profiles/pronunciation_dict.json.template` (上流配布、git 管理)
- `voice_profiles/pronunciation_dict.json` (`.gitignore`、初回起動時に自動コピー)

#### 動作

1. ペルソナ応答テキストを TTS エンジンに渡す**直前**に文字列置換
2. チャット UI 表示や SAIMemory 保存テキストには影響しない (TTS 専用フィルタ)
3. 適用順: **ペルソナ別 override (registry の `pronunciation_dict`) → グローバル**
4. キーは長い順に適用 (部分一致による意図しない置換を防止)
5. `_` で始まるキーはコメント扱いで無視

#### 辞書フォーマット例

```json
{
    "_comment": "コメントは _ で始めれば無視される",
    "まはー": "マハー",
    "SAIVerse": "サイバース"
}
```

#### グローバル辞書を 2 ソースから合成 (推奨: アドオン管理 UI)

SAIVerse 本体の **アドオン管理 → Voice TTS → 読み方辞書 (全ペルソナ共通)**
で key/value を直接追加・編集できる。本体側に `dict` 型の params_schema が
入った後 (本体 `feature/addon-dict-param-type` のマージが必要) のバージョン
から有効。

「全ペルソナ共通」のグローバル辞書は次の 2 ソースのマージ結果:

1. UI で編集した辞書 (DB 保存、GUI 編集向け、再起動なしで反映)
2. `voice_profiles/pronunciation_dict.json` (ファイル辞書、CLI 編集向け)

同一キーがある場合は **UI > ファイル** で UI 値が優先される。

#### ペルソナ別オーバーライド (registry.json)

特定ペルソナだけ別の読み方を持たせたい場合は従来通り `registry.json` の
ペルソナエントリに `pronunciation_dict` キーを書く:

```json
{
    "Eris_city_a": {
        "engine": "irodori",
        "ref_audio": "...",
        "params": {...},
        "pronunciation_dict": {
            "ナチュレ": "なつる"
        }
    }
}
```

#### 適用順序 (高→低)

1. ペルソナ別 (registry.json[<persona>].pronunciation_dict)
2. グローバル (UI ∪ file、UI が同一キーで優先)
3. そのまま (置換ルールに該当しなければ)

#### 変更ファイル

- `tools/speak/pronunciation_dict.py` (新規): 辞書ローダ + apply 関数。
  ファイル辞書はキャッシュし、UI 辞書は呼び出しごとに DB から fresh 取得
  (UI 編集即時反映)
- `tools/speak/playback_worker.py`: engine 呼び出し前に apply() を仕込む
- `tools/speak/profiles.py`: AddonPersonaConfig からの引き上げ対象から
  `pronunciation_dict` を除外 (UI ではグローバル設定なので個別 profile に
  載せない)
- `addon.json`: `pronunciation_dict` (type=dict, persona_configurable=false)
  を全ペルソナ共通設定として追加
- `voice_profiles/pronunciation_dict.json.template` (新規)
- `.gitignore`: ローカル辞書を追記
- `tests/test_pronunciation_dict.py` (新規): 23 件のユニットテスト
  (apply 基本 15 件 + UI ∪ file マージ 8 件)

#### 既知の限界

- 平文置換のみ(regex 非対応、必要なら v2 で検討)
- 置換結果が TTS エンジンで意図通り読まれるかは MeCab 解析次第。実際に
  聴いて確認しながら辞書を調整する想定
- 文書中の「は」を全て置換すると助詞「は」も置換されてしまう。固有名詞単位
  で登録するのが基本

### ユーザー編集ファイルをテンプレート方式に変更(feature/template-based-user-config)

`config/default.json` と `voice_profiles/registry.json` は **ユーザーが手元で編集する**前提のファイルだが、これらが git 管理下にあったため `git pull` 時に上流の更新と衝突し、毎回 `git stash → pull → stash pop` が必要だった。

#### 変更点

- `config/default.json` を `config/default.json.template` にリネーム(git 管理対象)
- `voice_profiles/registry.json` を `voice_profiles/registry.json.template` にリネーム
- `.gitignore` にローカル版(`config/default.json`、`voice_profiles/registry.json`)を追加
- ローダ(`playback_worker._load_config` / `profiles._load_registry`)を改修:
  - ローカル版が無ければ `.template` から**初回コピー**(first-run materialization)
  - 以降はローカル版を読む
  - 上流(`.template`)が更新されても**ローカル版は触られない**
- `setup.bat` にも明示的な first-run コピーステップを追加(冗長だが透明性のため)

#### 既存ユーザー向けマイグレーション(本コミットを pull する前に必須)

ローカルで `config/default.json` または `voice_profiles/registry.json` を編集している場合、そのまま `git pull` するとマージ衝突します。手順:

```bash
# 1. ローカル編集を退避
cp config/default.json /tmp/my_default.json
cp voice_profiles/registry.json /tmp/my_registry.json

# 2. 追跡ファイルを上流バージョンに戻す
git checkout HEAD -- config/default.json voice_profiles/registry.json

# 3. pull(衝突なく完了する)
git pull

# 4. 退避させた編集を戻す(これらは untracked になる)
cp /tmp/my_default.json config/default.json
cp /tmp/my_registry.json voice_profiles/registry.json
```

以降は `git pull` で衝突しません。`.template` が更新されても `git diff config/default.json.template` で差分を確認して手動で取り込めます。

#### 新規ユーザー(初回 setup.bat 実行)

何も意識しないで OK。setup.bat が `.template` から自動コピーします。

### `_shadowed_tools_namespace` hack を削除(chore/remove-tools-shadow-hack)

ホスト側に `addon_external_loader` が導入されたことで、各パックが
``sys.modules['tools']`` を一時剥がしする hack が不要になった。`gpt_sovits.py`
から以下を削除:

- `_shadowed_tools_namespace()` コンテキストマネージャ関数(20 行)
- TTS 初期化箇所での `with _cwd(_EXTERNAL_REPO), _shadowed_tools_namespace():`
  → `with _cwd(_EXTERNAL_REPO):`(`_cwd` は cwd 切替が必要なので残す)

**前提**: SAIVerse 本体側に `addon_external_loader` が入っていること
(ホスト 2026-04 以降のブランチ)。本機構が無いホストで本パックを動かすと、
GPT-SoVITS ロード時に本体側 `tools` パッケージとの名前衝突で `ImportError`
が発生する。古いホストで動かしたい場合は本コミット以前の版を使うこと。

**影響**: 並列スレッドで TTS ロード中にホスト側 import が走った場合の
名前空間汚染が原理的に発生しなくなる(従来 hack の根本欠陥を解消)。
Irodori-TTS には同種の hack は無いので変更なし。

### 再生系トグル/デバイスのラベル調整(docs/irodori-integration)

アドオン管理 UI でユーザーが項目の意図を誤認しないよう、ラベルとドキュメントを整理:

- 「クライアント側再生」→「**ブラウザ側再生**」(実質的な実装内容で呼称)
- 「出力オーディオデバイス」→「**サーバー側再生デバイス**」(サーバー側再生専用でブラウザ側には影響しないことを明示)
- README / SETUP / TROUBLESHOOTING / ARCHITECTURE の該当箇所を新ラベルに揃え、Irodori-TTS の導入手順を README に独立セクションとして追加。

### Irodori-TTS を本番利用可能に(experiment/irodori-tts)

実験的サポートのまま放置されていた Irodori-TTS バックエンドを、動作検証 + 速度最適化 + 品質改善まで進めて本番投入可能な状態にした。

- **接続修正**: `irodori.py` の import を上流の実モジュール `irodori_tts.inference_runtime` に合わせ、HF repo ID を `hf_hub_download` で解決、`SamplingRequest` のパラメータを registry の `params` から透過的に渡す構成に。
- **速度改善(RTF 5.65x → 0.27x)**: プロファイルで `decode_latent`(DACVAE の decode)が 16 秒と支配的だったため、codec を CPU → CUDA に移し、model/codec を両方 bf16 に揃えて dtype mismatch を解消。
- **疑似ストリーミング**: 上流 API は一括合成のみだが、アダプタ側で文単位チャンキング + 読点再分割 + 短ポーズ挿入を実装し、`supports_streaming=True` を有効化。
  - first_sound: 約 1.4 秒 (GPT-SoVITS ストリーミングに近い体感)
  - playback_worker の既存ストリーミング経路 (`sd.OutputStream` 逐次再生 + MP3 pub/sub クライアント配信) がそのまま動作
- **ゴミ音声ゼロ化**: Irodori が `seconds` 予算を埋めるために出していた幻覚的な破綻音声を `SamplingRequest.truncation_factor=0.75` で抑制。ASR(Whisper small)検証で 4分間のテキストを通しても**意味不明セグメント 0 件**・本文途切れ 0 件。
- **UI**: addon.json の params_schema に `engine` ドロップダウンを追加。ペルソナ別に `gpt_sovits` / `irodori` を切替可能。ref_text は irodori では使われない旨を description に明記。
- **依存追加**: `torchcodec>=0.10`(torchaudio.load が torch>=2.10 系で要求)、`lameenc>=1.5`(MP3 progressive)。
- **設定**: `config/default.json` の `engines.irodori` を新 API 準拠(`model_precision=bf16`, `codec_device=cuda` 等)に更新。

### アドオン管理 UI のペルソナ別設定をプルダウン方式に変更(feature/addon-persona-selector-ui)

ペルソナ数が増えるとアコーディオン並列表示が縦に伸びすぎて操作しづらかったため、プルダウンで対象ペルソナを選んで設定する方式に変更(案C)。

- ペルソナごとのアコーディオンを廃止し、プルダウン選択 + 選択ペルソナの設定フォームのみ表示
- プルダウン横の削除ボタンを撤去(誤操作防止、削除は本体のペルソナ管理画面から)
- 共通設定に「出力オーディオデバイス」ドロップダウンを追加(`GET /audio-devices` から取得)
- 「サーバー側再生」の既定値を **OFF** に変更(Tailscale/リモート運用が既定想定になったため)
- 「クライアント側再生」の既定値は **ON**

### 長時間音声再生の途中停止問題を解消(fix/audio-range-200)

2〜3 分の発話で `/stream` または `/audio` が途中で切れる不具合を修正。

- `/audio` GET: Range ヘッダを剥がして 200 OK で全量返す挙動に変更(Chrome の progressive 再生時に Content-Length 検証で切断する問題を回避)
- `/stream` GET: Next.js Route Handler を arrayBuffer 経由から pump 転送に変更(長時間合成で HTTP タイムアウトする問題を回避)
- 同時にブラウザ側バッファ展開で途中停止を救済

### クライアント側再生 + MP3 progressive 配信(feature/client-playback-actions)

Tailscale 越しのリモートブラウザから音声が鳴るようになり、モバイル運用に対応。

- **汎用 `ui_extensions.client_actions` 機構**を拡張パック基盤に追加(本体側):
  - `addon.json` で SSE イベント → クライアント側 JS executor を宣言的に接続
  - Voice TTS 専用ではなく、他のパックからも利用可能
  - `requires_active_tab` / `requires_enabled_param` / `on_failure_endpoint` を宣言で制御
- **`play_audio` action executor**(本体側): 宣言ベースで `<audio>` 要素を制御、primary/fallback URL、iOS Safari autoplay unlock、再生トークンによる連続発話対応
- **アクティブクライアントタブ自動判定**(本体側): BroadcastChannel + 最後の操作時刻で、複数タブ/端末間で「最後に触った端末のみが鳴る」動作(ヘッダーの Radio アイコンで可視化)
- **MP3 progressive 配信に変更**:
  - `tools/speak/audio_stream.py` を FIFO Queue から pub/sub パターンに刷新
  - `lameenc` で PCM → MP3 エンコード(Windows でも追加システム依存不要)
  - 複数コンシューマ同時接続時のチャンク欠落を解消
  - 新規 subscriber に既存フレームを seed で先入れ
  - iOS Safari が WAV の 0xFFFFFFFF ヘッダを拒否する問題を回避
- **Route Handler `/api/addon/[...path]`**(本体側): `/stream` を arrayBuffer バイパスで pump 転送、音声/動画 GET は Range を剥がす
- `addon.json` に `ui_extensions.client_actions` 宣言と `client_side_playback` トグルを追加
- `api_routes.py` に `POST /client_action_failed` を追加(フロント再生失敗時のテレメトリ)
- `requirements.txt` に `lameenc>=1.5` を追加

**既知の制限**: Next.js dev mode では iOS Safari がタブを discard するため、モバイル運用時は `npm run build && npm run start` が必須。

### バブル再生エンドポイント正常化とログの production 整理(experiment/addon-config-consume)

End-to-end で動作確認した結果判明した2点の修正を投入:

- `audio_path` メタデータキーは URL 格納専用とし、バックエンド配信用のファイルシステムパスは新規に `audio_file` キーに分離。従来は同じキーを URL とファイルパスで兼用していたため `api_routes.py` 側で 404 になっていた
- `FileResponse` の `content_disposition_type="inline"` を明示。デフォルト `attachment` ではブラウザが `<audio>` でインライン再生できなかった
- 診断用の INFO ログを DEBUG に格下げし、通常運用時のログを静粛化。`message_id=None` による連携失敗のみ WARNING に格上げ

### アドオン UI トグルを実際に尊重するよう接続(experiment/addon-config-consume)

SAIVerse 本体に `saiverse.addon_config.get_params` / `is_addon_enabled`(コミット `56c344a`)が追加されたため、拡張パック側で読み取って反映するようにした。

- `tools/speak/playback_worker.py`:
  - `_get_effective_params(persona_id)` を新設。本体 `saiverse.addon_config.get_params` を優先し、未提供時は `config/default.json` の既存値にフォールバック
  - `_process()` は config ではなく effective params を見て `streaming` / `server_side_playback` を判定
  - 有効 params のデバッグログを追加
  - `get_effective_params` を公開関数として export
- `tools/speak/schema.py`:
  - enqueue 前に `_enabled` / `auto_speak` をチェックし、OFF のときは即 return
  - 戻り値の `status` に `skipped_addon_disabled` / `skipped_auto_speak_off` を追加

これで UI の「自動発話」「サーバー側再生」「ストリーミング推論」トグルおよびペルソナ別上書きがランタイムに反映される。本体 API が無い環境では従来通り `config/default.json` の値で動作。

### SAIVerse 本体アドオン基盤への連携(experiment/addon-integration)

SAIVerse 本体の `feature/memory-notes-and-organize` ブランチで実装されたアドオン基盤(`ab22842`)に対応。

- `addon.json` 新規作成(マニフェスト):
  - `name` / `display_name` / `description` / `version`
  - `params_schema`: `auto_speak` (persona設定可)、`server_side_playback`、`streaming` の3つのトグル
  - `ui_extensions.bubble_buttons`: フロント側がメッセージバブル内に「音声を再生」ボタンを自動追加。`metadata_key=audio_path` が存在するときのみ表示
- `api_routes.py` 新規作成(旧 `api_routes.py.stub` を削除):
  - `GET /audio/{message_id}` 完成wavを `FileResponse` で配信
  - `GET /audio/{message_id}/stream` 合成進行中のチャンクを HTTP Chunked Transfer で配信
  - 認証は本体 `saiverse.addon_deps.get_manager` を `Depends()` で差し込み
- `tools/speak/audio_stream.py` 新規作成:
  - スレッドセーフな FIFO Queue レジストリ
  - `open_stream` / `push_chunk` / `close_stream` API
  - 0xFFFFFFFF サイズヘッダの WAV プレフィックスでブラウザ逐次再生可
- `tools/speak/playback_worker.py` 改修:
  - `_get_active_message_id()` で `tools.context.get_active_message_id` を取得
  - 合成完了時に `saiverse.addon_metadata.set_metadata` と `saiverse.addon_events.emit_addon_event` を呼び出し
  - `_Job` に `message_id` フィールド追加(enqueue 時点で capture)
  - `_play_streaming` は `audio_stream.open_stream/push_chunk/close_stream` を並行呼び出しし、同じ音声をサーバ再生と HTTP 配信の両方に流す
  - `server_side_playback=False` のときサーバ側再生をスキップ(HTTP 配信のみ)
- `config/default.json` に `server_side_playback: true` を追加

本体側の `addon_metadata` / `addon_events` / `addon_deps` モジュールが未提供の場合は全て警告ログのみで安全に無効化される(下位互換)。

### Qwen3-TTS エンジン削除

ストリーミング非対応で話し始めに 30 秒程度かかり実用性に欠けるため、メインエンジンから外しました。

- `tools/speak/engine/qwen3_tts.py` 削除
- `tools/speak/engine/__init__.py` から qwen3_tts ディスパッチ削除
- `scripts/install_backends.py` から qwen3_tts spec 削除、既定を gpt_sovits に
- `config/default.json` から qwen3_tts 設定ブロック削除、`default_engine` を gpt_sovits に
- `playback_worker.py` の既定エンジンフォールバックを gpt_sovits に
- README / SETUP / ARCHITECTURE / voice_profiles/README / requirements / setup.bat の Qwen3-TTS 記述を整理
- 履歴は残っているので、必要な場合は過去コミット `ca93ffb` 以前を参照可能

### ドキュメント整備

- README / SETUP / ARCHITECTURE / CHANGELOG を整備
- Windows 向けワンクリックセットアップ `setup.bat` を追加
- `install_backends.py` に opencc 自動除外、pre_install_pip、extra_dirs フィールドを追加
  (Windows で GPT-SoVITS セットアップが opencc ビルド失敗で止まる問題を解消)
- `requirements.txt` に soundfile を追加

## 2026-04-16 — ストリーミング合成と Markdown 除去

Merge: `44a7d42`
Feature: `fcd4e55`

発話テキスト → 音声再生までのレイテンシを大幅に短縮。

### 新機能
- GPT-SoVITS のストリーミング推論(`streaming_mode=True, parallel_infer=False`)に対応
- チャンク単位で即時再生(`sd.OutputStream` ベース)
- ストリーミング失敗時は非ストリーミング合成に自動フォールバック
- ペルソナ発話内の Markdown リンク・画像・強調・見出し・`saiverse://` URI などを TTS 前に除去する `text_cleaner.py` を追加

### 性能
- 初回発話(モデル未ロード): 話し始めまで 25〜30 秒 → **約 12 秒**(依然ロード時間が支配)
- 2回目以降(モデルロード済): → **約 1 秒**(実測 0.5〜1 秒程度、環境により変動)

### 変更ファイル
- `tools/speak/engine/base.py`: `SynthesisChunk` / `supports_streaming` / `synthesize_stream()` デフォルト実装
- `tools/speak/engine/gpt_sovits.py`: `supports_streaming=True`、`synthesize_stream()` 実装、`_build_inputs` リファクタ
- `tools/speak/playback_worker.py`: `_play_streaming()` 新設、初回チャンク到達時刻ログ
- `tools/speak/schema.py`: `clean_text_for_tts()` 前処理
- `tools/speak/text_cleaner.py`: 新規
- `config/default.json`: `streaming: true` 既定値

## 2026-04-16 — GPT-SoVITS 実用化

Merge: `a972124`
Feature: `ddee6f4`

Qwen3-TTS の合成時間が 2 分と実用性に欠けたため、GPT-SoVITS を主エンジンに採用。

### 新機能
- `engine/gpt_sovits.py`: GPT-SoVITS を SAIVerse プロセス内から直接呼び出せるよう実装
  - cwd を `external/GPT-SoVITS/` に一時変更する contextmanager
  - SAIVerse `tools/` パッケージとの名前空間衝突を回避する `sys.modules` シャドウイング
- `scripts/install_backends.py`:
  - `weights_local_dir` の `external/` プレフィックス修正
  - `pip_install_requirements` 仕様を追加(clone 後に requirements.txt を自動インストール)
- `voice_profiles/registry.json`: `_default.engine` を `gpt_sovits` に、GPT-SoVITS 推論パラメータを追加

### 性能
- Qwen3-TTS(1.7B): 1 発話あたり約 2 分
- GPT-SoVITS: 1 発話あたり 10〜30 秒(後のストリーミング対応で体感的な話し始めは約 1 秒まで短縮)

## 2026-04-16 — Qwen3-TTS を Base バリアントに修正

Merge: `acab51b`

初期実装で誤って `Qwen3-TTS-12Hz-1.7B-CustomVoice` を使っていたため、ゼロショット音声クローンが不可能だった(CustomVoice は事前登録話者専用で `generate_custom_voice(speaker=...)` を要求)。

### 修正
- モデルを `Qwen3-TTS-12Hz-1.7B-Base` に切替(`generate_voice_clone(ref_audio=..., ref_text=...)` を使用)
- `config/default.json` の `device` を `"auto"` → `"cuda:0"` に明示化(auto で CPU フォールバックが発生していた)
- `dtype` を `float16` → `bfloat16` に変更(RTX 4080 推奨、数値安定性向上)
- `output_device: 1` を設定し、リモートデスクトップ越し再生に対応
- `generate_voice_clone` 呼び出しに生成パラメータ(max_new_tokens / do_sample / top_k / top_p / temperature / repetition_penalty)を追加
- `voice_profiles/registry.json` の `_default.ref_text` をサンプル参照音声の内容に合わせて更新

## Initial — 初期実装

Commit: `694d272`

- ディレクトリ構造と Tool / Engine / Worker / Profile の骨子
- Qwen3-TTS エンジン実装(CustomVoice 版、後に Base 版に修正)
- GPT-SoVITS / Irodori-TTS エンジンのスケルトン
- `playbooks/public/sub_speak.json` で本体 sub_speak を上書きし `tts_speak` ノードを追加
- `scripts/install_backends.py` で各エンジンの自動導入
