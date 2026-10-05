# ARCHITECTURE

このドキュメントは saiverse-voice-tts の内部構造と設計判断を記述します。バグ修正・機能追加時の参考にしてください。

## 全体像

```
ペルソナ発話
   │
   ├─ SEA Playbook (sub_speak)
   │    └─ compose(LLM) → process_body → ★ tts_speak(speak_as_persona)
   │
   ▼
Tool: speak_as_persona(text)
   │
   ├─ clean_text_for_tts()        Markdown/URI除去
   ├─ get_active_message_id()     persona_context から取得
   └─ enqueue_tts(text, msg_id)   FIFOキューへ投入して即return(fire-and-forget)
   │
   ▼
Background worker thread (_TTSWorker._run)
   │
   ├─ profile = get_profile(persona_id)           registry.json または UI アップロードから
   ├─ engine = create_engine(profile.engine)      初回のみ lazy load
   │     (gpt_sovits / irodori は子プロセスの代理エンジン。子プロセスは
   │      エンジン専用の Python 環境の Python で動く — docs/out_of_process.md)
   │
   ├─ [ストリーミング対応エンジン]
   │    └─ engine.synthesize_stream() → SynthesisChunk yield
   │          ├─ sd.OutputStream.write(chunk)            ← サーバ側再生(opt-in)
   │          ├─ audio_stream.push_pcm(msg_id, chunk)    ← MP3 pub/sub にエンコード投入
   │          └─ 全チャンクを collect
   │
   └─ [非対応エンジン]
        └─ engine.synthesize() → SynthesisResult
              ├─ sd.play(blocking=True)                  ← サーバ側再生(opt-in)
              └─ audio_stream.push_complete(...)         ← 完成 MP3 を一括投入
   │
   ▼
wav 保存 (~/.saiverse/user_data/addon_data/saiverse-voice-tts/outputs/<job_id>.wav。v0.5.x までは ~/.saiverse/user_data/voice/out/)
set_metadata(msg_id, audio_file, audio_stream_url)
emit_addon_event("audio_ready") ───────► フロント client_action 発火
   │
   ▼
24時間 GC
```

### クライアント側再生チャネル

```
ブラウザ (iOS Safari / Chrome / etc)
   │  (SSE: /api/addon/events)
   │  ← emit_addon_event("audio_ready", {stream_url, audio_path})
   │
   ├─ Active tab 判定 (BroadcastChannel + last interaction time)
   │    └─ 非アクティブなら再生スキップ
   │
   └─ Active なら play_audio action executor
        │
        ├─ 初回: silent Blob URL を sync play() で autoplay unlock (iOS)
        ├─ primary: fetch GET /api/addon/saiverse-voice-tts/stream/<msg_id>
        │    └─ Route Handler pump (arrayBuffer 回避, Range 剥ぎ) → audio/mpeg
        │    └─ MediaSource/blob url を HTMLAudioElement に attach
        └─ fallback: GET /audio/<msg_id> (完成後の MP3/WAV)
```

## コンポーネント

### `tools/speak/schema.py`

Tool エントリポイント。`ToolSchema(name="speak_as_persona")` を公開し、SAIVerse の自動 Tool 発見機構に拾われる。

- `text` のみを引数に取る(persona_id は context 経由、message_id は将来本体が context に注入する予定)
- `clean_text_for_tts()` で前処理
- `enqueue_tts(cleaned, persona_id)` で即 return
- 戻り値: `{content: "", metadata: {voice_tts: {status, job_id, persona_id}}}`

**設計判断**: 同期版は意図的に提供していない。Playbook がSEA 実行をブロックしないよう常に非同期キュー投入。

### `tools/speak/text_cleaner.py`

Markdown と SAIVerse URI を除去する純関数モジュール。

| パターン | 処理 |
|---|---|
| `[text](url)` | `text` のみ残す |
| `![alt](url)` | 全て削除 |
| `[text][ref]` | `text` のみ残す |
| `**bold**` / `__bold__` | 中身のみ |
| `*italic*` / `_italic_` | 中身のみ |
| `` `code` `` | 中身のみ |
| `~~strike~~` | 中身のみ |
| `# heading` | `#` 記号削除 |
| `- list` / `* list` / `+ list` | マーカー削除 |
| `> quote` | `>` 削除 |
| `https://...`, `saiverse://...`, etc | 削除 |
| 3個以上の空行 | 2個に圧縮 |

**設計判断**: TTS エンジンに渡す前に一段階変換する。Building 履歴に残るテキスト自体は Markdown のままなので、チャット UI の見た目は壊れない。

### `tools/speak/profiles.py`

`voice_profiles/registry.json` をロードして persona_id → profile dict を返す。

- 初回読込のみキャッシュ(プロセス生存中は再読込しない)
- `ref_audio` の相対パスを pack root からの絶対パスに解決
- `persona_id` が registry にない場合は `_default` にフォールバック
- `_default` も無ければ `None` → TTS スキップ(ログ出力のみ)

**設計判断**: 参照音声の DL/配置はユーザー作業なので、無い場合はエラーではなく警告にとどめる。

### `tools/speak/playback_worker.py`

`_TTSWorker` シングルトン。FIFO `queue.Queue` + 単一ワーカースレッド + `sounddevice` 再生。

主要責務:
- キュー管理(`enqueue()` / `_run()`)
- 初回のみエンジン lazy load(`_get_engine()`)
- ストリーミング / 非ストリーミング切替(`_process()`)
- wav 保存(`_save_wav()`)
- 古い wav の GC(`_gc_old_files()`)
- プロセス終了時の graceful shutdown(`atexit`)

#### ストリーミング再生の実装(`_play_streaming()`)

```python
stream = None
collected: list[np.ndarray] = []
for chunk in engine.synthesize_stream(text=..., ...):
    if stream is None:
        # 初回チャンク到達 = サンプルレートが判明した瞬間
        stream = sd.OutputStream(samplerate=chunk.sample_rate, channels=1, device=...)
        stream.start()
    stream.write(chunk.audio.astype(np.float32))
    collected.append(chunk.audio)
stream.stop(); stream.close()
# 全チャンク結合 → wav 保存(履歴用)
```

**設計判断**: `sd.play()` ではなく `sd.OutputStream` を使うのは、合成と同時進行で書き込みたいため。`sd.play()` は一括バッファ受取が前提。

**失敗時の自動フォールバック**: ストリーミングが例外で失敗したら `_play_streaming()` が False を返し、`_process()` が通常の `engine.synthesize()` + `sd.play()` にフォールバック。

### `tools/speak/engine/`

`TTSEngine` 抽象クラスと具体実装。

#### `base.py`
- `TTSEngine`: 抽象基底、`synthesize()` 必須、`synthesize_stream()` はデフォルト実装(単一チャンクをラップ)
- `supports_streaming: bool`: クラスレベルフラグ
- `SynthesisResult`: 一括合成の結果(audio, sample_rate, duration_ms)
- `SynthesisChunk`: ストリーミング単位(audio, sample_rate)

#### `gpt_sovits.py`
最も完成度の高いエンジン。`supports_streaming = True`。

**難所とその対処**:
1. **cwd 依存**: GPT-SoVITS は `pretrained_models/...` を相対パスで開く。`_cwd(_EXTERNAL_REPO)` コンテキストマネージャでロード・推論両方を repo 直下で実行。
2. **`tools/` 名前空間衝突**: GPT-SoVITS 内部の `from tools.audio_sr import ...` が SAIVerse の `tools/` パッケージと衝突。`_shadowed_tools_namespace()` で import 中だけ `sys.modules['tools']` を退避。
3. **ストリーミングパラメータ**: `streaming_mode=True` + `parallel_infer=False` の組み合わせが必要(SoVITS V3/V4 は自動で return_fragment にフォールバック)。

#### `irodori.py`

Irodori-TTS (Aratako/Irodori-TTS-500M-v2) のアダプタ。**上流 API は一括合成のみ**で native streaming をサポートしないが、アダプタ側で**文単位チャンキングによる疑似ストリーミング**を実装し、`supports_streaming = True` として公開している。

**主要な内部定数**:

| 定数 | 値 | 用途 |
|---|---|---|
| `_BUDGET_K` / `_BUDGET_MARGIN` | 0.25 / 1.5 | `seconds = chars × K + M` でチャンク単位の予算を自動算出 |
| `_TRIM_K` / `_TRIM_MARGIN` | 0.25 / 1.5 | 合成後のハードトリム長(本文を切らないため広め) |
| `_TRUNCATION_FACTOR` | **0.75** | **ゴミ音声抑制の要**。低確率トークンを分布から除外 |
| `_LONG_CHUNK_CHARS` | 35 | 超えるチャンクは読点で再分割 |
| `_INTER_CHUNK_PAUSE_SEC` | 0.12 | チャンク間の自然な息継ぎ無音 |

**synthesize_stream() のフロー**:

```
文単位分割 (。！？!?)
    ↓
長文を読点で再分割 (>35文字)
    ↓
各チャンクを逐次:
   seconds = chars × 0.25 + 1.5
   SamplingRequest(
     text=chunk_text, ref_wav=...,
     num_steps=24, truncation_factor=0.75,
     seconds=seconds,
     trim_tail=True, tail_std_threshold=0.08, ...)
    ↓
   runtime.synthesize(req)  # 一括 (内部で ~1s)
    ↓
   _trim_tail_garbage(audio)  # ハードトリム + -50dB 末尾無音除去
    ↓
   yield SynthesisChunk
    ↓
   yield 120ms 無音チャンク (次チャンクとの区切り)
```

**なぜ `truncation_factor=0.75` が必要か**: Irodori は `seconds` 予算と実音声長が乖離するとモデルが低確率トークンを使って「予算埋めのゴミ発声」を生成する。truncation で分布の裾を切ると、モデルが自然な場所で停止するようになり**予算埋めゴミが生成されなくなる**。0.7 以下だと本文が切れ、0.8 以上だとゴミ再発。0.75 が実測のスイートスポット。

(エンジン間の性能比較表は [README の「バックエンド選択」](README.md#バックエンド選択) を参照)

**dtype 一貫性**: `model_precision` と `codec_precision` は揃える必要がある(不一致だと F.linear が `mat1/mat2 dtype mismatch` で落ちる)。既定は両方 `bf16` + `codec_device=cuda`。

**checkpoint 解決**: `RuntimeKey.checkpoint` は上流ではローカルファイルパスを想定しているが、アダプタ側で拡張子なしなら HF repo ID と判定して `hf_hub_download(repo_id=..., filename='model.safetensors')` で自動 DL する。

**torchaudio.load と torchcodec**: Irodori 内部の `_load_audio` が `torchaudio.load` を呼ぶが、torch 2.10+ 系ではこれが `torchcodec` backend を要求する。v0.6.0 からは Irodori-TTS の専用環境の requirements (`envs/irodori-*.txt`) に `torchcodec>=0.10` を明示している (パックの `requirements.txt` からは外した)。Irodori-TTS 自身は torchcodec を `<0.11` (torch 2.10 向け) に縛っていて、torch 2.11 の環境で torchcodec を読めないときは、Irodori 内部の `_load_audio` が soundfile に切り替える。

**device の自動切り替え (v0.6.0)**: `config/default.json.template` は NVIDIA GPU 向け (`cuda` + `bf16`)。上流は使えない device や CUDA/XPU 以外での `bf16` を拒むので、アダプタ (`_resolve_runtime_settings`) が、CUDA が使えなければモデルを MPS (Apple silicon) → CPU へ、コーデックを CPU へ落とし、`bf16` を `fp32` にする (WARNING ログ付き)。

### `tools/speak/audio_stream.py`

クライアント側再生用 MP3 progressive 配信レジストリ。pub/sub パターンで **複数コンシューマ同時配信**に対応。

```python
@dataclass
class _StreamContext:
    encoder: Optional[Any]  # lameenc.Encoder
    sample_rate: int
    frames: List[bytes] = field(default_factory=list)   # エンコード済み MP3 フレーム
    consumers: List["Queue[Optional[bytes]]"] = field(default_factory=list)
    closed: bool = False
```

**主要 API**:

| 関数 | 呼び出し元 | 役割 |
|---|---|---|
| `start(msg_id, sample_rate)` | `_TTSWorker` | 新規ストリーム開始、`lameenc.Encoder` 初期化 |
| `push_pcm(msg_id, pcm_float32)` | 各チャンク yield 時 | PCM → MP3 エンコード、全 consumers の Queue に分配 |
| `close(msg_id)` | 合成完了時 | encoder flush + None sentinel で全 consumers に終了通知 |
| `subscribe(msg_id)` | `/stream` HTTP handler | 新規 Queue を返し、**既存の全 frames を先頭に seed** |

**設計判断**:
- **WAV ではなく MP3**: iOS Safari は WAV の `Chunk Size = 0xFFFFFFFF`(ストリーミング目印)を拒否する。MP3 は長さヘッダ不要で progressive に自然対応
- **単一 Queue ではなく pub/sub**: ブラウザの fetch 再接続や複数コンシューマ(debug の curl 等)でチャンクが奪い合いになる問題を解消
- **遅参加者への seed**: `subscribe()` が呼ばれた時点で既に送信済みのフレームを Queue に先入れ。これで接続タイミングによる頭欠けを防ぐ
- **lameenc 選定**: libmp3lame の wheel バンドル版。Windows でも追加システム依存なし

### `api_routes.py`

FastAPI ルータ。起動時に `addon_loader` が検出してマウント。

| エンドポイント | 用途 |
|---|---|
| `GET /api/addon/saiverse-voice-tts/audio/{msg_id}` | 完成後の wav(または MP3)を配信。過去発話の手動再生用 |
| `GET /api/addon/saiverse-voice-tts/stream/{msg_id}` | MP3 progressive ストリーム。`audio_stream.subscribe()` で Queue 受け取り、Chunked Transfer Encoding で配信 |
| `GET /api/addon/saiverse-voice-tts/audio-devices` | `sounddevice.query_devices()` 結果を返す。UI の出力デバイス dropdown 用 |
| `POST /api/addon/saiverse-voice-tts/client_action_failed` | フロントが client_action 実行に失敗した時のフォールバック webhook |

**重要なインポート**:
```python
# tools/_loaded/speak/ にロードされるため、saiverse 本体の tools 名前空間を経由
from tools._loaded.speak.audio_stream import subscribe, push_pcm, ...
```
`from tools.speak.audio_stream import ...` だと本体の tools/ を参照してしまい `ModuleNotFoundError` になる。拡張パックの api_routes は常に `tools._loaded.<pack>.*` 経由。

## データフロー詳細

### 初回発話

```
t=0.0  persona → speak_as_persona(text)
t=0.01 enqueue → return
t=0.01 worker: profile lookup
t=0.02 worker: engine.synthesize_stream() 呼び出し
t=0.02 → lazy_load (10〜20秒): BERT / CNHuBERT / T2S / VITS ロード
t=20.0 ref_audio 埋め込み計算 (1〜2秒)
t=22.0 第一チャンク yield
t=22.0 sd.OutputStream 開始 → 再生開始
t=22.5 ← ★話し始めが聞こえるタイミング
t=N    最終チャンク yield + stream.stop()
t=N    wav 保存
```

### 2回目以降

```
t=0.0    enqueue
t=0.01   worker: engine 取得(キャッシュ済み)
t=0.01   engine.synthesize_stream()
t=0.01   ref_audio キャッシュヒット(同一ペルソナなら set_ref_audio スキップ)
t≈0.5〜1 第一チャンク yield → 再生開始 ← ★話し始め(余裕を見て5秒以内が目安)
t=N      完了
```

## 発話経路への接続

v0.4.0 以降は本体側の **`server_hooks`** 機構経由で `persona_speak` イベントを
購読する。本体の `sea/runtime_emitters.py` の `emit_speak` / `emit_say` 末尾で
`dispatch_hook("persona_speak", ...)` が ThreadPoolExecutor に submit され、
本パックの `speak_hook.on_persona_speak` がバックグラウンドスレッドで呼ばれる。

```
本体: persona が発話
   │
   ├─ emit_speak / emit_say
   │    ├─ Building history 記録
   │    ├─ set_active_message_id
   │    ├─ gateway_handle_ai_replies
   │    └─ dispatch_hook("persona_speak", text_for_voice=..., message_id=..., ...)
   │           │
   │           ▼ ThreadPoolExecutor (max_workers=4) で隔離実行
   │
   ▼
拡張パック: speak_hook.on_persona_speak
   │
   ├─ get_effective_params(persona_id) で auto_speak チェック
   ├─ clean_text_for_tts(text_for_voice)
   └─ enqueue_tts(cleaned, persona_id, message_id)  ← 既存の TTS Worker キューへ
```

ハンドラ宣言は `addon.json`:

```json
{
    "server_hooks": [
        {
            "event": "persona_speak",
            "handler": "speak_hook:on_persona_speak"
        }
    ]
}
```

本体は本パックが有効化されている時のみハンドラを register し、無効化トグルで
unregister する。ハンドラが例外を投げても本体側で隔離されるため、発話処理本体
が止まることはない。

### 旧 Playbook 上書き方式 (v0.3.x まで)

v0.3.x までは `playbooks/public/sub_speak.json` で本体の `builtin_data/playbooks/public/sub_speak.json`
を上書きし、末尾に `tts_speak(speak_as_persona)` ノードを追加する 3 ノード構成
で TTS を発火していた。

2026-05-01 の認知モデル Phase C-2/3 で本体の発話経路が Track ベース
(`track_user_conversation` / `track_external` 等) に刷新され、デフォルトの
ユーザー会話が `sub_speak` を経由しなくなったため、この方式は破綻した
(v0.4.0 で `server_hooks` 経由に移行)。

`speak_as_persona` ツール自体は v0.4.0 以降も残しているため、ペルソナが
`/spell speak_as_persona text='...'` で明示的に発話制御するケースは引き続き
動作する。

## 設計上の制約と拡張方針

### 現状の制約

1. **ペルソナ単位の音量/音響調整無し**: `speed` 等は調整可だが、ボリュームはアプリ外
2. **FIFO 直列再生のみ**: 複数ペルソナの同時発話は順番待ちになる
3. **transformers バージョン制約**: GPT-SoVITS は `transformers>=4.43,<=4.50` を要求するため、別のエンジンを追加する際はこの範囲を尊重する必要がある
4. **lameenc の buffering latency**: MP3 フレーム単位の出力のため、第一フレーム到達まで約 3 秒のバッファが必要(32kHz × mono × 1152 samples/frame × 数フレーム分)。GPT-SoVITS の合成レイテンシに隠れるため実感では問題にならない
5. **iOS Safari dev mode 不可**: Next.js `npm run dev` ではタブが discard される。モバイル運用は `npm run build && npm run start` 必須

### 解消済みの旧制約(参考)

v1.x で制約だった以下は v2.0 で解消:

| 旧制約 | 解消された仕組み |
|---|---|
| バックエンド PC のスピーカーからのみ再生 | `ui_extensions.client_actions` + MP3 progressive でブラウザ側再生対応 |
| Tailscale/リモートで音が聞こえない | クライアント側再生(iOS Safari autoplay unlock 込み) |
| 長時間音声で途中停止 | Route Handler pump + Range 剥ぎで解消 |
| 複数コンシューマでチャンク欠落 | audio_stream を FIFO Queue から pub/sub に変更 |

### 本体アドオン基盤(SAIVerse feature/memory-notes-and-organize 〜 feature/client-playback-actions)

本体側で以下が提供され、拡張パック側で連携済み:

- `persona_context.get_active_message_id()` → Tool から現在のメッセージ ID 取得
- `saiverse.addon_metadata.set_metadata` / `get_metadata` → メッセージ単位のキーバリュー
- `saiverse.addon_events.emit_addon_event` → `GET /api/addon/events` に SSE push
- `saiverse.addon_deps.get_manager` → FastAPI Depends に差し込む認証ゲート
- `saiverse.addon_config.get_params` → パック設定(トグル / 選択肢 / ペルソナ別設定)を取得
- `addon.json` マニフェストによる UI 拡張:
  - `ui_extensions.bubble_buttons` → バブル内ボタン自動生成
  - `ui_extensions.client_actions` → SSE イベント → クライアント側 JS executor を発火
- 拡張パック `api_routes.py` を起動時自動ロードして `/api/addon/<addon>/...` にマウント
- Next.js Route Handler `/api/addon/[...path]` による `/stream` パススルー(progressive 配信用)
- `play_audio` action executor(フロント組み込み、`ui_extensions.client_actions` 宣言ベース)
- アクティブクライアントタブ自動判定(BroadcastChannel + last interaction time)

拡張パック側の実装マッピング:

| 本体 API / 規約 | 拡張パック側の利用場所 |
|---|---|
| `get_active_message_id()` | `playback_worker._get_active_message_id()` |
| `set_metadata` | `playback_worker._notify_audio_ready()` |
| `emit_addon_event("audio_ready")` | `playback_worker._notify_audio_ready()` |
| `get_manager` Depends | `api_routes.get_audio` / `stream_audio` |
| `get_params` | `playback_worker._should_run_client_action()` 等で再生モード判定 |
| `ui_extensions.bubble_buttons` | `addon.json` の `play_audio` ボタン定義 |
| `ui_extensions.client_actions` | `addon.json` で `audio_ready` → `play_audio` を宣言 |
| `api_routes.py` 自動マウント | `/api/addon/saiverse-voice-tts/{audio,stream,audio-devices,client_action_failed}` |

### client_actions の発火フロー

```
1. 合成完了 or 初チャンク到達
       │
       ├─ set_metadata(msg_id, audio_stream_url=/stream/<msg_id>,
       │                       audio_file=/audio/<msg_id>)
       └─ emit_addon_event("audio_ready", {message_id, addon_name})
                │
                ▼ (SSE: /api/addon/events)
2. 全クライアントタブが受信
       │
       └─ addon.json の client_actions 定義を参照:
            {
              "event": "audio_ready",
              "action": "play_audio",
              "source_metadata_key": "audio_stream_url",
              "fallback_metadata_key": "audio_file",
              "requires_active_tab": true,
              "requires_enabled_param": "client_side_playback",
              "on_failure_endpoint": "/client_action_failed"
            }
       │
       ├─ requires_enabled_param 未 ON → スキップ
       ├─ requires_active_tab 未 active → スキップ
       └─ execute play_audio(source_url, fallback_url, msg_id)
```

### 汎用性の確保

`play_audio` action executor や client_actions 宣言機構は **Voice TTS 専用ではなく汎用**。他のパック(例: 効果音、動画添付等)でも `addon.json` に client_actions を書くだけで同じ仕組みが利用できる。パック側は SSE イベントを emit するだけで、フロントへの JS 注入は一切不要。

## 参考

- [GPT-SoVITS upstream](https://github.com/RVC-Boss/GPT-SoVITS)
- [Irodori-TTS upstream](https://github.com/Aratako/Irodori-TTS)
- [SAIVerse 本体 Tool 仕様](../../CLAUDE.md) — 拡張 Tool 作成ガイド
