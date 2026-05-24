# Intent: voice-tts の音声合成を別プロセスに切り出す

**ステータス**: 🟢 実装済み (実機検証待ち)
**作成日**: 2026-05-24
**関連**: 発生源は SAIVerse 本体リポジトリの `docs/issues/mcp_cancel_scope_spin_gil_starvation.md`

## なぜやるか

GPT-SoVITS の音声合成は SAIVerse 本体と同じ Python プロセスの中のスレッドで
動いている。AR デコードは 1 トークンずつ Python の処理を挟むため、GIL
(プロセス内で同時に Python コードを実行できるのは 1 スレッドだけ、という
Python の仕組み) が要る。本体プロセスの別のスレッドが GIL を握り続けると、
合成のトークンループが割り当てを奪われて激遅になる。

2026-05-24 に実際これが起きた: MCP の不具合 (別 issue) で本体のあるスレッドが
GIL を焼き続け、TTS が約 10 倍遅くなって 1 メッセージの合成が数時間化した。
その MCP 不具合自体は修正済み (所有タスク方式) だが、**「本体プロセスが何かで
重くなると TTS が道連れで遅くなる」という構造はそのまま残っている**。原因は
MCP に限らず、重い pulse 処理・大量メッセージの処理・将来追加されるアドオン
など、いくらでもありうる。

音声合成を別プロセスに分ければ、別プロセスは独自の GIL を持ち、OS が独立に
CPU/GPU を割り当てるので、**本体プロセスの負荷が何であれ TTS の合成速度は
影響を受けない**。あわせて、合成側がクラッシュしても本体は巻き添えにならない
(クラッシュ隔離) という副次的な利点もある。

## 守りたい不変条件

1. 音声合成のスループットが本体プロセスの負荷に左右されない。
2. 合成プロセスのクラッシュが本体を巻き込まない。本体は合成失敗として扱い、
   復旧 (プロセス再起動) を試みる。
3. ユーザーから見た挙動 (ストリーミング再生、stackchan への転送、フロー制御)
   は現状と変わらない。

## 設計の要 = 既存の TTSEngine が差し替え地点

現状、音声を作る処理は `TTSEngine` という共通インターフェースの裏にある:

- `playback_worker` は `create_engine(name, cfg)` でエンジンを得て、
  `synthesize(text, ref_audio, ref_text, params) -> SynthesisResult` と
  `synthesize_stream(...) -> Iterator[SynthesisChunk]` を呼ぶだけ。モデルには
  直接触らない (`playback_worker.py:389` `_get_engine`、`:732` 付近)。
- エンジンに渡すデータは単純: テキスト (str)、参照音声ファイルのパス (str)、
  参照テキスト (str)、設定 (数値中心の dict)。
- 返るデータも単純: 音声の断片 (float32 配列) と サンプルレート (int)。

したがって別プロセス化は **`TTSEngine` のもう 1 つの実装を足すだけ**で済む:

- **本体プロセス側**: `TTSEngine` を実装した「代理エンジン」。`synthesize_stream`
  が呼ばれたら、テキスト等を合成プロセスに送り、返ってくる音声断片を
  `SynthesisChunk` として yield する。`playback_worker` は無改造。
- **合成プロセス側**: 本物の `GPTSoVITSEngine` を保持する常駐プログラム。
  リクエストを受けて `synthesize_stream` を回し、音声断片を流し返す。モデルは
  起動時 (または初回) に 1 度だけ読み込んで居座らせる (読み込みに数百秒かかる
  ため毎回はやらない)。

参照音声はファイルパスで渡す。両プロセスは同じマシンなので、合成プロセスが
同じファイルを読めばよく、音声データ自体を送る必要はない。

## 決めたこと (2026-05-24 インタビュー)

1. **2 プロセス間の通信方法 = 標準入出力 (stdin/stdout)**。合成プロセスは本体が
   起動する子プロセス。テキスト等のリクエストと音声断片を、標準入出力に長さ付き
   フレーム (各メッセージの先頭に長さを置く) で流す。理由: 追加の依存もポート管理
   も要らず、子プロセスの起動・監視・終了が一番素直で後片付けの心配がない。

2. **適用範囲 = 最初から全面的に別プロセス**。設定での切り替えは設けず、本体は
   常に代理エンジン経由で合成プロセスを使う。本物の `GPTSoVITSEngine` は合成
   プロセスの中だけで動く。本体プロセス内で直接モデルを動かす経路は既定から外す。

3. **クラッシュ・無応答時 = 時間切れで諦めて自動再起動**。合成プロセスが一定時間
   応答しなければ、その回は合成失敗として扱い、合成プロセスを自動で再起動する。
   次の発話から復旧する。タイムアウト秒数は実装時に決める (初回はモデル読み込みで
   長くかかるため、初回と通常で別の上限にする)。

## 関連リソース

- `tools/speak/engine/subprocess_ipc.py`: 通信フレーム (本実装)
- `tools/speak/engine/subprocess_worker.py`: 子プロセス本体 (本実装)
- `tools/speak/engine/subprocess_proxy.py`: 本体側代理エンジン (本実装)
- `tools/speak/engine/gpt_sovits.py`:
  `GPTSoVITSEngine`、`synthesize`/`synthesize_stream`、`SynthesisChunk`
- `tools/speak/engine/__init__.py`:
  `TTSEngine` 基底、`create_engine` ファクトリ (gpt_sovits を代理エンジンに差し替え)
- `tools/speak/playback_worker.py`:
  `_get_engine` (389)、`_play_streaming` (434)、`_process` (732 付近)
- 発生源: SAIVerse 本体リポジトリの `docs/issues/mcp_cancel_scope_spin_gil_starvation.md`

## ログ

- 2026-05-24: ドラフト作成。コード調査で「TTSEngine が差し替え地点」と確認。
- 2026-05-24: インタビュー実施。通信=標準入出力(子プロセス)、適用=最初から全面、
  異常時=時間切れで自動再起動、に決定。
- 2026-05-24: 実装完了。`engine/subprocess_ipc.py` (フレーム), `subprocess_worker.py`
  (子プロセス本体), `subprocess_proxy.py` (本体側代理エンジン) を追加し、
  `engine/__init__.py` の `create_engine("gpt_sovits")` を代理エンジンに差し替え
  (環境変数 `VOICE_TTS_IN_PROCESS=1` で従来の in-process に戻せる escape hatch)。
  子プロセスの stderr は per-session の `voice_tts_worker.log` に転送 (従来 stdout
  のみで消えていた GPT-SoVITS 推論ログが残るようになった)。`playback_worker` は
  無改造 (TTSEngine 公開 IF のみ使用)。テスト追加 (`tests/test_voice_tts_subprocess.py`
  7件: フレーム往復・ストリーム・エラー伝播・タイムアウト再起動) + 既存アドオン
  テスト 67件パス、ruff クリーン。**実機 (GPU + モデル) での end-to-end は未検証**。
