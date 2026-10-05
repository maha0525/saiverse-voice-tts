# saiverse-voice-tts

SAIVerse 拡張パック。ペルソナが発話したテキストを、ペルソナごとに登録した参照音声でゼロショット音声クローン合成し、**バックエンド PC のスピーカーとブラウザの両方**で再生できます。Tailscale 越しのリモートでも動作。**2 回目以降は話し始めまで 5 秒以内** (環境により前後)。初回発話はモデルロードのため数十秒かかります。

エンジンは GPT-SoVITS と Irodori-TTS の 2 種類を本番利用でき、**ペルソナごとに使い分け可能**です。

## 特徴

- **ゼロショット音声クローン** — 3〜10秒の参照音声1つで、そのペルソナの声を再現
- **ストリーミング推論 + MP3 progressive 配信** — 合成完了を待たずチャンク単位でリアルタイム再生。iOS Safari 含むモバイルブラウザでも progressive 再生
- **ブラウザ側再生(Tailscale 対応)** — クライアントブラウザの `<audio>` 要素で再生、リモート運用可能。最後に操作したタブを自動的に「アクティブタブ」として判定し、そこだけで鳴る
- **サーバー側再生(opt-in)** — バックエンド PC のスピーカー直接再生も同時に有効化可能(RDP 等で両端での音出し)
- **サーバー側再生デバイスの選択** — アドオン管理 UI からサーバー側の出力先(既定デバイス/リモート オーディオ/特定スピーカー等)を選択
- **SAIVerse アドオン基盤対応** — メッセージバブルに🔊再生ボタンが自動表示。ペルソナごとに自動発話の ON/OFF、参照音声の差し替え、モード切替が UI から可能
- **エンジン切替可能な設計** — GPT-SoVITS / Irodori-TTS の 2 エンジン、アドオン管理 UI の「TTS エンジン」ドロップダウンでペルソナ別に切替可能
- **全てローカル推論** — 外部APIサーバ不要。インターネット接続は初回の重み DL 時のみ
- **Markdown/URI 自動除去** — ペルソナ応答中のリンク URL やマークアップは読み上げない

## 導入 (アドオンカタログ — 正規の経路)

v0.6.0 から、SAIVerse の **アドオン管理 → カタログ** から導入できます (Windows / Linux / Mac)。導入時に「使う音声エンジンを選んでください」と訊かれるので、使うものを選びます (複数可):

- **OpenAI TTS / ElevenLabs** — クラウド API。GPU 不要、API キーを入れるだけ
- **GPT-SoVITS** — ローカル合成。数 GB をダウンロード。Windows / Linux は NVIDIA GPU 推奨、Mac は動作未確認 (Apple silicon のみ。Intel Mac 向けの torch 2.11 は配布されていない)
- **Irodori-TTS** — 同上

GPT-SoVITS と Irodori-TTS のパッケージ (torch など) は、SAIVerse 本体の Python 環境ではなく、**エンジンごとの専用の Python 環境** (`~/.saiverse/addon_install/saiverse-voice-tts/envs/`) に入ります。本体のパッケージの版は変わりません。合成はその専用環境の Python の別プロセスで動きます。あとからエンジンを足したくなったら、アドオン管理の「導入済み」から同じ質問を開き直して選択肢を追加します。

導入後の **参照音声の配置** は、アドオン管理 UI の「参照音声」からアップロードします (ローカル合成のエンジンを使う場合)。

- 合成した音声 (wav) の保存先は `~/.saiverse/user_data/addon_data/saiverse-voice-tts/outputs/` です (v0.5.x までは `~/.saiverse/user_data/voice/out/`)。
- 以前 `setup.bat` で導入していた場合: カタログから更新すると、新しいエンジンは専用環境に入ります。`setup.bat` が本体の `.venv` に入れた torch などのパッケージは消されずに残ります (動作の妨げにはなりません)。

## クイックスタート(Windows、手動・開発者向け)

> アドオンカタログを使わずに手で入れる旧来の経路です。この経路では、エンジンのパッケージがすべて SAIVerse 本体の `.venv` に入ります (専用環境は作られません)。

SAIVerse 本体のセットアップ完了後、このリポジトリを `expansion_data/` 配下に配置し `setup.bat` を実行するだけです。**既定では GPT-SoVITS のみ**を導入します(Irodori も使いたい場合は [Irodori-TTS を使う](#irodori-tts-を使う) を参照)。

```batch
cd %USERPROFILE%\SAIVerse\expansion_data
git clone https://github.com/Nature109/saiverse-voice-tts.git
cd saiverse-voice-tts
setup.bat
```

`setup.bat` は以下を全自動で行います:

1. SAIVerse の `.venv` をアクティベート
2. パック依存(`numpy` / `sounddevice` / `soundfile` / `huggingface_hub` / `lameenc`)をインストール
3. `install_backends.py` で GPT-SoVITS を clone + 依存インストール + 重み DL
4. `torch` が CUDA 版であることを確認(そうでなければ `cu121` 版を自動再導入)
5. 音声デバイス一覧を表示、参照音声ファイルの有無をチェック

完了後、**参照音声の配置**だけ手動で:

1. `voice_profiles/samples/_default/ref.wav` に 3〜10秒の日本語音声 wav を置く(3秒以上10秒以内は必須)
2. `voice_profiles/registry.json` の `ref_text` を wav の書き起こしに合わせる
   (または SAIVerse の **アドオン管理 UI** から直接アップロード可能)

あとは SAIVerse 本体を通常通り起動するだけで、ペルソナの発話に合わせて音声が再生されます。

> **モバイル / Tailscale 運用での重要な注意**: Next.js の **dev mode (`npm run dev`)** では iOS Safari がタブを自動破棄する事象が観測されています。モバイルから利用する場合は **production build (`npm run build && npm run start`)** で SAIVerse を起動してください。

## Irodori-TTS を使う

Irodori-TTS は GPT-SoVITS と**別の声質傾向**を試したい場合や、一括合成が速い系を使いたい場合に使う 2 番目のエンジンです。本番利用可能ですが、セットアップとペルソナの割り当てに少し手順が要ります。

### 1. エンジンを導入する

アドオンカタログから入れた場合は、導入時の質問で Irodori-TTS を選ぶか、あとから「選択肢を追加」で足します。以下は手動 (`setup.bat`) の場合。

既に GPT-SoVITS を導入済みでも、Irodori-TTS の追加は独立して実行できます:

```batch
cd %USERPROFILE%\SAIVerse\expansion_data\saiverse-voice-tts
setup.bat irodori
```

新規セットアップ時に両方入れたい場合は `setup.bat all`。

`setup.bat irodori` が実行するのは:

- `external/Irodori-TTS/` へ upstream repo を shallow clone
- Irodori-TTS を `pip install -e` (Irodori 自身の依存として `torchcodec` も入る。Irodori 内部の `torchaudio.load` が torch 2.10+ 系で必要とする)
- HuggingFace から **重み 2 点を自動 DL**:
  - `Aratako/Irodori-TTS-500M-v2` (約 1.9GB、生成モデル本体)
  - `Aratako/Semantic-DACVAE-Japanese-32dim` (約 410MB、音声 codec)

### 2. エンジンをペルソナに割り当てる

アドオン管理 UI 経由(推奨):

1. SAIVerse 起動後、サイドバー → **アドオン管理** → **Voice TTS** を展開
2. 「ペルソナ別設定」のプルダウンから **対象ペルソナ**を選択
3. 「**TTS エンジン**」ドロップダウンを `irodori` に変更
4. 「**参照音声**」に wav をアップロード(3〜10 秒の日本語肉声、GPT-SoVITS 用の wav をそのまま使い回し可)
5. 「参照音声の書き起こし」は **空でよい**(Irodori は参照 wav のみから話者特徴を推定するため ref_text は使わない)

`registry.json` 経由で設定する場合:

```json
{
    "YourPersona_city_a": {
        "engine": "irodori",
        "ref_audio": "samples/YourPersona_city_a/ref.wav",
        "ref_text": "",
        "params": {
            "num_steps": 32,
            "seed": 42
        }
    }
}
```

### 3. 起動して確認

バックエンドを再起動して対象ペルソナに話しかけます。初回のみ Irodori ランタイムのロードで約 12 秒、**2 回目以降は話し始めまで約 1.4 秒**(文単位チャンキングによる疑似ストリーミング対応)。

より詳しいセットアップ手順(`setup.bat irodori` の内部動作、ログ観点等)は [SETUP.md](SETUP.md#irodori-tts-を使う場合) を、エンジン別パラメータの一覧は [voice_profiles/README.md](voice_profiles/README.md) を、内部設計は [ARCHITECTURE.md](ARCHITECTURE.md) を参照してください。

## クラウド TTS エンジン (OpenAI / ElevenLabs)

GPU を持たないユーザー向けに、API 経由の TTS エンジンを 2 種類用意しています。**追加のソフトウェアインストール不要、API key を入れるだけ**で使えます。アドオン管理 UI でペルソナ別に切替可能。

| エンジン | 特徴 | コスト感 (100 文字 1 回) |
|---|---|---|
| **OpenAI TTS** | preset 9 voices、ボイスクローン無し、設定が最小 | 約 $0.0015 (≈ 0.2 円、`tts-1`) |
| **ElevenLabs** | ボイスクローン対応 (Instant Voice Cloning)、品質最上位 | 月額制 ($5〜)、 100 文字 = 50 credit |

詳細な API key 取得手順、料金プラン、Voice ID の作り方、トラブルシューティングは [SETUP.md の「クラウド TTS エンジン」](SETUP.md#クラウド-tts-エンジン-openai--elevenlabs-を使う場合) を参照してください。

> 読み方辞書 (pronunciation_dict)、「音声を再生成」⟲ ボタンはどちらのエンジンでも引き続き動作します (engine 非依存)。

## 動作要件

| 項目 | 最小 | 推奨 |
|---|---|---|
| OS | Windows 10/11 or Linux | Windows 11 |
| Python | 3.10 | 3.10 |
| GPU | なし(CPU 推論は非実用) | NVIDIA RTX 3060 以上 / VRAM 6GB 以上 |
| CUDA | — | 12.1 以上 |
| ディスク | 8GB(GPT-SoVITS のみ) | 20GB(両エンジン + 各上流依存込み) |
| SAIVerse 本体 | アドオン基盤対応版 | 同上 |

**CPU 推論について**: GPT-SoVITS は 1 発話あたり数分、Irodori は CPU fp32 で動くが RTF 数倍で実用的ではありません。GPU 必須と考えてください。

**本体バージョンについて**: バブルの🔊再生ボタン・アドオン管理 UI 連携・ブラウザ側再生を有効化するには、SAIVerse 本体が以下を提供している必要があります:

- `saiverse.addon_metadata` / `saiverse.addon_events` / `saiverse.addon_deps` / `saiverse.addon_config`
- `tools.context.get_active_message_id` + 配線済みの `set_active_message_id`
- `/api/addon/events` SSE エンドポイントと `addon_loader` の自動マウント機構
- `ui_extensions.client_actions` 宣言対応 + `play_audio` action executor(ブラウザ側再生を有効化する場合)
- Route Handler `/api/addon/[...path]` による `/stream` パススルー対応(progressive 再生を有効化する場合)

これらが存在しない旧バージョンでも、バックエンド PC のスピーカーから自動再生する基本機能のみ動作します。

## バックエンド選択

| エンジン | 話し始め(初回) | 話し始め(2回目以降) | 品質 | 推論速度(RTF) | ディスク | ストリーミング |
|---|---|---|---|---|---|---|
| **GPT-SoVITS**(既定) | 数十秒(モデルロード含む) | 約 0.5〜1 秒 | ◎ | 1.3〜1.5x | 約 4GB | ネイティブ対応(上流 `streaming_mode`) |
| **Irodori-TTS** | 約 12 秒(モデルロード含む) | 約 1.4 秒 | ○ | 0.3〜0.35x | 約 2GB | 疑似ストリーミング(文単位チャンキング) |
| **OpenAI TTS** (クラウド) | 約 0.5〜1 秒 (API 接続) | 約 0.5〜1 秒 | ○ | 0.3x 程度 (API 側) | 0 (API のみ) | ネイティブ対応 (HTTP chunked) |
| **ElevenLabs** (クラウド) | 約 0.5〜1 秒 (API 接続) | 約 0.5〜1 秒 | ◎ (クローン可) | 0.3x 程度 (API 側) | 0 (API のみ) | ネイティブ対応 (HTTP chunked) |

> RTF = 合成時間 ÷ 出力音声長。小さいほど高速。Irodori の RTF 0.3x は「音声長の約 3 分の 1 の時間で合成が完了する」= リアルタイム再生に十分追いつく速度。

### 使い分けガイド

- **GPU あり / 完全ローカル運用**: GPT-SoVITS が既定・推奨。日本語実績が多く、ネイティブストリーミングで話し始めが最速。ペルソナごとに別の声質傾向を試したいなら Irodori-TTS に切替
- **GPU 無し / 手軽に試したい**: OpenAI TTS。API key だけで preset 9 voices から選べる。ボイスクローンは無いが安価で安定
- **GPU 無し + ボイスクローンしたい**: ElevenLabs。ダッシュボードで Instant Voice Cloning して voice_id を取得 → addon UI に貼り付け。品質高いが課金
- **長文(30 秒以上)の読み上げを頻繁に行う**: いずれもストリーミング対応だが、Irodori は固定コストが大きい分長文で RTF が有利。API 系は文字数課金なので長文連発は注意

`setup.bat` はデフォルトで GPT-SoVITS を導入します。Irodori 追加は [Irodori-TTS を使う](#irodori-tts-を使う) を、クラウドエンジン (OpenAI / ElevenLabs) のセットアップは [クラウド TTS エンジン (OpenAI / ElevenLabs)](#クラウド-tts-エンジン-openai--elevenlabs) を参照。

## ファイル構成

```
saiverse-voice-tts/
├── README.md
├── SETUP.md                       ← 詳細セットアップ
├── TROUBLESHOOTING.md             ← トラブルシューティング(目次付き)
├── ARCHITECTURE.md                ← 内部アーキテクチャ
├── CHANGELOG.md                   ← 変更履歴
├── LICENSE                        ← Apache License 2.0
├── setup.bat                      ← Windows 向けワンクリックセットアップ
├── addon.json                     ← アドオン manifest (UI 設定項目/ボタン/client_actions)
├── api_routes.py                  ← FastAPI ルート (/audio/*, /stream, client_action_failed, audio-devices)
├── requirements.txt               ← パック固有の最小依存
├── scripts/install_backends.py    ← 上流 clone + 重み DL + pip install
├── external/                      ← .gitignore。各上流リポジトリを clone
│   ├── GPT-SoVITS/
│   └── Irodori-TTS/
├── tools/speak/
│   ├── schema.py                  ← speak_as_persona Tool
│   ├── text_cleaner.py            ← Markdown/URI 除去
│   ├── profiles.py                ← registry.json + UI 設定ローダ
│   ├── playback_worker.py         ← FIFO キュー + ストリーミング再生
│   ├── audio_stream.py            ← MP3 pub/sub 配信レジストリ
│   └── engine/
│       ├── base.py                ← TTSEngine 抽象
│       ├── gpt_sovits.py
│       └── irodori.py             ← 疑似ストリーミング + truncation_factor 制御
├── playbooks/public/
│   └── sub_speak.json             ← 本体 sub_speak を上書きし、tts_speak ノード追加
├── voice_profiles/
│   ├── README.md                  ← プロファイル追加手順
│   ├── registry.json.template     ← 上流が配布する既定値(git 管理)
│   ├── registry.json              ← ユーザー編集用ローカル(.gitignore、初回起動時に template からコピー)
│   └── samples/<persona_id>/ref.wav
└── config/
    ├── default.json.template      ← 上流が配布する既定値(git 管理)
    └── default.json               ← ユーザー編集用ローカル(.gitignore、初回起動時に template からコピー)
```

> **テンプレート方式について**: `config/default.json` と `voice_profiles/registry.json` は**ユーザーが各環境固有の値を書き込むローカルファイル**で、git 管理外です。上流が配布するのは `.template` のみで、初回 `setup.bat` 実行(またはバックエンド初回起動)時にローカル版が自動生成されます。これにより `git pull` 時にユーザー編集が衝突しません。

## 使い方

### アドオン管理 UI (推奨)

SAIVerse 画面のサイドバー → **アドオン管理** → **Voice TTS** から以下が設定できます:

| 項目 | 種類 | 既定 | 説明 |
|---|---|---|---|
| 自動発話 | toggle (ペルソナ別) | ON | ペルソナ発話のたびに TTS 合成を自動実行 |
| TTS エンジン | dropdown (ペルソナ別) | `gpt_sovits` | `gpt_sovits` / `irodori` から選択 |
| 参照音声 | file (ペルソナ別) | - | そのペルソナの声を再現するための wav (3〜10 秒、必須) |
| 参照音声の書き起こし | text (ペルソナ別) | - | wav で話している内容 (句読点含め正確に)。**irodori では無視される** |
| サーバー側再生 | toggle | **OFF** | バックエンド PC のスピーカーから再生 |
| ブラウザ側再生 | toggle | **ON** | アクティブなクライアントタブ(ブラウザ)で自動再生 |
| ストリーミング推論 | toggle | ON | 合成完了を待たずチャンク単位で逐次再生 |
| サーバー側再生デバイス | dropdown | `<default>` | サーバー側再生のみで使用する出力先デバイス |

### 参照音声の追加(CLI で手動編集する場合)

アドオン UI でアップロードするのが楽ですが、CLI で一括管理したい場合は `voice_profiles/registry.json` にペルソナ ID をキーとしてエントリを追加します。フォーマット・エンジン別の `params` 一覧は [voice_profiles/README.md](voice_profiles/README.md) を参照してください。

アドオン管理 UI からアップロードした場合は本体側のアドオンストレージに保存され、**registry.json より UI 側の値が優先**されます。

### 再生方式の組み合わせ

| ブラウザ側再生 | サーバー側再生 | 動作 |
|---|---|---|
| ON (既定) | OFF (既定) | **Tailscale 含む全環境で推奨**。アクティブなブラウザタブから再生 |
| ON | ON | ブラウザと PC スピーカー両方で再生(バックエンド PC 手元で使うとき等) |
| OFF | ON | ブラウザで鳴らさず PC スピーカーのみ(ブラウザを開いていない運用) |
| OFF | OFF | 合成は実行、音は鳴らない(バブル再生ボタンで後から手動再生のみ) |

### Tailscale 運用について

**ブラウザ側再生**によりリモートブラウザで音声が鳴るようになりました:

- SAIVerse を **production build** で起動 (`npm run build && npm run start`) — dev mode では iOS Safari がタブ discard を起こす
- Tailscale 越しにスマホ等からアクセス
- 初回に画面を 1 回タップ (iOS Safari の autoplay unlock のため)
- ペルソナと会話 → ブラウザから音声が progressive に再生される

複数端末を同時に開いている場合、**最後に画面を触った端末**が自動的に「アクティブクライアント」として鳴らされます。ヘッダーの Radio アイコンで可視化されます。

## ライセンス

本パック自体は **Apache License 2.0**([LICENSE](LICENSE) 参照)。各 TTS エンジンの再配布は上流ライセンスに従ってください。

| エンジン | コード | 重み | 再配布 |
|---|---|---|---|
| GPT-SoVITS | MIT | MIT | ✅ |
| Irodori-TTS | MIT | MIT | ✅(要: ベース依存の個別再確認) |

配布物には `external/<repo>/LICENSE` を同梱してください。
