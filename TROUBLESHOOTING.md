# トラブルシューティングガイド

saiverse-voice-tts 拡張パックで発生しやすい問題と、その診断手順・解決方法をまとめています。

---

## 目次

### セットアップ関連
- [setup.bat が途中で止まる](#setupbat-が途中で止まる)
- [opencc のビルドエラー(Windows)](#opencc-のビルドエラーwindows)
- [editdistance のビルドエラー(Python 3.13+)](#editdistance-のビルドエラーpython-313)
- [ModuleNotFoundError が出る](#modulenotfounderror-が出る)
- [DLL アクセス拒否エラー(pip install 時)](#dll-アクセス拒否エラーpip-install-時)

### GPU / CUDA 関連
- [CUDA が認識されない(`CUDA: False`)](#cuda-が認識されないcuda-false)
- [GPU が複数あるがどちらが使われるかわからない](#gpu-が複数あるがどちらが使われるかわからない)
- [GPU 稼働率が 0% のまま](#gpu-稼働率が-0-のまま)
- [CUDA out of memory (VRAM 不足)](#cuda-out-of-memory-vram-不足)

### TTS 合成関連
- [🔊ボタンがずっと「準備中」のまま](#ボタンがずっと準備中のまま)
- [合成が極端に遅い(2分以上)](#合成が極端に遅い2分以上)
- [合成は成功するが音が鳴らない](#合成は成功するが音が鳴らない)
- [fast-langdetect のキャッシュディレクトリエラー](#fast-langdetect-のキャッシュディレクトリエラー)
- [文字化けしたテキストが読み上げられる](#文字化けしたテキストが読み上げられる)
- [参照音声と異なる声で合成される](#参照音声と異なる声で合成される)

### Irodori-TTS 固有
- [Irodori で `ModuleNotFoundError: torchcodec`](#irodori-で-modulenotfounderror-torchcodec)
- [Irodori で `dtype mismatch` / `F.linear` エラー](#irodori-で-dtype-mismatch--flinear-エラー)
- [Irodori で `Irodori-TTS repository not found`](#irodori-で-irodori-tts-repository-not-found)
- [Irodori で合成が極端に遅い(RTF 5x 以上)](#irodori-で合成が極端に遅いrtf-5x-以上)
- [Irodori で末尾にゴミ音声が混ざる / 文末で途切れる](#irodori-で末尾にゴミ音声が混ざる--文末で途切れる)
- [Irodori で HuggingFace 重みダウンロードが失敗する](#irodori-で-huggingface-重みダウンロードが失敗する)

### 音声再生関連(サーバー側)
- [サーバー PC のスピーカーから音が出ない](#サーバー-pc-のスピーカーから音が出ない)
- [音声が途切れる / スタッターする](#音声が途切れる--スタッターする)

### 音声再生関連(クライアント側 / ブラウザ)
- [クライアント側再生が全く動かない](#クライアント側再生が全く動かない)
- [モバイル Safari で初回の音声が鳴らない(autoplay)](#モバイル-safari-で初回の音声が鳴らないautoplay)
- [複数端末を開いていてどの端末で鳴るか](#複数端末を開いていてどの端末で鳴るか)
- [ブラウザの🔊ボタンで過去の発話が再生できない](#ブラウザのボタンで過去の発話が再生できない)
- [ストリーミング再生が途中で止まる(長時間音声)](#ストリーミング再生が途中で止まる長時間音声)
- [モバイル Safari でタブが自動リロード/破棄される](#モバイル-safari-でタブが自動リロード破棄される)
- [Tailscale 越しで音が鳴らない](#tailscale-越しで音が鳴らない)
- [リモートデスクトップ(RDP)越しで音が鳴らない](#リモートデスクトップrdp-越しで音が鳴らない)

### アドオン管理 UI 関連
- [アドオン管理画面でトグルが反映されない](#アドオン管理画面でトグルが反映されない)
- [ペルソナ別設定でペルソナが追加できない](#ペルソナ別設定でペルソナが追加できない)
- [参照音声のアップロードが失敗する](#参照音声のアップロードが失敗する)
- [モバイルでアドオン管理モーダルがスクロールできない](#モバイルでアドオン管理モーダルがスクロールできない)
- [ペルソナ別設定のラベルが縦一列に改行される](#ペルソナ別設定のラベルが縦一列に改行される)

### SSE / ネットワーク関連
- [`[addon-events proxy] upstream fetch failed` が連続する](#addon-events-proxy-upstream-fetch-failed-が連続する)
- [ページリロードで再生ボタンが準備中に戻る](#ページリロードで再生ボタンが準備中に戻る)
- [ModuleNotFoundError: No module named 'lameenc'](#modulenotfounderror-no-module-named-lameenc)

---

## セットアップ関連

### setup.bat が途中で止まる

**診断**: エラーメッセージを確認。以下のどれに該当するか:

| エラーメッセージ | 原因 | 対処 |
|---|---|---|
| `SAIVerse virtual environment not found` | 本体の `.venv` 未作成 | SAIVerse 本体の `setup.bat` を先に実行 |
| `Failed building wheel for opencc` | Windows C++ ビルド失敗 | [opencc セクション](#opencc-のビルドエラーwindows)参照 |
| `Failed building wheel for editdistance` | Python 3.13+ 問題 | [editdistance セクション](#editdistance-のビルドエラーpython-313)参照 |
| `OSError: [WinError 5]` | DLL ロック | [DLL アクセス拒否](#dll-アクセス拒否エラーpip-install-時)参照 |
| `error: subprocess-exited-with-error` | pip install 中の個別パッケージ失敗 | エラー直前のパッケージ名を確認して個別対処 |

**一般的な再試行手順**:
```powershell
# SAIVerse の全 Python プロセスを停止
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 再実行
cd <SAIVerse>\expansion_data\saiverse-voice-tts
setup.bat
```

### opencc のビルドエラー(Windows)

```
error: [WinError 5] アクセスが拒否されました。
Failed to build opencc
```

**原因**: `opencc` パッケージの C++ ソースビルドが Windows で失敗する。

**解決**: `install_backends.py` の最新版では自動的に以下を行います:
1. `opencc-python-reimplemented`(純 Python 互換パッケージ)を先行インストール
2. GPT-SoVITS の requirements.txt から `opencc` 行を自動コメントアウト

古いバージョンで発生する場合は手動で:
```powershell
pip install opencc-python-reimplemented
# external/GPT-SoVITS/requirements.txt を開き、opencc 行と --no-binary=opencc 行をコメントアウト
pip install -r external\GPT-SoVITS\requirements.txt
```

### editdistance のビルドエラー(Python 3.13+)

```
error: command 'cl.exe' failed: ...
warning C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます
```

**原因**: Python 3.13 以降では `editdistance` の Windows 向けビルド済み wheel が未提供で、MSVC ソースビルドが走る。日本語 Windows の CP932 ロケールだと非 ASCII 文字でコンパイルエラーになる。

**解決**: `install_backends.py` と `setup.bat` の最新版では `CL=/utf-8` 環境変数を自動設定済み。

古いバージョンの場合は手動で:
```powershell
$env:CL = "/utf-8"
pip install editdistance
```

### ModuleNotFoundError が出る

| モジュール名 | 原因 | 対処 |
|---|---|---|
| `sounddevice` | パック依存未インストール | `pip install sounddevice` |
| `soundfile` | 同上 | `pip install soundfile` |
| `lameenc` | MP3 progressive 配信用依存未インストール | `pip install lameenc` |
| `ffmpeg` | GPT-SoVITS 依存未完了 | `pip install ffmpeg-python` |
| `pytorch_lightning` | 同上 | `pip install -r external\GPT-SoVITS\requirements.txt` |
| `TTS_infer_pack` | GPT-SoVITS clone 未完了 | `python scripts\install_backends.py gpt_sovits` |
| `qwen_tts` | Qwen3-TTS 未インストール | サポート外(GPT-SoVITS を使用してください) |

**包括的な修復**:
```powershell
cd <SAIVerse>\expansion_data\saiverse-voice-tts
pip install -r requirements.txt
python scripts\install_backends.py gpt_sovits
```

### DLL アクセス拒否エラー(pip install 時)

```
ERROR: Could not install packages due to an OSError: [WinError 5] ...onnxruntime_providers_shared.dll
```

**原因**: 別の Python プロセスが DLL をロード中でロックしている。

**解決**:
```powershell
# 1. SAIVerse バックエンドを停止(Ctrl+C)
# 2. 残存プロセスも停止
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 3. pip 一時ファイルのゴミを掃除
Remove-Item -Recurse -Force "$env:USERPROFILE\SAIVerse\.venv\Lib\site-packages\~*" -ErrorAction SilentlyContinue

# 4. 再インストール
pip install -r external\GPT-SoVITS\requirements.txt
```

---

## GPU / CUDA 関連

### CUDA が認識されない(`CUDA: False`)

**診断**:
```powershell
# Step 1: GPU ドライバが入っているか
nvidia-smi
# → 表示されない場合: NVIDIA ドライバをインストール

# Step 2: torch のバージョン確認
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# → "+cpu" が含まれていたら CPU 版 torch
```

**原因と対処**(多い順):

| 原因 | 確認方法 | 対処 |
|---|---|---|
| CPU 版 torch がインストールされた | `pip show torch` で `+cpu` | CUDA 版を再導入(下記) |
| NVIDIA ドライバ未インストール | `nvidia-smi` が通らない | [NVIDIA ドライバ](https://www.nvidia.com/drivers)をインストール |
| GPT-SoVITS 依存が torch を上書きした | setup.bat 実行直後に `+cpu` に変わった | CUDA 版を再導入(下記) |
| WSL/Docker 内で GPU 非公開 | `nvidia-smi` が通らない | WSL の GPU パススルー設定を確認 |

**CUDA 版 torch の再導入**(最も多い原因の修復):
```powershell
pip uninstall -y torch torchaudio torchvision
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

CUDA 12.x 環境の場合は `cu121` を、CUDA 11.x なら `cu118` を使用。`nvidia-smi` の右上に表示される CUDA Version を確認:
```
+-------------------------------------------+
| NVIDIA-SMI 5xx.xx    CUDA Version: 12.x   |
+-------------------------------------------+
```

### GPU が複数あるがどちらが使われるかわからない

**診断**:
```powershell
# 全 GPU の状態を表示
nvidia-smi

# Python から見える GPU 一覧
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}, VRAM {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB') for i in range(torch.cuda.device_count())]"
```

**GPT-SoVITS が使う GPU を指定する方法**:

方法1: 環境変数で制限(推奨):
```powershell
# GPU 0 のみ使用
$env:CUDA_VISIBLE_DEVICES = "0"
python main.py city_a

# GPU 1 のみ使用
$env:CUDA_VISIBLE_DEVICES = "1"
python main.py city_a
```

方法2: `config/default.json` のエンジン設定で指定:
```json
{
    "engines": {
        "gpt_sovits": {
            "ref_language": "ja",
            "target_language": "ja"
        }
    }
}
```
GPT-SoVITS は内部で `tts_infer.yaml` の `device: cuda` を使用。特定 GPU を指定するには `CUDA_VISIBLE_DEVICES` が最も確実。

**合成中にどの GPU が使われているか確認**:
```powershell
# 別ターミナルで実行。合成中に VRAM 使用量が増える GPU が対象
nvidia-smi -l 1
```

### GPU 稼働率が 0% のまま

**診断**: 合成中に `nvidia-smi` で確認:
```powershell
nvidia-smi
```

| 状態 | 意味 |
|---|---|
| Memory-Usage が数 GB + GPU-Util 30-100% | 正常に GPU 推論中 |
| Memory-Usage が 899MB 等(最小) + GPU-Util 0% | **GPU に載っていない = CPU 推論** |
| Memory-Usage が数 GB + GPU-Util 0% | モデルはロードされたが推論開始前 or ハング |

**原因と対処**:

1. **CUDA 版 torch でない** → [CUDA が認識されない](#cuda-が認識されないcuda-false)を参照
2. **GPT-SoVITS の `tts_infer.yaml` の device が `cpu` になっている**:
   ```powershell
   # 確認
   Select-String -Path "external\GPT-SoVITS\GPT_SoVITS\configs\tts_infer.yaml" -Pattern "device:"
   ```
   `custom:` セクションの `device: cuda` であることを確認。`cpu` になっていたら `cuda` に変更。
3. **CUDA_VISIBLE_DEVICES が空文字に設定されている**: `echo $env:CUDA_VISIBLE_DEVICES` で確認

### CUDA out of memory (VRAM 不足)

```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**GPT-SoVITS の VRAM 消費目安**: 約 2〜4 GB(モデルロード時)

**対処**:
```powershell
# 1. 他の GPU プロセスを確認
nvidia-smi

# 2. 不要なプロセスを停止(Chrome の GPU プロセス等は多くの VRAM を消費する)

# 3. GPT-SoVITS の is_half を有効化(VRAM 半減、品質微減):
# external/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml の custom セクションで
# is_half: true
```

VRAM が 4GB 未満の GPU では GPT-SoVITS は厳しいため、6GB 以上推奨。

---

## TTS 合成関連

### speak_as_persona が一度も呼ばれない(プレイブック未インポート)

**症状**: TTS に関するログが一切出ない。`speak_as_persona enqueued` も無い。

**原因**: 拡張パックの `playbooks/public/sub_speak.json`（compose → process_body → **tts_speak** の3ノード版）が DB にインポートされておらず、本体デフォルトの2ノード版（compose → process_body のみ）が使われている。

SAIVerse はプレイブックを**ファイルではなく DB から読み込む**ため、拡張パックを配置しただけでは反映されない。

**確認**:
```powershell
# DB 上の sub_speak ノード数を確認
python -c "
from database.session import SessionLocal
from database.models import Playbook
db = SessionLocal()
row = db.query(Playbook).filter(Playbook.name == 'sub_speak').first()
if row:
    import json
    nodes = json.loads(row.nodes_json)
    print(f'nodes: {len(nodes)}')
    for n in nodes:
        print(f'  {n[\"id\"]} ({n[\"type\"]})')
else:
    print('sub_speak not found in DB')
db.close()
"
# 期待: 3ノード (compose, process_body, tts_speak)
# NG: 2ノード (compose, process_body) → tts_speak が無い
```

**解決**:
```powershell
cd <SAIVerse>
.venv\Scripts\activate

# 方法1: sub_speak だけインポート
python scripts\import_playbook.py --file expansion_data\saiverse-voice-tts\playbooks\public\sub_speak.json

# 方法2: 全プレイブック一括(expansion_data 含む)
python scripts\import_all_playbooks.py --force
```

インポート後に SAIVerse を再起動。

**予防**: `setup.bat` の最新版ではセットアップ時にプレイブックインポートを自動実行します。手動で拡張パックを配置した場合は `import_all_playbooks.py --force` を忘れずに。

### 🔊ボタンがずっと「準備中」のまま

**診断フロー**:

```
ステップ1: TTS 合成自体が完了しているか
  → ログで "TTS wav saved" or "TTS streamed wav saved" が出ているか確認
  
ステップ2: message_id が取得できているか
  → ログで "enqueue: job=... message_id=..." を確認
  → message_id=None なら本体側の set_active_message_id 配線問題

ステップ3: addon_metadata に登録されたか
  → ログで "notify_audio_ready: msg=... metadata=True event=True" を確認

ステップ4: SSE イベントが配信されたか
  → ログで "addon_events: emit ... subscribers=N" を確認
  → subscribers=0 なら SSE 接続問題

ステップ5: フロントが受信しているか
  → ブラウザ DevTools Console で [addon-events] ログを確認
```

**ログの確認方法**:
```powershell
$log = (Get-ChildItem $env:USERPROFILE\.saiverse\user_data\logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\backend.log"
Select-String -Path $log -Pattern "enqueue: job=|notify_audio_ready|addon_events.*emit|TTS wav saved|TTS streamed" -Encoding UTF8 | Select-Object -Last 20
```

**各ステップの解決策**:

| ステップ | 原因 | 解決 |
|---|---|---|
| 1 で止まる | TTS エンジンのロードまたは合成が失敗 | [合成が遅い](#合成が極端に遅い2分以上)または[GPU 関連](#gpu--cuda-関連)を参照 |
| 2 で message_id=None | 本体側の ContextVar 配線不備 | SAIVerse 本体を最新版に更新 |
| 3 で metadata=False | `saiverse.addon_metadata` モジュール未提供 | 本体のアドオン基盤が必要 |
| 4 で subscribers=0 | SSE 未接続 | [SSE セクション](#addon-events-proxy-upstream-fetch-failed-が連続する)参照 |
| 5 で未受信 | フロント側の購読問題 | ブラウザ Ctrl+Shift+R でハードリロード |

### 合成が極端に遅い(2分以上)

**診断**:
```powershell
# 合成中に GPU 使用状況を確認
nvidia-smi

# ログで合成時間を確認
Select-String -Path $log -Pattern "TTS wav saved|TTS streamed wav saved|first chunk ready" -Encoding UTF8 | Select-Object -Last 5
```

**原因と対処**:

| 症状 | 原因 | 対処 |
|---|---|---|
| GPU-Util 0%, VRAM 最小 | CPU で推論している | [CUDA 認識](#cuda-が認識されないcuda-false)を確認 |
| GPU-Util 30%+, 初回のみ遅い | モデルロード(正常) | 2回目以降は数秒。初回は10-30秒が正常 |
| 毎回2分以上 | ストリーミング OFF で長文合成 | アドオン設定で「ストリーミング推論」を ON に |
| 参照音声が10秒超 or 3秒未満 | 長さが範囲外 | 3秒以上10秒以内に切り詰める（必須） |

### 合成は成功するが音が鳴らない

**最初に確認**: アドオン管理 UI で「ブラウザ側再生」と「サーバー側再生」のどちらが ON か。

| 設定 | 期待される動作 |
|---|---|
| ブラウザ側再生 ON (既定) | アクティブなブラウザタブから音が鳴る |
| サーバー側再生 ON | バックエンド PC のスピーカーから音が鳴る |
| 両方 OFF | 合成は実行されるが自動再生されない(🔊ボタンで手動再生のみ) |

**合成自体が成功しているか確認**:
```powershell
$log = (Get-ChildItem $env:USERPROFILE\.saiverse\user_data\logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\backend.log"
Select-String -Path $log -Pattern "TTS wav saved|TTS streamed wav saved|first chunk ready" -Encoding UTF8 | Select-Object -Last 3
```
合成ログが出ていない場合は [🔊ボタンがずっと「準備中」のまま](#ボタンがずっと準備中のまま)を参照。

**切り分け**:
- 生成済み wav を直接開いて音があるか確認
  ```powershell
  $wav = Get-ChildItem "$env:USERPROFILE\.saiverse\user_data\voice\out\*.wav" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  start $wav.FullName
  ```
  - 無音 → 参照音声 or `ref_text` が不適切 → [参照音声と異なる声](#参照音声と異なる声で合成される)
  - 音が出る → 再生経路の問題 → 下記のサーバー側 / クライアント側セクションを参照

**サーバー側再生が OFF になっていないか**:
```powershell
python -c "from saiverse.addon_config import get_params; p = get_params('saiverse-voice-tts'); print('server:', p.get('server_side_playback'), '/ client:', p.get('client_side_playback'))"
```
v2.0 以降、サーバー側再生の既定値は **OFF**(Tailscale/リモート運用を想定)。バックエンド PC のスピーカーから直接鳴らしたい場合はアドオン管理 UI で明示的に ON にする必要があります。

### fast-langdetect のキャッシュディレクトリエラー

```
TTS synthesis failed (engine=gpt_sovits): fast-langdetect: Cache directory not found
```

**解決**:
```powershell
mkdir external\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect
```
`install_backends.py` の最新版では自動作成されます。

### 文字化けしたテキストが読み上げられる

**原因**: Markdown のリンク URL (saiverse://... の UUID 等) が除去されずに TTS に渡されている。

**確認**:
```powershell
Select-String -Path $log -Pattern "speak_as_persona enqueued" -Encoding UTF8 | Select-Object -Last 3
```
ログの `len=` と `orig_len=` を比較。差がなければテキストクリーナーが動作していない。

**解決**: 拡張パックを最新版に更新。`tools/speak/text_cleaner.py` が存在するか確認。

### 参照音声と異なる声で合成される

**確認ポイント**:

1. **参照音声の長さ**: 3秒以上10秒以内が必須。範囲外は合成品質が著しく低下する
   ```powershell
   python -c "import soundfile as sf; d,sr = sf.read(r'voice_profiles\samples\_default\ref.wav'); print(f'{len(d)/sr:.1f}秒 {sr}Hz')"
   ```

2. **ref_text と実際の発話内容の一致**: `registry.json` の `ref_text` が参照音声の書き起こしと正確に一致しているか(句読点含む)

3. **参照音声の品質**:
   - 背景雑音・BGM がない
   - リップノイズが少ない
   - 感情の起伏が極端でない(フラット推奨)
   - モノラル推奨(ステレオは片チャンネルのみ使用される場合あり)

4. **エンジン設定**: `registry.json` 内の `params.temperature` が高すぎると声質がブレる。`0.7〜1.0` 推奨

---

## Irodori-TTS 固有

### Irodori で `ModuleNotFoundError: torchcodec`

**症状**: 初回発話時に以下のスタックが出る:
```
ImportError: TorchCodec is required for load_with_torchcodec.
```

**原因**: `torch >= 2.10` 系の `torchaudio.load` は backend として `torchcodec` を要求する。Irodori が ref_wav 読み込みでこれを呼ぶので、未インストールだと失敗する。

**解決**:
```powershell
cd $env:USERPROFILE\SAIVerse
.venv\Scripts\activate
pip install torchcodec
```

`setup.bat` の最新版(2026-04 以降)では自動インストール済み。古いパックで発生した場合は `setup.bat` を再実行するか上記コマンドで追加導入。v0.6.0 以降、アドオンカタログで導入した場合は Irodori-TTS の専用環境 (`envs/irodori-*.txt`) に入るので、この症状は出ない想定 (出たら専用環境の Python で `-m pip install torchcodec`)。

### Irodori で `dtype mismatch` / `F.linear` エラー

**症状**: 合成中に以下のエラー:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**原因**: モデル(`model_precision`)と codec(`codec_precision`)の dtype が不一致。codec の出力を model がそのまま使うので、揃っていないと内部の Linear 層で落ちる。

**確認**:
```powershell
python -c "import json; c = json.load(open(r'config\default.json', encoding='utf-8')); print(c['engines']['irodori'])"
```

**正しい組み合わせ**:

| 用途 | model_precision | codec_device | codec_precision | 備考 |
|---|---|---|---|---|
| **GPU 本番(推奨)** | bf16 | cuda | bf16 | 最速、既定 |
| GPU 品質優先 | fp32 | cuda | fp32 | 少し遅いが安定 |
| CPU 運用 | fp32 | cpu | fp32 | RTF 数倍、非実用的 |

**解決**: `config/default.json` の `engines.irodori` を上表のいずれかの組み合わせにして SAIVerse 再起動。

### Irodori で `Irodori-TTS repository not found`

**症状**: 初回発話時に以下のエラー:
```
RuntimeError: Irodori-TTS repository not found at .../external/Irodori-TTS.
Run: python scripts/install_backends.py irodori
```

**原因**: エンジンに `irodori` を選択したが、`external/Irodori-TTS/` に upstream repo が clone されていない。

**解決**:
```batch
cd %USERPROFILE%\SAIVerse\expansion_data\saiverse-voice-tts
setup.bat irodori
```

または直接:
```batch
python scripts\install_backends.py irodori
```

### Irodori で合成が極端に遅い(RTF 5x 以上)

**症状**: Irodori で「3 秒の音声の合成に 15〜20 秒かかる」。GPU は見えているのに遅い。

**診断**: 内部ステージタイミングをログで確認。`decode_latent` が突出して遅いはず。アドオンカタログで導入した場合、Irodori-TTS のパッケージは専用環境にだけ入っているので、下の `python` は専用環境の Python (`%USERPROFILE%\.saiverse\addon_install\saiverse-voice-tts\envs\irodori\Scripts\python.exe`) に読み替える。
```powershell
python -c "
import sys; sys.path.insert(0, 'external/Irodori-TTS')
from irodori_tts.inference_runtime import SamplingRequest
from tools.speak.engine.irodori import IrodoriEngine
import json
c = json.load(open('config/default.json', encoding='utf-8'))
e = IrodoriEngine(c['engines']['irodori'])
e._lazy_load()
import time
t = time.time()
r = e._runtime.synthesize(SamplingRequest(text='テスト', ref_wav='voice_profiles/samples/_default/ref.wav', num_steps=20, seed=42))
print(f'total={time.time()-t:.2f}s')
for stage, sec in r.stage_timings:
    print(f'  {stage}: {sec*1000:.0f}ms')
"
```

**原因と対処**:

| stage_timing | 状態 | 原因 | 対処 |
|---|---|---|---|
| `decode_latent: 15000ms+` | CPU で codec 動作中 | codec_device が cpu | **config で `codec_device: cuda`, `codec_precision: bf16`** |
| `decode_latent: 250ms` + `sample_rf: 1500ms` | 正常 | fp32 運用 | bf16 にすると 2x 速くなる |

**既定(config/default.json)が codec=cuda bf16 になっているはず**。何かで書き換えられていないか確認。

### Irodori で末尾にゴミ音声が混ざる / 文末で途切れる

**症状**: 合成音声の末尾に意味不明な発声が混入、または文の途中で切れる。

**原因**: `truncation_factor` または `seconds` 予算のバランス不良。

v0.3.0 以降のパックでは `tools/speak/engine/irodori.py` で:
- `truncation_factor = 0.75` (ハードコード、ゴミ抑制の要)
- `seconds = chars * 0.25 + 1.5` (チャンク単位で自動算出)
- `hard_trim = chars * 0.25 + 1.5` (後処理)

がバランスされている。

**確認**: 現行の irodori.py でパラメータを独自上書きしていないか:
```powershell
Select-String -Path "tools\speak\engine\irodori.py" -Pattern "_BUDGET_K|_TRIM_K|_TRUNCATION_FACTOR"
```

想定値:
- `_BUDGET_K = 0.25`, `_BUDGET_MARGIN = 1.5`
- `_TRIM_K = 0.25`, `_TRIM_MARGIN = 1.5`
- `_TRUNCATION_FACTOR = 0.75`

**対処**:
- 文末が切れる → `_TRIM_MARGIN` を 0.5 秒程度上げる
- ゴミが残る → `_TRUNCATION_FACTOR` を 0.70 に下げる(下げ過ぎると逆に切れる)
- `registry.json` の `params.truncation_factor` で個別ペルソナごとに上書き可能

### Irodori で HuggingFace 重みダウンロードが失敗する

**症状**: 初回発話時に HuggingFace へのアクセスで失敗する。

**診断**:
```powershell
# HF キャッシュが存在するか確認
dir "$env:USERPROFILE\.cache\huggingface\hub\models--Aratako--Irodori-TTS-500M-v2"
dir "$env:USERPROFILE\.cache\huggingface\hub\models--Aratako--Semantic-DACVAE-Japanese-32dim"
```

ディレクトリが存在しない、または `snapshots/` 以下が空なら未 DL。

**対処**:
```powershell
# 手動で先に DL しておく
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Aratako/Irodori-TTS-500M-v2', filename='model.safetensors')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Aratako/Semantic-DACVAE-Japanese-32dim', filename='weights.pth')"
```

レート制限に当たっている場合は `HF_TOKEN` 環境変数に HuggingFace トークンを設定すると改善します:
```powershell
$env:HF_TOKEN = "hf_xxxxx..."
```

---

## 音声再生関連(サーバー側)

### サーバー PC のスピーカーから音が出ない

> v2.0 以降、**サーバー側再生の既定値は OFF** です。ブラウザ経由で鳴らす「ブラウザ側再生」が既定になりました。バックエンド PC のスピーカーから直接鳴らしたい場合はこのセクション。

**診断**:
```powershell
$log = (Get-ChildItem $env:USERPROFILE\.saiverse\user_data\logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\backend.log"

# 1. 合成ログ確認
Select-String -Path $log -Pattern "TTS wav saved|TTS streamed" -Encoding UTF8 | Select-Object -Last 3

# 2. sounddevice デバイス一覧
python -c "import sounddevice as sd; print(sd.query_devices())"

# 3. アドオン設定でサーバー側再生が ON か
python -c "from saiverse.addon_config import get_params; p = get_params('saiverse-voice-tts'); print('server_side_playback:', p.get('server_side_playback'))"
```

**対処**:

| 状態 | 対処 |
|---|---|
| `server_side_playback: False` (既定) | アドオン管理 UI で「サーバー側再生」を ON に |
| デバイスが「リモート オーディオ」 | アドオン UI の「サーバー側再生デバイス」で実デバイスを選択 |
| sounddevice not installed | `pip install sounddevice` |
| 合成ログ自体がない | [準備中のまま](#ボタンがずっと準備中のまま)を参照 |

### 音声が途切れる / スタッターする

**v2.0 以降の前提**: ストリーミング再生は MP3 progressive + pub/sub 配信に刷新され、複数コンシューマ間でチャンクが奪い合いになる問題は解消されています。それでも途切れる場合は以下。

**原因と対処**:

| 症状 | 原因 | 対処 |
|---|---|---|
| サーバー側再生のみ途切れる | 合成速度 < 再生速度 | ストリーミング OFF で完了後再生 / GPU 負荷を下げる |
| ブラウザ側再生のみ途切れる | ネットワーク帯域不足(Tailscale 等) | ストリーミング OFF で完了後再生 |
| 両方途切れる | 合成側の問題 | [合成が極端に遅い](#合成が極端に遅い2分以上)を参照 |
| 音声が後半で急停止 | Next.js Route Handler の Range 処理 | 本体を `fix/audio-range-200` 以降に更新(GET 時に Range を剥がして 200 で全量返す挙動が必要) |

**テキスト分割を細かくする**(チャンクが早く出るようにする):
`voice_profiles/registry.json` で `"text_split_method": "cut1"` に変更。

---

## 音声再生関連(クライアント側 / ブラウザ)

### クライアント側再生が全く動かない

**前提**: v2.0 以降の新機能。動作には SAIVerse 本体が以下を提供している必要があります:
- `ui_extensions.client_actions` 宣言対応
- `play_audio` action executor
- Route Handler `/api/addon/[...path]` による `/stream` パススルー

**診断フロー**:

```
ステップ1: ペルソナ発話後、ブラウザ DevTools Network に
    /api/addon/saiverse-voice-tts/stream/<msg_id> または
    /api/addon/saiverse-voice-tts/audio/<msg_id>
  のリクエストが出るか
  → 出ない: ステップ2へ(action が発火していない)
  → 出るが 404: サーバー側で message_id が未登録 → [準備中のまま](#ボタンがずっと準備中のまま)
  → 出るが 200 で音が出ない: ステップ3へ

ステップ2: Console に [client-action] または [play_audio] のログが出ているか
  → 出ない: SSE イベント未受信 → [addon-events proxy](#addon-events-proxy-upstream-fetch-failed-が連続する)
  → エラーログ: 下記表を参照

ステップ3: Network タブで Response の Content-Type を確認
  → audio/mpeg (期待): MP3 progressive OK
  → audio/wav: 古いバックエンド、本体/パック更新が必要
  → text/html: rewrite/Route Handler が効いていない → 本体更新
```

**よくあるエラー**:

| Console メッセージ | 原因 | 対処 |
|---|---|---|
| `NotAllowedError: play() failed...` | autoplay 未解除 | [初回の音声が鳴らない](#モバイル-safari-で初回の音声が鳴らないautoplay)参照 |
| `play_audio: action executor not registered` | 本体が client_actions 未対応 | SAIVerse 本体を最新版に更新 |
| `AbortError: ...` | 連続発話で前の再生がキャンセル(正常) | 無視してよい |
| `NotSupportedError: operation is not supported` | WAV ヘッダが破損(0xFFFFFFFF) | パックを v2.0+(MP3 化済み)に更新 |

**パックの playback modality 確認**:
```powershell
python -c "from saiverse.addon_config import get_params; p = get_params('saiverse-voice-tts'); print('client:', p.get('client_side_playback'), '/ server:', p.get('server_side_playback'))"
```
既定は `client: True / server: False`。

### モバイル Safari で初回の音声が鳴らない(autoplay)

**症状**: PC ブラウザでは鳴るのに、iPhone/iPad の Safari で最初の発話が鳴らない。再生ボタンは「準備完了」になるが音が出ない。

**原因**: iOS Safari の autoplay ポリシー。ページロード後にユーザー操作(タップ)が一度も発生していない状態では、プログラムからの `audio.play()` が拒否されます。

**解決**: **SAIVerse の画面を一度タップしてから**ペルソナに話しかけてください。

- 最初のタップで `HTMLAudioElement` の autoplay unlock が完了します(パック内で silent WAV を同期再生)
- 一度解除されれば、そのタブが開いている間は自動再生が有効になります
- タブを閉じて再度開いた場合は再度タップが必要です

**確認**: DevTools Console に以下のログが出るか:
```
[play_audio] unlock: ok
```
`unlock: fail` が出ている場合は、タップが gesture として認識されていない可能性があります。サイドバー開閉など分かりやすい UI 操作を一度行ってみてください。

**Android Chrome**: 通常は autoplay 制約が緩く、タップ不要で鳴ることが多いです。鳴らない場合は同じ対処で解決します。

### 複数端末を開いていてどの端末で鳴るか

**動作**: ブラウザ側再生 ON の場合、**最後に画面を触った端末**が自動的に「アクティブクライアント」となり、その端末だけで音が鳴ります。

- 複数タブを同じブラウザで開いている場合: BroadcastChannel API で相互通知し、アクティブなタブのみが再生
- 複数端末(PC + スマホ等): 最新のユーザー操作(メッセージ送信、タップ等)を検知した端末がアクティブ

**確認**: ヘッダーの Radio アイコンで可視化されます。
- アイコンが光っている = この端末がアクティブクライアント
- アイコンが薄い = 別端末がアクティブ(この端末では鳴らない)

**想定外の端末で鳴る/鳴らない場合**:
1. 鳴らしたい端末で画面を一度タップ or メッセージを送信 → アクティブになるはず
2. それでも切り替わらない場合はブラウザ側の設定/キャッシュ問題 → Ctrl+Shift+R でハードリロード

**全端末で同時に鳴らしたい場合**: サーバー側再生を ON にして、各端末がサーバーと同じ LAN 上にあることが前提(実質 RDP/Tailscale 経由の聞こえ方に依存)。

### ブラウザの🔊ボタンで過去の発話が再生できない

**診断**(ブラウザ DevTools):

1. **Network タブ**: 🔊ボタンクリック後に `/api/addon/saiverse-voice-tts/audio/...` リクエストが出るか
2. **Console タブ**: `[PlayAudioButton]` 関連のエラーが出ているか

**対処**:

| 症状 | 原因 | 対処 |
|---|---|---|
| Network にリクエストが出ない | ボタンのクリックハンドラ未発火 | Ctrl+Shift+R でハードリロード |
| 404 返却 | `audio_file` メタデータ未登録 or wav ファイル削除済み | 新規発話でテスト(wav は既定 24 時間で GC) |
| 200 だが無音 | キャッシュ問題 | Ctrl+Shift+R でハードリロード |
| `NotAllowedError` | ブラウザ autoplay 制約 | 画面タップ後に再試行 |

**直接再生テスト**(ブラウザアドレスバーに):
```
http://localhost:3000/api/addon/saiverse-voice-tts/audio/<message_id>
```
音声プレイヤーが出て再生できれば配信は正常。ボタン UI の問題。

### ストリーミング再生が途中で止まる(長時間音声)

**症状**: 2〜3 分の発話で、途中(例: 2:05 付近)で再生が止まる。

**過去の既知バグ**(v2.0 で解消済):
1. Next.js の Route Handler が `arrayBuffer()` で完全ボディを待って配信していた → 長時間音声で backend が timeout
2. Range リクエストで 206 Partial Content を返していたが Chrome がストリーム中でも切断を見てしまう
3. WAV Chunk Size ヘッダが 0xFFFFFFFF(ストリーミングの目印)で iOS Safari が拒否

**v2.0 での修正**:
- `/stream` エンドポイントは Route Handler で pump 転送(arrayBuffer 回避)
- 音声/動画 GET は Range を剥がして 200 OK で全量返す
- フォーマットを MP3 progressive に変更

**まだ止まる場合の確認**:
```powershell
# backend ログで "TTS streamed wav saved" が出て合成が完了しているか
Select-String -Path $log -Pattern "TTS streamed wav saved|speak_as_persona enqueued" -Encoding UTF8 | Select-Object -Last 5
```
- 合成完了ログが出ている → ネットワーク/配信側の問題 → 本体を最新版に更新
- 合成完了ログが出ない → 合成が途中で失敗 → GPU OOM 等、上記 [CUDA out of memory](#cuda-out-of-memory-vram-不足)参照

### モバイル Safari でタブが自動リロード/破棄される

**症状**: iPhone Safari で SAIVerse タブを開いていると、数分〜十数分で勝手にリロードされる、または他タブから戻ると空白ページになっている。結果として再生中の音声が途切れる。

**原因**: Next.js の **dev mode (`npm run dev`)** はホットリロード関連のオーバーヘッド(大量の JS チャンク、WebSocket 接続、ソースマップ等)が大きく、iOS Safari のメモリ管理で低優先度タブとして扱われ discard されます。

**解決**: Next.js を **production build** で起動する:
```batch
cd %USERPROFILE%\SAIVerse\frontend
npm run build
npm run start
```

| モード | モバイル運用 |
|---|---|
| `npm run dev` | ❌ iOS Safari がタブを discard する |
| `npm run build && npm run start` | ✅ 安定動作 |

**判定方法**: ブラウザ DevTools Console で開発モードの警告が出ているか。開発用の `__next_f` グローバル等が存在すれば dev mode。

**PC ブラウザで運用する場合は dev mode でも問題ありません**。モバイル運用時のみ production 必須。

### Tailscale 越しで音が鳴らない

**前提チェックリスト**:

1. **ブラウザ側再生が ON** になっているか
   ```powershell
   python -c "from saiverse.addon_config import get_params; print(get_params('saiverse-voice-tts').get('client_side_playback'))"
   ```
2. **Next.js を production build で起動**しているか([モバイル Safari でタブ破棄](#モバイル-safari-でタブが自動リロード破棄される)参照)
3. **Tailscale ホスト名**(`*.ts.net`)または Tailscale IP でアクセスしているか
4. **iOS Safari なら画面を一度タップ**したか([autoplay](#モバイル-safari-で初回の音声が鳴らないautoplay)参照)
5. フロントエンド/バックエンドが **Tailscale ネットワーク上でアクセス可能**か(Windows ファイアウォールで 3000 番/8000 番が開いているか)

**Windows ファイアウォール確認**:
```powershell
# Tailscale サブネットからの受信許可
Get-NetFirewallRule -DisplayName "*saiverse*" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "SAIVerse 3000" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow
New-NetFirewallRule -DisplayName "SAIVerse 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

**HTTPS が必要な機能**: MediaRecorder(マイク入力)は HTTPS が必要で Tailscale の HTTP では動作しません。TTS 再生(`<audio>`)は HTTP で動作します。

### リモートデスクトップ(RDP)越しで音が鳴らない

> **v2.0 からは RDP 不要**: ブラウザ側再生で Tailscale 経由のブラウザから直接音を鳴らせます。RDP は「サーバー側再生を RDP クライアント PC で聞きたい」ケースでのみ使用。

**RDP でサーバー側再生を聞く設定**:

1. mstsc.exe → 「オプションの表示」 → 「ローカル リソース」
2. 「リモート オーディオ」の「設定」
3. 「このコンピューターで再生」を選択して接続

**対処**:
```powershell
# 1. リモート オーディオのデバイス番号を確認
python -c "import sounddevice as sd; print(sd.query_devices())"
# "<" マーク or 名前に「リモート」を含むデバイス番号を控える

# 2. アドオン管理 UI の「サーバー側再生デバイス」で該当デバイスを選択
# 3. 「サーバー側再生」を ON に
```

---

## アドオン管理 UI 関連

### アドオン管理画面でトグルが反映されない

**診断**:
```powershell
# 本体側の addon_config API が提供されているか
python -c "from saiverse.addon_config import get_params; print(get_params('saiverse-voice-tts'))"
```

**対処**:
- `get_params` が `ImportError` → SAIVerse 本体がアドオン基盤未対応。本体を最新版に更新
- `get_params` は通るが値が変わらない → ブラウザで設定変更後に SAIVerse を再起動(設定はランタイムでキャッシュされないが、Toolが新しい pulse で読み直す)

### ペルソナ別設定でペルソナが追加できない

**症状**: 「ペルソナ別設定」のラベルだけ表示され、「+」ボタンがない。

**原因**: `GET /api/people/` エンドポイントが 404 を返している。

**確認**:
```powershell
# ブラウザで直接アクセス
# http://localhost:3000/api/people/
# → JSON 配列が返れば OK、404 ならエンドポイント未実装
```

**対処**: SAIVerse 本体を最新版(`feature/tts-addon-integration-fixes` 以降)に更新。

### 参照音声のアップロードが失敗する

**診断**: アドオン管理 UI で参照音声をアップロードした際のエラーメッセージを確認。

| エラー | 原因 | 対処 |
|---|---|---|
| `File type 'xxx' not allowed` | 対応外のファイル形式 | wav / mp3 / flac / ogg のみ対応 |
| `File too large` | サイズ上限超過 | 50MB 以下のファイルを使用 |
| `Invalid name` | persona_id に特殊文字 | persona_id は英数字・ハイフン・アンダースコアのみ |
| `Addon manifest not found` | addon.json が読めない | 拡張パックの配置を確認 |
| `Upload failed` (汎用) | バックエンドとの通信失敗 | バックエンドが起動しているか確認 |

### モバイルでアドオン管理モーダルがスクロールできない

**症状**: iPhone/iPad の Safari でアドオン管理画面を開くと、Voice TTS の設定項目が画面下部で切れてスクロールできない。

**原因**: SAIVerse 本体側のモーダル実装が、iOS Safari の nested scroll 制約に対応していなかった。具体的には:
- backdrop-filter が内部スクロールを阻害
- React の `onTouchMove` stopPropagation が native scroll を妨害
- nested `overflow-y: auto` が iOS Safari で効かない

**解決**: SAIVerse 本体を `feature/addon-manager-mobile-ui` 以降のバージョンに更新してください。このブランチで以下の修正が入っています:
- オーバーレイ自体をスクロールコンテナに変更
- モバイルで backdrop-filter を無効化
- タッチイベントハンドラを削除して native scroll に委譲

**確認**: 本体が対応済みなら「チュートリアル」画面など他のモーダルでも同様にスクロールできます。チュートリアルモーダルはスクロールできるのに Voice TTS 設定だけスクロールできない場合は、このパックの UI 定義の問題 → パックを最新版に更新。

### ペルソナ別設定のラベルが縦一列に改行される

**症状**: モバイル表示で「サーバー側再生デバイス」「参照音声の書き起こし」等のラベルが 1 文字ずつ縦に折れる。

**原因**: ラベルとフォームの flex レイアウトが幅 400px 以下で破綻。

**解決**: SAIVerse 本体を `feature/addon-manager-mobile-ui` 以降に更新(`@media (max-width: 600px)` でラベルをフォーム上に折り返す CSS が入っています)。

---

## SSE / ネットワーク関連

### `[addon-events proxy] upstream fetch failed` が連続する

**原因**: Next.js の SSE プロキシがバックエンドに接続できない。

**状況別**:

| タイミング | 問題か | 対処 |
|---|---|---|
| バックエンド起動前/起動中 | **正常** | バックエンド起動完了後に自動再接続 |
| バックエンド起動後も継続 | **問題** | 以下を確認 |

**バックエンド起動後も出る場合**:
```powershell
# 1. バックエンドが 8000 番で listen しているか
Test-NetConnection -ComputerName 127.0.0.1 -Port 8000

# 2. /api/addon/events が応答するか
python -c "import requests; r = requests.get('http://127.0.0.1:8000/api/addon/events', stream=True, timeout=3); print(r.status_code, r.headers.get('content-type'))"
# 期待: 200 text/event-stream

# 3. ルーティング順序の問題 (404 "Addon not found")
# → api/main.py で addon_events.router が addon.router より先に登録されているか確認
```

### ページリロードで再生ボタンが準備中に戻る

**原因と対処**:

| 原因 | 確認 | 対処 |
|---|---|---|
| metadata 先読みが動いていない | DevTools Network で `/api/addon/messages/.../metadata` リクエストを確認 | SAIVerse 本体を最新版に更新(page.tsx に prefetch 処理が必要) |
| メタデータが DB に無い | 上記リクエストが 200 で空 | 新規発話で再テスト |
| wav ファイルが GC で削除済み | `~/.saiverse/user_data/addon_data/saiverse-voice-tts/outputs/` (v0.5.x までは `~/.saiverse/user_data/voice/out/`) にファイルがあるか | `config/default.json` の `gc_hours` を延長(既定 24 時間) |

### ModuleNotFoundError: No module named 'lameenc'

**症状**: バックエンドログまたは `/stream` アクセス時に:
```
ModuleNotFoundError: No module named 'lameenc'
```

**原因**: v2.0 で MP3 progressive 配信用に追加した依存 `lameenc` が未インストール。

**解決**:
```batch
cd %USERPROFILE%\SAIVerse
.venv\Scripts\activate
pip install lameenc
```

または `setup.bat` を再実行すれば `requirements.txt` から自動インストールされます。

**確認**:
```powershell
python -c "import lameenc; e = lameenc.Encoder(); e.set_bit_rate(128); e.set_in_sample_rate(32000); e.set_channels(1); print('lameenc OK')"
```

**注意**: `lameenc` は PyPI wheel に libmp3lame がバンドルされているため、Windows でも追加システム依存(MSYS2 や vcpkg 等)は不要です。

---

## 共通の診断コマンド集

### バックエンドログの監視(リアルタイム)
```powershell
$log = (Get-ChildItem $env:USERPROFILE\.saiverse\user_data\logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\backend.log"
Get-Content -Path $log -Wait -Tail 0 -Encoding UTF8 | Select-String -Pattern "speak_as_persona|TTS wav|TTS synthesis|first chunk|notify_audio_ready|addon_events.*emit"
```

### GPU 状態の継続監視
```powershell
nvidia-smi -l 2
```

### アドオン設定の現在値
```powershell
python -c "from saiverse.addon_config import get_params; import json; print(json.dumps(get_params('saiverse-voice-tts'), indent=2, ensure_ascii=False))"
```

### ペルソナ別設定の確認
```powershell
python -c "from saiverse.addon_config import get_params; import json; print(json.dumps(get_params('saiverse-voice-tts', persona_id='Yui_city_a'), indent=2, ensure_ascii=False))"
```

### Tool 登録状況
```powershell
python -c "from tools import TOOL_REGISTRY; print('speak_as_persona:', 'speak_as_persona' in TOOL_REGISTRY, '/ total:', len(TOOL_REGISTRY))"
```

### 生成済み wav ファイル一覧(最新5件)
```powershell
Get-ChildItem "$env:USERPROFILE\.saiverse\user_data\voice\out\*.wav" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | Format-Table Name, Length, LastWriteTime
```

### 保存済みメタデータの確認
```powershell
python -c "
from database.models import AddonMessageMetadata
from database.session import SessionLocal
db = SessionLocal()
rows = db.query(AddonMessageMetadata).filter(AddonMessageMetadata.addon_name == 'saiverse-voice-tts').order_by(AddonMessageMetadata.created_at.desc()).limit(5).all()
for r in rows:
    print(f'msg={r.message_id} key={r.key} value={(r.value or \"\")[:60]}')
db.close()
"
```
