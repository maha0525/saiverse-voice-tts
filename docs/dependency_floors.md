# Python の依存の下限

## 何をしているか

`requirements.txt` (SAIVerse 本体の venv に入る) と `envs/*.txt` (GPT-SoVITS と
Irodori-TTS の専用環境に入る) は、PyPI から入るパッケージの全行に下限 (`>=` か `==`)
を書いている。下限が無いと、利用者の環境には導入した日の最新版が入ったまま上がらず、
後から出た脆弱性の修正版が届かない。

下限の決め方:

- 脆弱性の修正版があるパッケージは、その修正版を下限にする。依存として入るだけの
  パッケージ (certifi / pillow / aiohttp / protobuf) も、修正版を下限にするために
  直接の行として書く。
- それ以外は、2026-01-01 までに出た版を下限にする。既存の環境が既に満たしている版
  なので、アドオンの更新でこれらのパッケージは動かない。
- `requirements.txt` の numpy / requests / huggingface_hub は、リリース済みの
  SAIVerse の `requirements.lock` の版以下にする (本体の lock が constraints として
  渡るので、上回ると導入が失敗する)。開発中の lock ではなく、利用者が使っている
  リリースの lock と照らす。

`tests/test_dependency_floors.py` が、全行に下限があることと、脆弱性の修正版の下限が
下げられていないことを確かめる。

## 下限が利用者に届く経路

`addon.json` の `setup_version` が上がった版にアドオンを更新すると、SAIVerse 本体は
`setup.steps` を再実行する。`pip install -r` は、下限を満たさないパッケージだけを
引き上げる。**下限を上げたら `setup_version` も上げる** — 上げないと、更新しても
`pip install -r` が走らず、下限は新しく導入する人にしか効かない。

## 下限で直せない脆弱性 (2026-10-22 時点、OSV で確認)

隔離した環境に新しく入れた版を、全パッケージ OSV で照合した結果。

1. **GPT-SoVITS の環境の transformers (4.50.0)**。GPT-SoVITS の requirements が
   `transformers<=4.50` に縛っていて、4.50 以下に修正版の無い脆弱性が残る
   (GHSA-29pf-2h5f-8g72 の修正版は 5.3.0、GHSA-fgcw-684q-jj6r は 5.5.0、
   GHSA-xrqw-3rrv-vx5w は 5.10.0。GHSA-x9r9-c232-4q39 は修正版が無い)。
   直すには、GPT-SoVITS の推論が transformers 5 系で動くかを確かめて上限を外す
   作業が要る。この作業はまだ行っていない。
2. **Irodori-TTS の環境の pillow (12.3.0)**。修正版は 13.0.0 だが、Irodori-TTS の
   依存の gradio が `pillow<13.0` に縛っている。gradio は Irodori-TTS の WebUI
   (`gradio_app.py` 等) だけが使い、推論の入口 (`irodori_tts.inference_runtime`) は
   import しない。`scripts/patch_irodori_deps.py` で pyproject.toml から gradio を外すと
   pillow 13.0.0 が入ることを、隔離した環境で確かめた (`pip check` で壊れた依存なし、
   推論の入口の import も通る)。ただし既存の環境には gradio が残るので、その扱いを
   決めてから入れる。
3. **Irodori-TTS の環境の protobuf (4.25.x)**。CVE-2025-4565 は 4.25.8 で直るので
   下限にしたが、CVE-2026-0994 の修正版は 5.29.6 以降で、descript-audiotools
   (dacvae の依存) が `protobuf<5.0.0` に縛っている。PyPI の descript-audiotools
   0.7.2 は `protobuf<3.20` なので、`envs/irodori-*.txt` は GitHub の 0.7.4 を
   commit 固定で入れている。
4. **専用環境の setuptools (81.0.0)**。修正版は 83.0.0 だが、固定している
   torch 2.11.0 が `setuptools<82` に縛っている。torch の固定を上げるときに一緒に直る。
5. **専用環境の pip (24.3.1)**。専用環境を作る `python -m venv` が入れる版で、
   修正版は 26.2.0 以降。このファイルの下限では動かせない (専用環境を作る SAIVerse
   本体の側の話)。
6. **torch 2.11.0** に LOW の GHSA-rrmf-rvhw-rf47 (修正版 2.13.0)。torch は本体 venv で
   合成実績のある版に固定しているので、固定を上げる作業の中で扱う。
