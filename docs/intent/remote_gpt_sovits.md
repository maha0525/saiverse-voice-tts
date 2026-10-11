# Intent: 外部 GPT-SoVITS サーバーで合成する

**ステータス**: 検証待ち (実装・隔離テスト済み、実サーバー・再生は未検証)
**依頼**: Antigravity front / 80cb3f17-6f18-4423-b096-a1a3549114a9

利用者が Voice TTS の設定で外部サーバーを選ぶと、ローカルモデルを起動せず、
そのサーバーで音声を合成する。URL が空なら従来のローカル合成を使う。
ペルソナごとの声と、ブラウザ・バックエンド・vessel に渡る音声の対応を守る。

## 入力から再生まで

addon.json の全体設定とペルソナ別 URL → worker の実効設定 → エンジン選択 →
ペルソナ ID に対応する参照音声プロファイルの確認・未登録時のアップロード →
合成 HTTP 応答の WAV デコード → 既存の文ごとの連結・保存・配信。
エンジン選択とキャッシュは worker、HTTP と音声デコードはリモートエンジンが担う。
URL・タイムアウトの変更時は旧エンジンを閉じて選び直す。

## 不変条件

- 本番 DB・ペルソナ・会話・世界状態にテスト入力を送らず、LLM を呼ばない。
- URL 設定時はローカル推論への暗黙の切替を行わない。
- profile_id は synthesize の persona_id、またはエンジン設定の profile_id / persona_id。
  ID がなければ失敗として知らせ、複数ペルソナを同じ既定 ID に混ぜない。
- 参照音声と書き起こしが揃っている場合、404 / exists=false のみ登録する。
  登録済みプロファイルを上書きしない。サーバーに既存プロファイルがあれば参照音声なしでも使える。
- GET /health、GET/PUT /profiles/{id}、POST /synthesize は Skyrim TtsClient と同じ契約。
- 通信・HTTP・登録拒否・不正 WAV の失敗はログと RuntimeError で worker へ伝える。
- サーバーは完成 WAV を一括で返す。synthesize_stream はそれを一チャンクとして渡し、
  既存の sub_seq / finalize 処理で文ごとに連結する。サーバーの逐次推論とは表現しない。

## 検証

requests のモックと合成 WAV で未登録 / 登録済み / 通信失敗を確かめる。
合成人格・一時出力先で UI 実効設定 → worker → HTTP → WAV 保存まで通し、
URL 切替・タイムアウト変更・参照音声なし・複数文の連結を確かめる。
既存アドオンテストと変更 Python の ruff check を実行する。
NEBULA の実サーバー接続と実際の音質・再生は、この変更の隔離テストの対象外。

## 実施結果 (2026-10-12)

- 新規テスト 19 件成功。参照音声の未登録時 PUT、登録済み時のスキップ、
  明示的な health、HTTP / タイムアウト / 不正応答、ローカル選択を検証。
- UI 設定ローダ → worker → モック HTTP → WAV 保存の隔離経路を検証。
  2 文を 480 フレーム / 24kHz の WAV に連結し、別ペルソナ ID / URL の送信、
  タイムアウト変更、URL を空に戻したローカル復帰も確認。
- 全スイート 126 件中 123 件成功。変更前 HEAD の隔離コピーは
  107 件中 104 件成功で、失敗は両方とも test_regenerate.py の同じ 3 件:
  test_404_when_message_not_found / test_history_fallback_partial_body /
  test_history_fallback_when_body_missing_context。
  いずれも manager が未取得として HTTP 400 を返す既存テストの失敗。
- 変更 Python 6 ファイルの ruff check と git diff --check は成功。
- サンドボックス内では既存の偽子プロセス通信が停止したため中断し、
  全スイートはサンドボックス外で実行した。HTTP はモック、人格は合成データ、
  保存先は一時ディレクトリ。本番への書き込み・外部 LLM 呼び出しは行っていない。
