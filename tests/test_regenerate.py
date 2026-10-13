"""Tests for the POST /regenerate endpoint in api_routes.py.

エンドポイントは TTS 合成を裏で enqueue するだけなので、検証ポイントは:

- body から text/persona_id を受け取れば直接使う (source=body)
- 不足時は manager.building_histories から逆引き (source=history)
- 引き当て不能なら 404
- text/persona_id 両方欠ければ 400
- enqueue_tts に message_id が明示で渡る

playback_worker は host の tool loader が ``tools._loaded.speak.playback_worker``
として sys.modules に登録するため、テスト時は同じパスにダミーモジュールを
注入して enqueue_tts 呼び出しを検証する。
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Pack root を sys.path に追加
_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


class _FakeManager:
    """manager.building_histories だけ持つテスト用ダミー。"""

    def __init__(self, building_histories: Dict[str, List[Dict[str, Any]]]):
        self.building_histories = building_histories


def _make_app(manager: Any) -> FastAPI:
    """import api_routes してそのまま router を mount したテスト用 FastAPI app。"""
    if "api_routes" in sys.modules:
        del sys.modules["api_routes"]
    import api_routes  # type: ignore

    # _get_manager_dep() は遅延 import を返すので、ここで manager を返す
    # ダミー dependency でオーバーライドする。
    app = FastAPI()
    app.include_router(api_routes.router, prefix="/api/addon/saiverse-voice-tts")

    async def _dep() -> Any:
        return manager

    # 既に登録されている dependency を上書きする (Depends factory が返した object を解決)
    # api_routes 側で `Depends(_get_manager_dep())` の形なので、対象 dependency 関数を
    # ルートから収集して上書きする。
    # app.routes ではなく api_routes.router.routes を走査する: FastAPI 0.141 以降、
    # include_router した経路は app.routes 上で _IncludedRouter に包まれ、
    # APIRoute (と dependant) が平たく並ばなくなったため。
    for route in api_routes.router.routes:
        deps = getattr(route, "dependant", None)
        if deps is None:
            continue
        for sub in list(deps.dependencies):
            # depends factory の戻り値関数で manager を返すように差し替える
            app.dependency_overrides[sub.call] = _dep
    # 差し替え先が一つも見つからないと、本物の get_manager が走ってテストの
    # 前提が黙って崩れる。FastAPI の内部構造がまた変わったときに即座に気づけるよう、
    # ここで明示的に落とす。
    if not app.dependency_overrides:
        raise RuntimeError("manager dependency not found on api_routes.router routes")
    return app


class RegenerateEndpointTests(unittest.TestCase):
    """エンドポイントの挙動検証。host tool loader を模して
    ``tools._loaded.speak.playback_worker`` にダミーモジュールを注入する。"""

    def setUp(self):
        self.fake_manager = _FakeManager(
            building_histories={
                "Cafe_city_a": [
                    {
                        "message_id": "Cafe_city_a:42",
                        "role": "assistant",
                        "persona_id": "Yui_city_a",
                        "content": "おはよう、まはー!",
                    },
                    {
                        "message_id": "Cafe_city_a:43",
                        "role": "user",
                        "content": "おはよう",
                    },
                ],
            }
        )
        # host の tool loader 模倣: tools._loaded.speak.playback_worker を sys.modules に登録
        self.fake_pw = types.ModuleType("tools._loaded.speak.playback_worker")
        self.mock_enqueue = MagicMock(return_value="JOB-MOCK")
        self.fake_pw.enqueue_tts = self.mock_enqueue  # type: ignore[attr-defined]
        # 親パッケージも一応生やしておく (sys.modules.get は子のみ参照するが、import 経路の整合性のため)
        sys.modules.setdefault("tools", types.ModuleType("tools"))
        sys.modules.setdefault("tools._loaded", types.ModuleType("tools._loaded"))
        sys.modules.setdefault("tools._loaded.speak", types.ModuleType("tools._loaded.speak"))
        sys.modules["tools._loaded.speak.playback_worker"] = self.fake_pw

        self.app = _make_app(self.fake_manager)
        self.client = TestClient(self.app)

    def tearDown(self):
        sys.modules.pop("tools._loaded.speak.playback_worker", None)

    def test_body_path_uses_text_and_persona_directly(self):
        """body に text/persona_id があれば history を見ずにそのまま使う。"""
        self.mock_enqueue.return_value = "JOB-1"
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={
                "message_id": "MSG-X",
                "text": "再合成したいテキスト",
                "persona_id": "Yui_city_a",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["status"], "enqueued")
        self.assertEqual(body["job_id"], "JOB-1")
        self.assertEqual(body["source"], "body")
        self.mock_enqueue.assert_called_once_with(
            text="再合成したいテキスト",
            persona_id="Yui_city_a",
            message_id="MSG-X",
        )

    def test_history_fallback_when_body_missing_context(self):
        """body に text/persona_id が無ければ building_histories から逆引きする。"""
        self.mock_enqueue.return_value = "JOB-2"
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={"message_id": "Cafe_city_a:42"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["source"], "history")
        self.mock_enqueue.assert_called_once_with(
            text="おはよう、まはー!",
            persona_id="Yui_city_a",
            message_id="Cafe_city_a:42",
        )

    def test_history_fallback_partial_body(self):
        """body に text だけあって persona_id が無くても history で補完できる。"""
        self.mock_enqueue.return_value = "JOB-3"
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={
                "message_id": "Cafe_city_a:42",
                "text": "上書きテキスト",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["source"], "history")
        self.mock_enqueue.assert_called_once()
        kwargs = self.mock_enqueue.call_args.kwargs
        self.assertEqual(kwargs["text"], "上書きテキスト")
        self.assertEqual(kwargs["persona_id"], "Yui_city_a")

    def test_404_when_message_not_found(self):
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={"message_id": "missing-id"},
        )
        self.assertEqual(res.status_code, 404)
        self.mock_enqueue.assert_not_called()

    def test_400_when_message_id_missing(self):
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={},
        )
        self.assertEqual(res.status_code, 400)
        self.mock_enqueue.assert_not_called()

    def test_400_when_history_message_has_empty_text(self):
        """history 由来でも text が空文字なら 400。"""
        self.fake_manager.building_histories["B"] = [
            {"message_id": "EMPTY", "persona_id": "P", "content": ""},
        ]
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={"message_id": "EMPTY"},
        )
        self.assertEqual(res.status_code, 400)
        self.mock_enqueue.assert_not_called()

    def test_500_when_playback_worker_module_unavailable(self):
        """host tool loader が pack を未ロード → 500 で明示エラー。"""
        sys.modules.pop("tools._loaded.speak.playback_worker", None)
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/regenerate",
            json={
                "message_id": "MSG-Y",
                "text": "x",
                "persona_id": "P",
            },
        )
        self.assertEqual(res.status_code, 500)
        self.assertIn("playback_worker", res.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
