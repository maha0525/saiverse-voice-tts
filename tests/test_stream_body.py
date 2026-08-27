"""Tests for ``_stream_body`` in api_routes.py — the consumer wait must stay bounded.

無期限の ``q.get()`` を ``run_in_executor`` に渡すと、その worker thread は
「作業中」のまま executor へ戻らない。結果として二つが同時に壊れる:

  1. クライアント切断でリクエストが cancel されても待っている thread は残る
     (別 thread で走る同期処理に asyncio の cancel は届かない)。
  2. インタプリタ終了時、``concurrent.futures`` が ``atexit`` より前に走らせる
     ``_python_exit`` がその thread を join し続けるため、ホストが ``atexit`` に
     登録した後始末へ到達できず、プロセスが終了しない。

2026-08-27、SAIVerse バックエンドが Ctrl+C で終了しなくなった実害がこれで、
``close_stream`` まで到達しなかったストリームの待ちが 3 本残っていた。

audio_stream は host の tool loader が ``tools._loaded.speak.audio_stream``
として sys.modules に登録するため、テストでは同じパスにダミーを注入する。
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

# Pack root を sys.path に追加
_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

_AUDIO_STREAM_PATH = "tools._loaded.speak.audio_stream"


def _install_fake_audio_stream(q: "Queue") -> None:
    """``subscribe`` が渡した queue をそのまま返すダミーを注入する。"""
    mod = types.ModuleType(_AUDIO_STREAM_PATH)
    mod.subscribe = lambda message_id: q  # type: ignore[attr-defined]
    mod.has_stream = lambda message_id: True  # type: ignore[attr-defined]
    sys.modules[_AUDIO_STREAM_PATH] = mod


class StreamBodyConsumerWaitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved = sys.modules.get(_AUDIO_STREAM_PATH)

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop(_AUDIO_STREAM_PATH, None)
        else:
            sys.modules[_AUDIO_STREAM_PATH] = self._saved

    async def test_pending_stream_does_not_pin_executor_thread(self) -> None:
        """終了印が来ないストリームを cancel しても executor worker が解放される。

        これが本命の回帰テスト。旧実装 (``run_in_executor(None, q.get)``) では
        worker が queue で永久に寝るため、最後の shutdown が返らない。
        """
        # 何も入れない = close_stream が一度も来なかったストリーム
        q: "Queue" = Queue()
        _install_fake_audio_stream(q)

        import api_routes  # noqa: E402

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-exec")
        asyncio.get_running_loop().set_default_executor(executor)

        gen = api_routes._stream_body("msg-pending")
        task = asyncio.create_task(gen.__anext__())
        # worker が q.get に入るまで待つ
        await asyncio.sleep(0.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        try:
            await gen.aclose()
        except RuntimeError:
            # cancel 済みの generator は既に終了している場合がある
            pass

        # worker が queue へ戻っていれば shutdown(wait=True) は即返る。
        # 旧実装ではここが永久にブロックし、wait が False になる。
        finished = threading.Event()
        threading.Thread(
            target=lambda: (executor.shutdown(wait=True), finished.set()),
            daemon=True,
        ).start()
        self.assertTrue(
            finished.wait(timeout=5.0),
            "executor worker is still pinned by the consumer wait",
        )

    async def test_yields_frames_until_sentinel(self) -> None:
        """通常経路: frame を順に yield し、sentinel (None) で終わる。"""
        q: "Queue" = Queue()
        q.put(b"frame-1")
        q.put(b"frame-2")
        q.put(None)
        _install_fake_audio_stream(q)

        import api_routes  # noqa: E402

        chunks = [chunk async for chunk in api_routes._stream_body("msg-done")]

        self.assertEqual(chunks, [b"frame-1", b"frame-2"])

    async def test_waits_across_gaps_longer_than_the_poll_interval(self) -> None:
        """poll 間隔より長い無音を挟んでも frame を取りこぼさない。

        待ちを刻んだことで frame を落とすようになっていないかの確認。
        """
        q: "Queue" = Queue()
        _install_fake_audio_stream(q)

        import api_routes  # noqa: E402

        async def _feed_late() -> None:
            await asyncio.sleep(api_routes._CONSUMER_POLL_INTERVAL * 2.5)
            q.put(b"late-frame")
            q.put(None)

        feeder = asyncio.create_task(_feed_late())
        chunks = [chunk async for chunk in api_routes._stream_body("msg-slow")]
        await feeder

        self.assertEqual(chunks, [b"late-frame"])

    async def test_missing_stream_ends_without_yielding(self) -> None:
        """subscribe が None を返したら (stream 破棄済み) 何も yield せず終わる。"""
        mod = types.ModuleType(_AUDIO_STREAM_PATH)
        mod.subscribe = lambda message_id: None  # type: ignore[attr-defined]
        mod.has_stream = lambda message_id: False  # type: ignore[attr-defined]
        sys.modules[_AUDIO_STREAM_PATH] = mod

        import api_routes  # noqa: E402

        chunks = [chunk async for chunk in api_routes._stream_body("msg-gone")]

        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
