"""clean_text_for_tts: 声に出さない記法の除去。"""
import sys
import unittest
from pathlib import Path

_SPEAK_DIR = Path(__file__).resolve().parent.parent / "tools" / "speak"
if str(_SPEAK_DIR) not in sys.path:
    sys.path.insert(0, str(_SPEAK_DIR))

from text_cleaner import clean_text_for_tts  # noqa: E402


class BodyEmoteTests(unittest.TestCase):
    def test_marker_is_not_read_aloud(self):
        self.assertEqual(
            clean_text_for_tts('おはよう。{"body_emote": "wave"}今日もいい天気だね。'),
            "おはよう。今日もいい天気だね。",
        )

    def test_spacing_variants_and_unknown_values_are_removed(self):
        self.assertEqual(clean_text_for_tts('{ "body_emote" : "smile" }やあ'), "やあ")
        self.assertEqual(clean_text_for_tts('やあ{"body_emote": "dance"}'), "やあ")
        self.assertEqual(clean_text_for_tts('やあ{"body_emote": 3}'), "やあ")

    def test_marker_only_text_becomes_empty(self):
        self.assertEqual(clean_text_for_tts('{"body_emote": "nod"}'), "")

    def test_other_braces_are_kept(self):
        self.assertEqual(clean_text_for_tts('{"name": "エア"} という形'), '{"name": "エア"} という形')


if __name__ == "__main__":
    unittest.main()
