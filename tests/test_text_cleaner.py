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


class UserOnlyTests(unittest.TestCase):
    # SAIVerse 本体 sea/runtime_llm.py が吹き出しの文に入れる形 (再生成ボタンは
    # この文をそのまま送ってくる)。
    PLAYBOOK_BLOCK = (
        '<user_only alt="Playbook 起動">\n'
        "/spell name='run_playbook' args={\"name\": \"generate_image\"}\n"
        '<details class="spellResult"><summary class="spellSummary">'
        '<span class="spellIcon"><svg width="14" height="14"><path d="M12 2z"/></svg></span>'
        "<span>Playbook 起動</span></summary>\n\n"
        "画像「ルシア・スピネルの究極全身像」の生成が完了しました。\n\n"
        "モデル: gpt_image_2_5_sunburst\n\n"
        "</details>\n"
        "</user_only>\n"
    )

    def test_spell_block_is_not_read_aloud(self):
        text = "描いてみるね。\n" + self.PLAYBOOK_BLOCK + "どうかな、気に入った？"
        cleaned = clean_text_for_tts(text)
        self.assertNotIn("spell", cleaned)
        self.assertNotIn("生成が完了しました", cleaned)
        self.assertNotIn("Playbook", cleaned)
        self.assertIn("描いてみるね。", cleaned)
        self.assertIn("どうかな、気に入った？", cleaned)

    def test_message_of_only_a_spell_block_becomes_empty(self):
        self.assertEqual(clean_text_for_tts(self.PLAYBOOK_BLOCK), "")

    def test_two_blocks_keep_the_words_between_them(self):
        text = self.PLAYBOOK_BLOCK + "まずひとつ。" + self.PLAYBOOK_BLOCK + "もうひとつ。"
        # 囲みの後ろの改行は残る (読み上げには影響しない)
        self.assertEqual(clean_text_for_tts(text), "まずひとつ。\nもうひとつ。")

    def test_in_heart_is_not_read_aloud(self):
        self.assertEqual(
            clean_text_for_tts("うん。<in_heart>本当は少し不安</in_heart>大丈夫だよ。"),
            "うん。大丈夫だよ。",
        )


if __name__ == "__main__":
    unittest.main()
