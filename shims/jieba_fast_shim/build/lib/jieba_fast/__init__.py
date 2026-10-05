"""jieba_fast の顔をした、純 Python の jieba への中継。

GPT-SoVITS は中国語のテキスト処理で ``jieba_fast`` (jieba の C 拡張による
高速化フォーク。API は同じ) を無条件に import する (``text/chinese.py`` 等 —
日本語の合成でも import は走る)。jieba_fast は PyPI に wheel が無く sdist
だけなので、C コンパイラの無い利用者の環境では pip が失敗する。

この中継パッケージは、純 Python の jieba を ``jieba_fast`` の名前でそのまま
提供する。GPT-SoVITS が使うのは ``setLogLevel`` / ``posseg.lcut`` /
``cut_for_search`` だけ (固定 commit 2d9193b を grep して確認) で、どれも
jieba に同じ名前である。中国語の分かち書きが C 拡張より遅くなるだけで、
結果は変わらない。opencc を opencc-python-reimplemented に差し替えたのと
同じ理屈 (scripts/install_backends.py の _strip_opencc_from_requirements)。
"""
from jieba import *  # noqa: F401,F403

import jieba as _jieba


def __getattr__(name):
    return getattr(_jieba, name)
