"""``jieba_fast.posseg`` の顔をした ``jieba.posseg`` への中継 (__init__.py を参照)。"""
from jieba.posseg import *  # noqa: F401,F403

import jieba.posseg as _posseg


def __getattr__(name):
    return getattr(_posseg, name)
