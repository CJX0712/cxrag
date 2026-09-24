"""共享分词器：中英文友好，离线可用。

- 英文/数字：按词（小写）
- 中文：按字符二元分词（bigram），中文 IR 标准做法，能正确度量语义重叠，
  避免单字分词导致的「共享一字即判相关」噪声。
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    toks: list[str] = list(_WORD.findall(t))
    for run in _CJK.findall(t):
        if len(run) == 1:
            toks.append(run)
        else:
            for i in range(len(run) - 1):
                toks.append(run[i : i + 2])
    return toks
