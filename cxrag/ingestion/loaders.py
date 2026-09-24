"""文档加载器：纯文件读取，无外部依赖。

支持 .txt / .md；DirectoryLoader 递归扫描目录。PDF 等二进制格式留作扩展点。
"""
from __future__ import annotations

import os
from typing import List

from ..core.base import BaseLoader
from ..core.errors import CXError, CODES
from ..core.types import Document


class TextLoader(BaseLoader):
    """加载单个纯文本文件。"""

    def load(self, path: str) -> List[Document]:
        if not os.path.isfile(path):
            raise CXError("E_LOAD", f"文件不存在: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        doc_id = os.path.basename(path)
        return [Document(doc_id=doc_id, text=text, source=path)]


class MarkdownLoader(TextLoader):
    """Markdown 视为纯文本加载（保留原文，切分器再做语义切分）。"""


class DirectoryLoader(BaseLoader):
    """递归加载目录下所有 .txt / .md。"""

    def __init__(self, exts: tuple[str, ...] = (".txt", ".md")):
        self.exts = exts

    def load(self, path: str) -> List[Document]:
        if not os.path.isdir(path):
            raise CXError("E_LOAD", f"目录不存在: {path}")
        docs: List[Document] = []
        for root, _dirs, files in os.walk(path):
            for fn in sorted(files):
                if fn.lower().endswith(self.exts):
                    full = os.path.join(root, fn)
                    docs.extend(TextLoader().load(full))
        if not docs:
            raise CXError("E_LOAD", f"目录中无匹配文件: {path} ({self.exts})")
        return docs
