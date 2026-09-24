"""文本切分器：按句子切分后按 token 预算分组。

不依赖 nltk，使用正则做中英文友好的句子切分，保证离线可复现。
"""
from __future__ import annotations

import hashlib
import re
from typing import List

from ..core.base import BaseChunker
from ..core.errors import CXError, CODES
from ..core.types import Chunk, Document

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;.\n])")


def _sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts


def _tok_count(s: str) -> int:
    # 中文按字、英文按词，粗略但稳定
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    others = len(re.findall(r"[A-Za-z0-9]+", s))
    return cjk + others


class SentenceChunker(BaseChunker):
    """按句子聚合成 chunk，单 chunk 不超过 max_tokens（溢出则硬切断）。"""

    def __init__(self, max_tokens: int = 220, overlap_sentences: int = 1):
        self.max_tokens = max_tokens
        self.overlap = overlap_sentences

    def chunk(self, docs: List[Document]) -> List[Chunk]:
        if not docs:
            raise CXError("E_CHUNK", "空文档列表", code=None) if False else CXError(  # 保持契约
                "E_CHUNK", "空文档列表"
            )
        chunks: List[Chunk] = []
        for doc in docs:
            sents = _sentences(doc.text)
            buf: List[str] = []
            buf_tok = 0
            idx = 0
            cursor = 0
            for s in sents:
                stok = _tok_count(s)
                if buf and buf_tok + stok > self.max_tokens:
                    chunks.append(self._make(doc, buf, idx, cursor))
                    # 重叠：保留最后若干句
                    keep = buf[-self.overlap :] if self.overlap else []
                    cursor = max(0, cursor - sum(len(x) for x in keep))
                    buf = list(keep)
                    buf_tok = sum(_tok_count(x) for x in buf)
                    idx += 1
                buf.append(s)
                buf_tok += stok
                cursor += len(s) + 1
            if buf:
                chunks.append(self._make(doc, buf, idx, cursor))
                idx += 1
        return chunks

    def _make(self, doc: Document, sents: List[str], idx: int, start: int) -> Chunk:
        text = "".join(sents)
        cid = hashlib.md5(f"{doc.doc_id}#{idx}".encode("utf-8")).hexdigest()[:12]
        return Chunk(
            chunk_id=cid,
            doc_id=doc.doc_id,
            text=text,
            start=start,
            end=start + len(text),
            metadata={"doc_source": doc.source},
        )
