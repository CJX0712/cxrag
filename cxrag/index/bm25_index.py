"""稀疏索引：BM25（rank_bm25）。

业界 IR 基线，零依赖、确定性强。score 越大越相关。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from ..core.base import BaseIndex
from ..core.errors import CXError, CODES
from ..core.tokenize import tokenize
from ..core.types import Chunk


class BM25Index(BaseIndex):
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
        self._order: List[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: List[Chunk]) -> "BM25Index":
        if not chunks:
            raise CXError("E_INDEX_BUILD", "chunks 为空")
        self._chunks = {c.chunk_id: c for c in chunks}
        self._order = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        return self

    def sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            raise CXError("E_INDEX_EMPTY", "请先 build")
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._order[i], float(scores[i])) for i in top]

    def dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        # 稀疏索引无稠密能力，返回空（由 HybridIndex 组合稠密索引）
        return []

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise CXError("E_INDEX_EMPTY", f"未知 chunk_id: {chunk_id}")
        return self._chunks[chunk_id]

    @property
    def size(self) -> int:
        return len(self._order)
