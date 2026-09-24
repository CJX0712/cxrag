"""混合索引：组合稀疏(BM25) + 稠密(LSA/FAISS)。

对外暴露统一 BaseIndex 接口；检索层调用 sparse_search / dense_search 后做 RRF 融合。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..core.base import BaseIndex
from ..core.errors import CXError, CODES
from ..core.types import Chunk
from .bm25_index import BM25Index
from .dense_index import DenseIndex


class HybridIndex(BaseIndex):
    def __init__(self, dim: int = 256):
        self.sparse = BM25Index()
        self.dense = DenseIndex(dim=dim)
        self._chunks: Dict[str, Chunk] = {}

    def build(self, chunks: List[Chunk]) -> "HybridIndex":
        if not chunks:
            raise CXError("E_INDEX_BUILD", "chunks 为空")
        self.sparse.build(chunks)
        self.dense.build(chunks)
        self._chunks = {c.chunk_id: c for c in chunks}
        return self

    def sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        return self.sparse.sparse_search(query, k)

    def dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        return self.dense.dense_search(query, k)

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise CXError("E_INDEX_EMPTY", f"未知 chunk_id: {chunk_id}")
        return self._chunks[chunk_id]

    @property
    def size(self) -> int:
        return len(self._chunks)
