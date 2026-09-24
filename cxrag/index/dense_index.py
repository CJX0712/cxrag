"""稠密索引：TF-IDF → TruncatedSVD(LSA) → 余弦检索。

离线生成稠密语义向量（无需下载任何模型权重）。优先使用 FAISS（工业级 ANN），
不可用时回退 numpy 精确检索，接口完全一致，保证可复现。
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from ..core.base import BaseIndex
from ..core.errors import CXError, CODES
from ..core.tokenize import tokenize
from ..core.types import Chunk

try:  # 优先复用 FAISS
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - 依赖可选
    faiss = None
    _HAS_FAISS = False


class DenseIndex(BaseIndex):
    def __init__(self, dim: int = 256):
        self.dim = dim
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._pipe: Pipeline | None = None
        self._matrix: np.ndarray | None = None
        self._faiss = None
        self.backend = "faiss" if _HAS_FAISS else "numpy"

    def build(self, chunks: List[Chunk]) -> "DenseIndex":
        if not chunks:
            raise CXError("E_INDEX_BUILD", "chunks 为空")
        self._chunks = {c.chunk_id: c for c in chunks}
        self._order = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        self._pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize,
                        token_pattern=None,
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=min(self.dim, max(2, len(texts) - 1)))),
                ("norm", Normalizer(copy=False)),
            ]
        )
        self._matrix = self._pipe.fit_transform(texts).astype("float32")
        if self.backend == "faiss":  # pragma: no cover - 依赖可选
            idx = faiss.IndexFlatIP(self._matrix.shape[1])
            idx.add(self._matrix)
            self._faiss = idx
        return self

    def _embed(self, query: str) -> np.ndarray:
        if self._pipe is None:
            raise CXError("E_INDEX_EMPTY", "请先 build")
        v = self._pipe.transform([query]).astype("float32")
        return v

    def sparse_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return []

    def dense_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise CXError("E_INDEX_EMPTY", "请先 build")
        q = self._embed(query)
        if self.backend == "faiss":  # pragma: no cover - 依赖可选
            scores, ids = self._faiss.search(q, k)
            out = []
            for score, i in zip(scores[0], ids[0]):
                if i < 0:
                    continue
                out.append((self._order[i], float(score)))
            return out
        sims = self._matrix @ q[0]
        top = np.argsort(-sims)[:k]
        return [(self._order[int(i)], float(sims[i])) for i in top]

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise CXError("E_INDEX_EMPTY", f"未知 chunk_id: {chunk_id}")
        return self._chunks[chunk_id]

    @property
    def size(self) -> int:
        return len(self._order)
