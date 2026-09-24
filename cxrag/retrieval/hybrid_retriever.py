"""RRF（Reciprocal Rank Fusion）融合检索。

Cormack et al. 2011：对多路排序按 1/(k+rank) 求和，无需调权、鲁棒。
"""
from __future__ import annotations

from typing import Dict, List

from ..core.base import BaseIndex, BaseRetriever
from ..core.errors import CXError, CODES
from ..core.types import Chunk, Query, RetrievalResult

_RRF_K = 60


class HybridRetriever(BaseRetriever):
    def __init__(self, index: BaseIndex, top_n: int = 30):
        self.index = index
        self.top_n = top_n  # 每路召回候选数

    def retrieve(self, query: Query, k: int) -> List[RetrievalResult]:
        if not query or not query.text.strip():
            raise CXError("E_RETRIEVE", "空查询")
        sparse = self.index.sparse_search(query.text, self.top_n)
        dense = self.index.dense_search(query.text, self.top_n)
        fused: Dict[str, float] = {}
        src: Dict[str, dict] = {}
        for ranking in (sparse, dense):
            for rank, (cid, score) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
                src.setdefault(cid, {})["rrf"] = fused[cid]
                if ranking is sparse:
                    src[cid]["bm25"] = score
                else:
                    src[cid]["dense"] = score
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: List[RetrievalResult] = []
        for rank, (cid, score) in enumerate(ordered):  # type: ignore[arg-type]
            chunk = self.index.get_chunk(cid)
            results.append(
                RetrievalResult(chunk=chunk, score=float(score), rank=rank, source_scores=src.get(cid, {}))
            )
        return results
