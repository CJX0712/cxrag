"""学习排序重排器（LearningReranker）。

用检索阶段的 bm25 / dense / rrf 分数作为特征，训练轻量逻辑回归对候选重排。
无标注数据时退化为恒等重排（保持检索阶段顺序），保证离线 demo 仍可跑。
"""
from __future__ import annotations

import math
from typing import List

from ..core.base import BaseIndex, BaseReranker, BaseRetriever
from ..core.errors import CXError, CODES
from ..core.types import Query, RetrievalResult


class LearningReranker(BaseReranker):
    def __init__(self):
        self.model = None
        self._scaler = None
        self._fitted = False

    @staticmethod
    def _features(res: RetrievalResult) -> List[float]:
        s = res.source_scores
        bm25 = s.get("bm25", 0.0)
        dense = s.get("dense", 0.0)
        rrf = s.get("rrf", 0.0)
        return [math.log1p(max(bm25, 0.0)), math.log1p(max(dense, 0.0)), rrf]

    def fit(self, dataset) -> "LearningReranker":
        """dataset: dict = {"index": BaseIndex, "retriever": BaseRetriever, "gold": [...]}。

        gold 元素: {"query": str, "relevant": [chunk_id,...]}。
        若正样本不足，退化为未训练（恒等重排）。
        """
        index: BaseIndex = dataset["index"]
        retriever: BaseRetriever = dataset["retriever"]
        gold = dataset.get("gold", [])
        if not gold:
            self._fitted = False
            return self
        X, y = [], []
        for item in gold:
            q = Query(text=item["query"])
            rel = set(item.get("relevant", []))
            cands = retriever.retrieve(q, k=max(20, len(rel) * 3 + 5))
            for r in cands:
                X.append(self._features(r))
                y.append(1 if r.chunk.chunk_id in rel else 0)
        if sum(y) == 0 or sum(y) == len(y):
            self._fitted = False
            return self
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X)
        self.model = LogisticRegression().fit(self._scaler.transform(X), y)
        self._fitted = True
        return self

    def rerank(self, results: List[RetrievalResult], query: Query) -> List[RetrievalResult]:
        if not self._fitted or not results:
            return results
        X = self._scaler.transform([self._features(r) for r in results])  # type: ignore[union-attr]
        proba = self.model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        for r, p in zip(results, proba):
            r.score = float(p)
            r.source_scores = {**r.source_scores, "rerank": float(p)}
        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i
        return ranked
