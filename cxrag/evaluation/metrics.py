"""量化指标实现（离线、无 nltk 依赖）。

- IR 指标：recall@k、MRR、nDCG@k（标准公式）
- 文本指标：ROUGE-1/2/L（自实现轻量版，避免 nltk 联网下载）
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ..core.base import BaseEvaluator, BaseRetriever
from ..core.types import Query


def recall_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = list(retrieved[:k])
    hits = len(set(top) & relevant)
    return hits / len(relevant)


def mrr(relevant: set, retrieved: Sequence[str]) -> float:
    for i, cid in enumerate(retrieved):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: List[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    top = list(retrieved[:k])
    rels = [1.0 if c in relevant else 0.0 for c in top]
    ideal = _dcg([1.0] * min(len(relevant), k))
    if ideal <= 0:
        return 0.0
    return _dcg(rels) / ideal


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] or [tuple(tokens)]


def rouge_n(hyp: str, ref: str, n: int) -> float:
    from ..core.tokenize import tokenize

    a, b = tokenize(hyp), tokenize(ref)
    if len(a) < n or len(b) < n:
        return 0.0
    hg, rg = _ngrams(a, n), _ngrams(b, n)
    overlap = len(set(hg) & set(rg))
    prec = overlap / len(hg) if hg else 0.0
    rec = overlap / len(rg) if rg else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l(hyp: str, ref: str) -> float:
    from ..core.tokenize import tokenize

    a, b = tokenize(hyp), tokenize(ref)
    if not a or not b:
        return 0.0
    l = _lcs_len(a, b)
    prec = l / len(a)
    rec = l / len(b)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def rouge(hyp: str, ref: str) -> Dict[str, float]:
    return {"rouge1": rouge_n(hyp, ref, 1), "rouge2": rouge_n(hyp, ref, 2), "rougeL": rouge_l(hyp, ref)}


class Evaluator(BaseEvaluator):
    """基于检索器对 gold 数据集评测。

    dataset 结构：
      {"queries": [{"query": str, "relevant": [chunk_id,...]}],
       "summaries": [{"hyp": str, "reference": str}]}  # 可选 ROUGE
    """

    def __init__(self, retriever: BaseRetriever, ks: Sequence[int] = (1, 3, 5, 10)):
        self.retriever = retriever
        self.ks = tuple(ks)

    def evaluate(self, dataset: dict) -> dict:
        gold = dataset.get("queries", [])
        if not gold:
            raise ValueError("dataset 缺少 queries")
        per_query = []
        for item in gold:
            q = Query(text=item["query"])
            res = self.retriever.retrieve(q, k=max(self.ks))
            retrieved = [r.chunk.chunk_id for r in res]
            rel = set(item.get("relevant", []))
            row: Dict[str, object] = {"query": item["query"], "relevant": len(rel)}
            for k in self.ks:
                row[f"recall@{k}"] = recall_at_k(rel, retrieved, k)
            row["mrr"] = mrr(rel, retrieved)
            row["ndcg@10"] = ndcg_at_k(rel, retrieved, 10)
            per_query.append(row)

        agg: Dict[str, float] = {}
        n = len(per_query) or 1
        for k in self.ks:
            agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / n  # type: ignore[arg-type]
        agg["mrr"] = sum(r["mrr"] for r in per_query) / n  # type: ignore[arg-type]
        agg["ndcg@10"] = sum(r["ndcg@10"] for r in per_query) / n  # type: ignore[arg-type]

        if "summaries" in dataset:
            rg = [rouge(s["hyp"], s["reference"]) for s in dataset["summaries"]]
            for key in ("rouge1", "rouge2", "rougeL"):
                agg[key] = sum(x[key] for x in rg) / (len(rg) or 1)

        return {"per_query": per_query, "aggregate": agg}
