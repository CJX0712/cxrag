"""Pipeline：编排全链路 index → retrieve → rerank → generate → evaluate。

单一入口，保证「模块可组合成完整可运行链路」。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .core.errors import CXError, CODES
from .core.types import Answer, Chunk, Query
from .evaluation import Evaluator
from .generation import ExtractiveGenerator
from .index import HybridIndex
from .ingestion import DirectoryLoader, SentenceChunker, TextLoader
from .retrieval import HybridRetriever, LearningReranker


class CxRAG:
    def __init__(
        self,
        dim: int = 256,
        top_n: int = 30,
        reranker: Optional[LearningReranker] = None,
        generator: Optional[ExtractiveGenerator] = None,
    ):
        self.index = HybridIndex(dim=dim)
        self.retriever = HybridRetriever(self.index, top_n=top_n)
        self.reranker = reranker or LearningReranker()
        self.generator = generator or ExtractiveGenerator()

    # ---------- 建库 ----------
    def index_documents(self, docs) -> int:
        chunks = SentenceChunker().chunk(docs)
        self.index.build(chunks)
        return self.index.size

    def index_path(self, path: str) -> int:
        loader = DirectoryLoader() if os.path.isdir(path) else TextLoader()
        return self.index_documents(loader.load(path))

    # ---------- 重排训练 ----------
    def _expand(self, gold: List[dict]) -> List[dict]:
        out = []
        for item in gold:
            rel = set(item.get("relevant", []))
            for d in item.get("relevant_docs", []):
                rel |= {c.chunk_id for c in self.index._chunks.values() if c.doc_id == d}
            out.append({"query": item["query"], "relevant": list(rel)})
        return out

    def fit_reranker(self, gold: List[dict]) -> bool:
        expanded = self._expand(gold)
        self.reranker.fit({"index": self.index, "retriever": self.retriever, "gold": expanded})
        return self.reranker._fitted

    # ---------- 问答 ----------
    def ask(self, query_text: str, k: int = 5, rerank: bool = True):
        q = Query(text=query_text)
        results = self.retriever.retrieve(q, k=k)
        if rerank:
            results = self.reranker.rerank(results, q)
        context = [r.chunk for r in results[:k]]
        answer = self.generator.generate(q, context)
        answer.meta = {
            **answer.meta,
            "results": [
                {"chunk_id": r.chunk.chunk_id, "score": round(r.score, 4), "snippet": r.chunk.text[:100]}
                for r in results[:k]
            ],
        }
        return answer, results

    # ---------- 评测 ----------
    def evaluate(self, gold: List[dict], ks=(1, 3, 5, 10)) -> dict:
        expanded = self._expand(gold)
        return Evaluator(self.retriever, ks=ks).evaluate({"queries": expanded})

    # ---------- Demo ----------
    def run_demo(self, corpus_dir: str, gold_path: str) -> dict:
        t0 = time.time()
        n = self.index_path(corpus_dir)
        build_time = time.time() - t0

        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)["queries"]
        expanded = self._expand(gold)

        t1 = time.time()
        ir = Evaluator(self.retriever, ks=(1, 3, 5, 10)).evaluate({"queries": expanded})
        eval_time = time.time() - t1

        fitted = self.fit_reranker(gold)

        # 样本问答
        samples = []
        for q in gold[:3]:
            ans, _ = self.ask(q["query"], k=5)
            samples.append({"query": q["query"], "answer": ans.text, "generator": ans.generator})

        return {
            "chunks": n,
            "backend": self.index.dense.backend,
            "build_time_s": round(build_time, 3),
            "eval_time_s": round(eval_time, 3),
            "reranker_fitted": fitted,
            "ir_metrics": ir["aggregate"],
            "samples": samples,
        }
