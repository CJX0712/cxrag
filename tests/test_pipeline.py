import json
import os

from cxrag.pipeline import CxRAG

CORPUS = "cxrag/demo/corpus"
GOLD = "cxrag/demo/gold.json"


def test_pipeline_demo_runs():
    rag = CxRAG(dim=32)
    report = rag.run_demo(CORPUS, GOLD)
    assert report["chunks"] > 0
    assert report["ir_metrics"]["recall@10"] >= 0.9
    assert report["ir_metrics"]["mrr"] >= 0.9
    assert len(report["samples"]) == 3


def test_pipeline_query_returns_answer():
    rag = CxRAG(dim=32)
    rag.index_path(CORPUS)
    ans, results = rag.ask("向量数据库的作用是什么？", k=5)
    assert ans.text
    assert results
    assert any("向量" in r.chunk.text for r in results[:3])


def test_pipeline_reranker_improves_or_equal():
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)["queries"]
    rag = CxRAG(dim=32)
    rag.index_path(CORPUS)
    before = rag.evaluate(gold)["aggregate"]["recall@5"]
    rag.fit_reranker(gold)
    after = rag.evaluate(gold)["aggregate"]["recall@5"]
    assert after >= before - 1e-9
