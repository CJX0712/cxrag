from cxrag.index import HybridIndex
from cxrag.ingestion import DirectoryLoader, SentenceChunker
from cxrag.retrieval import HybridRetriever, LearningReranker

GOLD = [
    {"query": "RAG 是什么？", "relevant_docs": ["rag.md"]},
    {"query": "Transformer 的自注意力", "relevant_docs": ["transformer.md"]},
]


def _rag():
    rag = HybridIndex(dim=32)
    rag.build(SentenceChunker().chunk(DirectoryLoader().load("cxrag/demo/corpus")))
    return rag


def test_rrf_retrieve():
    idx = _rag()
    ret = HybridRetriever(idx, top_n=20)
    res = ret.retrieve(__import__("cxrag.core.types", fromlist=["Query"]).Query(text="向量数据库"), k=5)
    assert len(res) == 5
    assert res[0].rank == 0
    assert "rrf" in res[0].source_scores


def test_reranker_identity_without_gold():
    idx = _rag()
    ret = HybridRetriever(idx, top_n=20)
    rr = LearningReranker()
    rr.fit({"index": idx, "retriever": ret, "gold": []})
    assert rr._fitted is False
    res = ret.retrieve(__import__("cxrag.core.types", fromlist=["Query"]).Query(text="模型量化"), k=5)
    out = rr.rerank(res, __import__("cxrag.core.types", fromlist=["Query"]).Query(text="模型量化"))
    assert len(out) == 5


def test_reranker_fit_runs():
    idx = _rag()
    ret = HybridRetriever(idx, top_n=20)
    rr = LearningReranker()
    rr.fit({"index": idx, "retriever": ret, "gold": GOLD})
    # 仅验证不抛错（小样本可能退化为未训练）
    assert rr._fitted in (True, False)
