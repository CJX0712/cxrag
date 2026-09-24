import pytest

from cxrag.core.errors import CXError
from cxrag.index import BM25Index, DenseIndex, HybridIndex
from cxrag.ingestion import DirectoryLoader, SentenceChunker

CHUNKS = SentenceChunker().chunk(DirectoryLoader().load("cxrag/demo/corpus"))


def _chunks():
    return SentenceChunker().chunk(DirectoryLoader().load("cxrag/demo/corpus"))


def test_bm25_build_and_search():
    idx = BM25Index().build(_chunks())
    assert idx.size == len(CHUNKS)
    hits = idx.sparse_search("RAG 检索", k=3)
    assert len(hits) == 3
    assert all(0.0 <= s for _, s in hits)


def test_bm25_empty():
    with pytest.raises(CXError):
        BM25Index().sparse_search("x", k=1)


def test_dense_backend_and_search():
    idx = DenseIndex(dim=32).build(_chunks())
    assert idx.backend in ("faiss", "numpy")
    hits = idx.dense_search("向量数据库 语义检索", k=3)
    assert len(hits) == 3


def test_hybrid_search():
    idx = HybridIndex(dim=32).build(_chunks())
    s = idx.sparse_search("Transformer", k=5)
    d = idx.dense_search("Transformer", k=5)
    assert s and d
    chunk = idx.get_chunk(s[0][0])
    assert chunk.doc_id
