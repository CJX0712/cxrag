from cxrag.evaluation import ndcg_at_k, recall_at_k, rouge


def test_recall_at_k():
    rel = {"a", "b", "c"}
    assert recall_at_k(rel, ["a", "b", "x"], k=3) == 2 / 3
    assert recall_at_k(rel, ["a", "b", "c"], k=3) == 1.0


def test_ndcg():
    rel = {"a", "b"}
    assert ndcg_at_k(rel, ["a", "b", "c"], k=3) > 0.9
    assert ndcg_at_k(rel, ["c", "x", "y"], k=3) == 0.0


def test_rouge_perfect():
    r = rouge("检索增强生成", "检索增强生成")
    assert r["rouge1"] == 1.0
    assert r["rougeL"] == 1.0


def test_rouge_partial():
    r = rouge("RAG 检索", "RAG 生成")
    assert 0.0 < r["rouge1"] < 1.0
