from cxrag.core.tokenize import tokenize


def test_tokenize_english_lower():
    assert "rag" in tokenize("RAG System")
    assert "system" in tokenize("RAG System")


def test_tokenize_chinese_chars():
    toks = tokenize("检索增强生成")
    # 中文按二元分词（bigram）
    assert "检索" in toks and "增强" in toks and "生成" in toks
    assert "rag" in tokenize("RAG 系统")


def test_tokenize_empty():
    assert tokenize("") == []
