from cxrag.core.types import Chunk, Query
from cxrag.generation import ExtractiveGenerator


def _ctx():
    return [
        Chunk(chunk_id="a", doc_id="rag.md", text="RAG 的核心动机是解决大语言模型的知识截止与幻觉问题。模型在回答之前，先从知识库检索相关片段。"),
        Chunk(chunk_id="b", doc_id="x.md", text="今天天气不错，与主题无关的一句话用于测试过滤。"),
    ]


def test_extractive_relevant():
    gen = ExtractiveGenerator(top_sentences=2)
    ans = gen.generate(Query(text="RAG 解决什么问题？"), _ctx())
    assert ans.generator == "extractive-textrank"
    assert "RAG" in ans.text
    assert "天气" not in ans.text  # 无关句被过滤


def test_extractive_empty_context():
    from cxrag.core.errors import CXError

    try:
        ExtractiveGenerator().generate(Query(text="x"), [])
        assert False
    except CXError as e:
        assert e.code == "E_GEN"
