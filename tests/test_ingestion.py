from cxrag.core.errors import CXError
from cxrag.ingestion import DirectoryLoader, SentenceChunker, TextLoader

CORPUS = "cxrag/demo/corpus"


def test_text_loader():
    docs = TextLoader().load(f"{CORPUS}/rag.md")
    assert len(docs) == 1
    assert "RAG" in docs[0].text


def test_directory_loader():
    docs = DirectoryLoader().load(CORPUS)
    assert len(docs) == 5


def test_chunker_produces_chunks():
    docs = DirectoryLoader().load(CORPUS)
    chunks = SentenceChunker(max_tokens=80).chunk(docs)
    assert len(chunks) > 5
    assert all(c.chunk_id for c in chunks)


def test_loader_missing_file():
    try:
        TextLoader().load("nope.md")
        assert False, "should raise"
    except CXError as e:
        assert e.code == "E_LOAD"
