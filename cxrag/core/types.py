"""核心数据类型定义。

所有模块之间通过以下 dataclass 传递数据，保证接口稳定、可独立验证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    """原始文档。"""

    doc_id: str
    text: str
    source: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """文档切片，检索的最小单元。"""

    chunk_id: str
    doc_id: str
    text: str
    start: int = 0
    end: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Query:
    text: str
    query_id: str = ""


@dataclass
class RetrievalResult:
    """单次检索命中。"""

    chunk: Chunk
    score: float
    rank: int = 0
    source_scores: Dict[str, float] = field(default_factory=dict)  # bm25 / dense / rrf


@dataclass
class Answer:
    """生成式答案。"""

    text: str
    query: str
    context: List[Chunk] = field(default_factory=list)
    generator: str = ""
    meta: Dict = field(default_factory=dict)
