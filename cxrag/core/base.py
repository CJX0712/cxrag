"""抽象基类（ABC）：定义各模块的统一接口契约。

单一职责：每个模块只暴露最小必要的抽象方法；具体实现可替换、可独立测试。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .types import Answer, Chunk, Document, Query, RetrievalResult


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """加载文件/目录为 Document 列表。"""


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: List[Document]) -> List[Chunk]:
        """把 Document 切成可检索的 Chunk。"""


class BaseIndex(ABC):
    @abstractmethod
    def build(self, chunks: List[Chunk]) -> "BaseIndex":
        """构建索引（幂等：重复调用重建）。"""

    @abstractmethod
    def sparse_search(self, query: str, k: int) -> List[tuple[str, float]]:
        """稀疏检索，返回 [(chunk_id, score)]，score 越大越相关。"""

    @abstractmethod
    def dense_search(self, query: str, k: int) -> List[tuple[str, float]]:
        """稠密检索，返回 [(chunk_id, score)]，余弦相似度。"""

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk:
        """按 id 取回 Chunk。"""

    @property
    @abstractmethod
    def size(self) -> int:
        """索引中的 chunk 数量。"""


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: Query, k: int) -> List[RetrievalResult]:
        """端到端检索（含融合），返回有序结果。"""


class BaseReranker(ABC):
    @abstractmethod
    def fit(self, dataset) -> "BaseReranker":
        """用带标注数据训练（无标注时退化为恒等重排）。"""

    @abstractmethod
    def rerank(self, results: List[RetrievalResult], query: Query) -> List[RetrievalResult]:
        """对检索结果重排。"""


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: Query, context: List[Chunk]) -> Answer:
        """基于上下文生成答案。"""


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, dataset) -> dict:
        """对数据集评测，返回指标字典。"""
