"""抽取式生成器：基于 TextRank 的无监督抽取（networkx 实现）。

离线默认生成器。对检索到的上下文做句子级 TextRank 排序，
选出与查询词重叠度高、且中心性高的句子拼接成答案；同时给出摘要。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

import networkx as nx

from ..core.base import BaseGenerator
from ..core.errors import CXError, CODES
from ..core.tokenize import tokenize
from ..core.types import Answer, Chunk, Query


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？!?；;.\n])", text) if s.strip()]


def _textrank_sentences(sentences: List[str], damping: float = 0.85, iters: int = 50) -> List[float]:
    """返回每个句子的中心性分数。"""
    n = len(sentences)
    if n <= 1:
        return [1.0] * n
    g = nx.Graph()
    g.add_nodes_from(range(n))
    toks = [set(tokenize(s)) for s in sentences]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(toks[i] & toks[j])
            if inter:
                # 相似度按 Jaccard
                union = len(toks[i] | toks[j]) or 1
                w = inter / union
                g.add_edge(i, j, weight=w)
    # 相似度按 Jaccard
    ranks = nx.pagerank(g, alpha=damping, max_iter=iters, weight="weight")
    return [ranks.get(i, 0.0) for i in range(n)]


class ExtractiveGenerator(BaseGenerator):
    def __init__(self, top_sentences: int = 3, summary_sentences: int = 4):
        self.top = top_sentences
        self.sum_n = summary_sentences

    def generate(self, query: Query, context: List[Chunk]) -> Answer:
        if not context:
            raise CXError("E_GEN", "生成上下文为空")
        q_tok = set(tokenize(query.text))
        sent_pool: List[tuple[str, float, float, float, bool]] = []  # (sentence, centrality, overlap, rank_bonus, chunk_relevant)
        for ci, ch in enumerate(context):
            rank_bonus = 1.0 / (1.0 + ci)  # 检索越靠前权重越高
            sents = _sentences(ch.text)
            scores = _textrank_sentences(sents)
            # 该 chunk 是否含查询词，决定答案句是否可来自此 chunk
            chunk_rel = any(len(q_tok & set(tokenize(s))) > 0 for s in sents)
            for s, sc in zip(sents, scores):
                overlap = len(q_tok & set(tokenize(s))) / (len(q_tok) or 1)
                sent_pool.append((s, sc, overlap, rank_bonus, chunk_rel))
        # 答案：优先来自含查询词的 chunk；不足再用其他 chunk 的高中心性句补齐
        scored = sorted(sent_pool, key=lambda x: (x[2] * 3.0 + x[1] * x[3]), reverse=True)
        relevant = [x for x in scored if x[4]] + [x for x in scored if not x[4]]
        seen = set()
        answer_sents: List[str] = []
        for s, _c, _o, _b, _r in relevant:
            key = s[:24]
            if key in seen:
                continue
            seen.add(key)
            answer_sents.append(s)
            if len(answer_sents) >= self.top:
                break
        # 摘要：仅按 centrality
        sum_scored = sorted(sent_pool, key=lambda x: x[1], reverse=True)
        seen2 = set()
        summary_sents: List[str] = []
        for s, _c, _o, _b, _r in sum_scored:
            key = s[:24]
            if key in seen2:
                continue
            seen2.add(key)
            summary_sents.append(s)
            if len(summary_sents) >= self.sum_n:
                break
        text = "".join(answer_sents) if answer_sents else "".join(summary_sents[: self.top])
        return Answer(
            text=text,
            query=query.text,
            context=context,
            generator="extractive-textrank",
            meta={
                "summary": "".join(summary_sents),
                "answer_sentences": len(answer_sents),
            },
        )
