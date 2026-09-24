"""命令行入口：index / query / demo / eval。

用法示例见 DEPLOYMENT.md / USAGE.md。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .generation import ExtractiveGenerator
from .index import HybridIndex
from .pipeline import CxRAG
from .retrieval import HybridRetriever, LearningReranker

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CORPUS = os.path.join(_HERE, "demo", "corpus")
_DEFAULT_GOLD = os.path.join(_HERE, "demo", "gold.json")


def _build(args) -> CxRAG:
    rag = CxRAG(dim=args.dim)
    print(f"[index] 加载: {args.path}")
    n = rag.index_path(args.path)
    print(f"[index] 已建库 chunks={n} 稠密后端={rag.index.dense.backend}")
    if args.gold and os.path.isfile(args.gold):
        with open(args.gold, "r", encoding="utf-8") as f:
            gold = json.load(f)["queries"]
        fitted = rag.fit_reranker(gold)
        print(f"[index] 重排器训练: {'已训练' if fitted else '退化为恒等(无/弱标注)'}")
    return rag


def cmd_index(args):
    _build(args)


def cmd_query(args):
    rag = _build(args)
    ans, results = rag.ask(args.query, k=args.k)
    print("\n=== 答案 ===")
    print(ans.text)
    print("\n=== 检索依据 (top{}) ===".format(args.k))
    for r in results[: args.k]:
        print(f"  [{r.score:.4f}] {r.chunk.doc_id} :: {r.chunk.text[:80]}")


def cmd_eval(args):
    rag = _build(args)
    metrics = rag.evaluate(json.load(open(args.gold, encoding="utf-8"))["queries"])
    print("\n=== IR 评测指标 ===")
    print(json.dumps(metrics["aggregate"], ensure_ascii=False, indent=2))


def cmd_demo(args):
    rag = CxRAG(dim=args.dim)
    print(f"[demo] 语料: {_DEFAULT_CORPUS}")
    report = rag.run_demo(_DEFAULT_CORPUS, _DEFAULT_GOLD)
    print("\n========== CxRAG Demo 验收报告 ==========")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(prog="cxrag", description="CxRAG 混合检索增强问答引擎")
    p.add_argument("--dim", type=int, default=256, help="稠密向量维度")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="构建索引")
    pi.add_argument("path", help="文件或目录")
    pi.add_argument("--gold", default="", help="gold 标注(json)用于训练重排器")
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="单次问答")
    pq.add_argument("query", help="问题")
    pq.add_argument("path", nargs="?", default="", help="可选：先建库的数据路径")
    pq.add_argument("--gold", default="", help="可选 gold 用于重排")
    pq.add_argument("--k", type=int, default=5)
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", help="IR 评测")
    pe.add_argument("gold", help="gold 标注(json)")
    pe.add_argument("path", help="数据路径")
    pe.set_defaults(func=cmd_eval)

    pd = sub.add_parser("demo", help="跑通内置 demo 并输出验收报告")
    pd.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    if getattr(args, "cmd", None) == "query" and not args.path:
        # demo 语料兜底
        args.path = _DEFAULT_CORPUS
        args.gold = _DEFAULT_GOLD
    if getattr(args, "cmd", None) == "query" and not args.gold:
        args.gold = _DEFAULT_GOLD if os.path.isfile(_DEFAULT_GOLD) else ""
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
