# CxRAG · 混合检索增强问答引擎

> 作者：**晨星** ｜ License：MIT ｜ 版本：1.0.0

一套**完全离线、CPU 可用、零手工干预即可跑通**的检索增强问答（RAG）系统。
整合业界成熟开源组件，按单一职责模块化，每个模块可独立验证，组合成端到端可运行链路。

## 特性

- 🔀 **混合检索**：稀疏 BM25 + 稠密 LSA/FAISS，RRF 融合，兼顾关键词与语义
- 🧠 **学习重排**：基于检索特征的轻量逻辑回归重排（无标注时退化为恒等，demo 仍可跑）
- 📝 **抽取式生成**：TextRank 无监督抽取，答案 100% 源自检索上下文，**零幻觉**
- 🔌 **LLM 可插拔**：预留 OpenAI / llama-cpp 接口，按需接入生成式大模型
- 📊 **量化评测**：recall@k / MRR / nDCG / ROUGE，自带基线对照
- 📦 **可复现**：版本锁定（`requirements.lock`），一键脚本部署

## 快速开始

```bash
# 方式一：一键（Linux/macOS）
bash setup.sh

# 方式二：一键（Windows）
setup.bat

# 方式三：手动
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.lock
python -m cxrag.cli demo
```

## 命令行

```bash
python -m cxrag.cli demo                                   # 跑通内置 demo + 验收报告
python -m cxrag.cli query "向量数据库的作用是什么？"        # 单次问答
python -m cxrag.cli index <文件或目录> --gold gold.json     # 建库 + 训练重排器
python -m cxrag.cli eval gold.json <数据路径>               # IR 评测
```

## 验收基线（本机，6 条查询 / 10 chunks / FAISS 后端）

| 检索方案 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25-only | 0.833 | 1.000 | 1.000 | 0.918 |
| Dense-only (LSA) | 0.917 | 1.000 | 1.000 | 0.968 |
| **Hybrid (RRF)** | **0.917** | **1.000** | **1.000** | **0.932** |

> 抽取式答案全部源自检索上下文（grounding=100%，零幻觉）；ROUGE 因抽取式与首段参考不同天然偏低，非目标指标。

## 目录结构

```
cxrag/
├── cxrag/                 # 源码包
│   ├── core/              # 类型 / 错误码 / ABC / 分词器
│   ├── ingestion/         # 加载 + 切分
│   ├── index/             # BM25 + 稠密(FAISS) + 混合索引
│   ├── retrieval/         # RRF 融合 + 学习重排
│   ├── generation/        # TextRank 抽取 + LLM 接口
│   ├── evaluation/        # IR + ROUGE 指标
│   ├── pipeline.py        # 编排全链路
│   ├── cli.py             # 命令行入口
│   └── demo/              # 内置语料 + gold 标注
├── tests/                 # 单元 + 集成测试
├── docs/                  # 架构 / 部署 / 使用 文档
├── requirements.txt       # 核心依赖（可读）
├── requirements.lock      # 版本锁定（可复现）
├── pyproject.toml
├── setup.sh / setup.bat   # 一键部署
└── LICENSE
```

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)、[`docs/USAGE.md`](docs/USAGE.md)。
