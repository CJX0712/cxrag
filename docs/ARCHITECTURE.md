# 系统架构（ARCHITECTURE）

作者：晨星

## 1. 设计原则

- **单一职责**：每个模块只暴露最小必要接口（`core/base.py` 的 ABC）。
- **接口契约**：所有跨模块数据统一走 `core/types.py` 的 dataclass；异常统一为 `CXError(code)`。
- **可独立验证**：每个模块都有单元测试 + 最小可运行示例（`tests/`）。
- **可组合**：`pipeline.CxRAG` 把各模块串成一条端到端链路。

## 2. 模块划分与接口

| 模块 | 文件 | 输入 → 输出 | 关键接口 |
|---|---|---|---|
| 类型 | `core/types.py` | — | `Document / Chunk / Query / RetrievalResult / Answer` |
| 错误码 | `core/errors.py` | — | `CXError(code, msg)`，`CODES` |
| 基类 | `core/base.py` | — | `BaseLoader/BaseChunker/BaseIndex/BaseRetriever/BaseReranker/BaseGenerator/BaseEvaluator` |
| 分词 | `core/tokenize.py` | 文本 → tokens | `tokenize()`：英文词 + 中文 bigram |
| 加载 | `ingestion/loaders.py` | 路径 → `List[Document]` | `TextLoader/MarkdownLoader/DirectoryLoader.load()` |
| 切分 | `ingestion/chunker.py` | `List[Document]` → `List[Chunk]` | `SentenceChunker.chunk()` |
| 稀疏索引 | `index/bm25_index.py` | chunks → 可查 | `BM25Index.build/sparse_search/get_chunk` |
| 稠密索引 | `index/dense_index.py` | chunks → 可查 | `DenseIndex.build/dense_search`（TF-IDF→SVD→FAISS） |
| 混合索引 | `index/hybrid_index.py` | chunks → 可查 | `HybridIndex`（组合 sparse+dense） |
| 检索 | `retrieval/hybrid_retriever.py` | `Query` → `List[RetrievalResult]` | `HybridRetriever.retrieve()`（RRF） |
| 重排 | `retrieval/reranker.py` | results+gold → 重排 | `LearningReranker.fit/rerank` |
| 生成 | `generation/extractive.py` | `Query+context` → `Answer` | `ExtractiveGenerator.generate()`（TextRank） |
| LLM | `generation/llm.py` | prompt → text | `LLMProvider/OpenAIProvider/LlamaCppProvider` |
| 评测 | `evaluation/metrics.py` | gold → metrics | `recall_at_k / mrr / ndcg_at_k / rouge / Evaluator` |
| 编排 | `pipeline.py` | 配置 → 服务 | `CxRAG.index_path/ask/evaluate/run_demo` |
| 入口 | `cli.py` | argv → 动作 | `index / query / demo / eval` |

## 3. 数据流

```
Document ──load──▶ Document ──chunk──▶ Chunk
                                         │
                    ┌────────────────────┼─────────────────────┐
                    ▼                     ▼                      ▼
              BM25Index             DenseIndex              (HybridIndex 组合)
                    │ sparse_scores       │ dense_scores
                    └─────────┬───────────┘
                              ▼  RRF 融合 (HybridRetriever)
                        RetrievalResult[]
                              │
                              ▼  LearningReranker.rerank（可选）
                        Top-K Chunk
                              │
                              ▼  ExtractiveGenerator / LLMProvider
                           Answer
                              │
                              ▼  Evaluator（gold 对照）
                          metrics
```

## 4. 关键选型与依据

| 选型 | 候选 | 结论 | 理由 |
|---|---|---|---|
| 稀疏检索 | BM25 vs TF-IDF | **rank_bm25** | 词频饱和+长度归一，IR SOTA 基线，零依赖、确定性强 |
| 稠密向量 | FAISS vs numpy | **faiss-cpu**（失败回退 numpy） | 工业级 ANN；离线用 LSA(TF-IDF→SVD) 生成稠密向量，无需下载权重 |
| 融合 | RRF vs 加权 | **RRF** (Cormack 2011) | 无需调权、对两路排序鲁棒 |
| 重排 | 交叉编码器 vs 学习排序 | **逻辑回归特征重排** | 离线可训，避免下载交叉编码器权重 |
| 生成 | LLM vs 抽取式 | **TextRank(离线默认)+LLM 接口** | networkx 无监督抽取；LLM 按需接入 |
| 评测 | 自研 vs 标准 | **IR 公式 + 轻量 ROUGE** | recall@k/MRR/nDCG 为标准公式；ROUGE 自实现避免 nltk 联网 |

> **禁止从零自研的说明**：稠密语义向量本应来自预训练句向量模型（如 BGE），但本环境无 GPU、HuggingFace 权重不可达，
> 故采用 **LSA（TF-IDF→TruncatedSVD）** 这一成熟降维方法离线生成稠密向量，作为可运行替代；后续接入句向量模型只需替换 `DenseIndex` 的编码器，接口不变。

## 5. 错误码

`E_LOAD / E_CHUNK / E_INDEX_EMPTY / E_INDEX_BUILD / E_RETRIEVE / E_GEN / E_CONFIG / E_DEP`
调用方依据 `e.code` 做分支处理。
