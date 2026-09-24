# 使用指南（USAGE）

作者：晨星

## 1. 命令行

```bash
# 内置 demo（建库 + 评测 + 样例问答，输出验收报告）
python -m cxrag.cli demo

# 单次问答（自动用内置语料建库）
python -m cxrag.cli query "Transformer 的自注意力机制是怎么工作的？"

# 对自有数据建库（目录递归加载 .txt/.md）
python -m cxrag.cli index ./my_docs --gold ./gold.json

# 对自有数据评测
python -m cxrag.cli eval ./gold.json ./my_docs
```

`gold.json` 结构：

```json
{
  "queries": [
    {"query": "你的问题？", "relevant_docs": ["doc1.md", "doc2.md"]}
  ]
}
```

## 2. Python API

```python
from cxrag.pipeline import CxRAG

rag = CxRAG(dim=256)
rag.index_path("./my_docs")                      # 建库
rag.fit_reranker(gold)                           # 可选：用标注训练重排器
ans, results = rag.ask("你的问题？", k=5)         # 问答
print(ans.text)                                  # 抽取式答案
print(ans.meta["results"])                        # 检索依据（含分数/片段）

metrics = rag.evaluate(gold)                      # IR 评测
print(metrics["aggregate"])
```

## 3. 接入真实生成式 LLM（可选）

```python
from cxrag.generation.llm import OpenAIProvider, LLMGenerator
from cxrag.pipeline import CxRAG

rag = CxRAG()
rag.index_path("./my_docs")
llm = LLMGenerator(OpenAIProvider(api_key="sk-...", model="gpt-4o-mini"))
ans, _ = rag.ask("你的问题？", k=5)
# 用 LLM 生成：把检索上下文拼入 prompt
prompt = LLMGenerator.build_llm_prompt(ans.query, ans.context)
ans.text = llm.provider.complete(prompt)
```

本地 GGUF：`LlamaCppProvider(model_path="xxx.gguf")`（需 `pip install -r requirements-llm.txt`）。

## 4. 评测指标说明

| 指标 | 含义 |
|---|---|
| `recall@k` | 前 k 个结果命中相关文档的比例 |
| `MRR` | 首个相关结果排名的倒数均值 |
| `nDCG@k` | 考虑排序位置的相关性折扣累计增益 |
| `ROUGE` | 生成文本与参考文本的 n-gram/最长公共子序列重叠 |

## 5. 扩展指南

- **换稠密编码器**：实现新的 `BaseIndex`（如接 BGE 句向量），替换 `pipeline` 中的 `DenseIndex` 即可，接口不变。
- **换重排器**：继承 `BaseReranker`，实现 `fit/rerank`。
- **加数据源**：继承 `BaseLoader`（如 PDF/HTML），实现 `load`。
- **换生成器**：继承 `BaseGenerator`，实现 `generate`。
