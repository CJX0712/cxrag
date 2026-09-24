# 部署指南（DEPLOYMENT）

作者：晨星

## 1. 环境要求

- Python ≥ 3.10（本机验证 3.13）
- 无需 GPU、无需联网下载模型权重
- 操作系统：Windows / Linux / macOS

## 2. 一键部署（推荐）

克隆后执行对应脚本，**零手工干预**：

```bash
# Linux / macOS
bash setup.sh

# Windows
setup.bat
```

脚本动作：`建 venv → 装 requirements.lock → 跑 python -m cxrag.cli demo`。

## 3. 手动部署

```bash
cd cxrag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.lock  # 版本锁定，保证可复现
python -m cxrag.cli demo
```

> 可选：接入真实生成式 LLM → `pip install -r requirements-llm.txt`

## 4. 依赖与可复现

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 核心依赖（可读清单，未锁版本） |
| `requirements.lock` | `pip freeze` 精确版本，**干净环境一键复现的依据** |
| `requirements-llm.txt` | 可选 LLM 依赖（OpenAI / llama-cpp） |
| `pyproject.toml` | 包元数据 + 入口 `cxrag` 命令 |

本机锁定版本（节选）：`numpy 2.5.3 / scikit-learn 1.9.1 / scipy 1.18.1 / rank_bm25 0.2.2 / faiss-cpu 1.15.1 / networkx 3.7`。

## 5. 验证（Definition of Done 自检）

```bash
# 1) 单测全过
python -m pytest tests/ -q
# 2) demo 跑通（克隆 → 一键 → 端到端）
python -m cxrag.cli demo
# 3) 复现性：删除 .venv 重新执行 setup 仍可跑通
```

预期：23 个测试通过；demo 输出 `ir_metrics.recall@10 = 1.0`、`mrr = 1.0`。

## 6. 可选：Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN python -m venv .venv && .venv/bin/pip install -r requirements.lock
CMD ["/app/.venv/bin/python", "-m", "cxrag.cli", "demo"]
```

## 7. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `E_LOAD 文件不存在` | 路径错 | 用绝对路径或先 `index` 建库 |
| faiss 不可用 | 环境无 wheel | 自动回退 numpy 精确检索，功能不变 |
| 答案为空 | 上下文为空 | 检查 `index` 是否成功、`ask` 的 k>0 |
| 重排器未训练 | gold 不足/无标注 | 属预期，退化为恒等重排，仍可跑 |
