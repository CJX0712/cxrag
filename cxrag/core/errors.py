"""统一错误码与异常类型。

错误码用于跨模块契约，调用方依据 code 做分支处理。
"""


class CXError(Exception):
    """CxRAG 统一异常。"""

    def __init__(self, code: str, message: str, *, cause: BaseException | None = None):
        self.code = code
        self.message = message
        self.cause = cause
        super().__init__(f"[{code}] {message}")

    def __str__(self) -> str:  # pragma: no cover - 仅用于可读
        return f"[{self.code}] {self.message}"


# 错误码表（code -> 语义）
CODES: dict[str, str] = {
    "E_LOAD": "文档加载失败",
    "E_CHUNK": "文本切分失败",
    "E_INDEX_EMPTY": "索引为空，请先 build",
    "E_INDEX_BUILD": "索引构建失败",
    "E_RETRIEVE": "检索失败",
    "E_GEN": "答案生成失败",
    "E_CONFIG": "配置错误",
    "E_DEP": "可选依赖缺失",
}
