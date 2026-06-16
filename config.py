"""RAG 子系统 — 配置"""
import os
from pathlib import Path

# 项目根
RAG_ROOT = Path("/data/rag")

# ChromaDB 持久化路径
CHROMA_DATA_PATH = "/data/rag/chroma"

# 向量维度（豆包 doubao-embedding-large-text-250515 的维度）
EMBEDDING_DIM = 2048

# Embedding 批处理参数
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_MAX_RETRIES = 3

# Embedding Prefix（BGE 惯例，对豆包同样有效）
# 设为 None 不使用 prefix；设字符串则自动加在文本前
EMBEDDING_CONTENT_PREFIX = ""    # 文档端 prefix
EMBEDDING_QUERY_PREFIX = ""      # 查询端 prefix

# 切片参数
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CHUNK_MIN_SIZE_TARGET = 500

# 检索参数
TOP_K = 15                     # 全局 top_k 上限
QUERY_EXPANSION_COUNT = 3      # Query 多路改写数

# 模型配置
EMBEDDING_MODEL = "ep-xxxxxxxx"      # 实际值从环境变量 RAG_EMBEDDING_ENDPOINT 读取
QUERY_REWRITE_MODEL = "doubao-seed-2-0-pro-260215"


def get_embedding_endpoint() -> str:
    """Embedding endpoint ID，优先环境变量，fallback 到配置常量"""
    return os.environ.get("RAG_EMBEDDING_ENDPOINT", EMBEDDING_MODEL)

# 豆包 API 配置
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

def get_ark_api_key() -> str:
    """从环境变量 / .env 读取豆包 API Key"""
    key = os.environ.get("DOUBAO_API_KEY", "")
    if key:
        return key
    env_path = Path(os.path.expanduser("~/.hermes/.env"))
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DOUBAO_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError("DOUBAO_API_KEY not found in env or ~/.hermes/.env")
