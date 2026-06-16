"""Embedder — 向量化模块

职责：把文本转成向量
隔离边界：输入 text -> 输出 float 数组

支持两种 API 模式：
- 文本 embedding：POST /api/v3/embeddings（OpenAI 兼容，批量）
- 多模态 embedding：POST /api/v3/embeddings/multimodal（每次一文，并发）
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from config import (
    ARK_BASE_URL, EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_RETRIES,
    EMBEDDING_CONTENT_PREFIX, EMBEDDING_QUERY_PREFIX,
    get_ark_api_key,
)
from models import Document, Chunk

logger = logging.getLogger(__name__)

# 多模态并发数
_MULTIMODAL_CONCURRENCY = 5


class Embedder:
    """文本向量化，支持 content/query 双 prefix + 分批写入"""

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.api_key = get_ark_api_key()
        self.content_prefix = EMBEDDING_CONTENT_PREFIX
        self.query_prefix = EMBEDDING_QUERY_PREFIX
        # vision 模型走多模态端点
        self._is_vision = "vision" in model.lower() or model.startswith("ep-")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量转向量，自动分批，自动加 content_prefix"""
        if self.content_prefix:
            texts = [f"{self.content_prefix}{t}" for t in texts]
        return self._batch_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """单句查询转向量，自动加 query_prefix"""
        if self.query_prefix:
            text = f"{self.query_prefix}{text}"
        return self._batch_embed([text])[0]

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """从 Chunk 列表取 text 批量转向量"""
        texts = [c.text for c in chunks]
        return self.embed(texts)

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """内核：分批调 API，含重试"""
        results: List[Optional[List[float]]] = [None] * len(texts)

        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            batch_indices = list(range(start, min(start + EMBEDDING_BATCH_SIZE, len(texts))))

            for attempt in range(EMBEDDING_MAX_RETRIES):
                try:
                    if self._is_vision:
                        vectors = self._call_multimodal(batch)
                    else:
                        vectors = self._call_text(batch)
                    for idx, vec in zip(batch_indices, vectors):
                        results[idx] = vec
                    break
                except Exception as e:
                    logger.warning(
                        "embed batch [%d:%d] attempt %d failed: %s",
                        start, start + len(batch), attempt + 1, e
                    )
                    if attempt == EMBEDDING_MAX_RETRIES - 1:
                        raise
                    time.sleep(1)

        assert all(r is not None for r in results), "Some embeddings were not generated"
        return results  # type: ignore

    def _call_text(self, texts: List[str]) -> List[List[float]]:
        """标准 OpenAI 兼容文本 embedding"""
        import urllib.request, json, ssl
        ctx = ssl.create_default_context()
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{ARK_BASE_URL}/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        result = json.loads(resp.read())
        return [d["embedding"] for d in result["data"]]

    def _call_multimodal(self, texts: List[str]) -> List[List[float]]:
        """多模态 embedding：每次一文，并发执行"""
        with ThreadPoolExecutor(max_workers=_MULTIMODAL_CONCURRENCY) as pool:
            fut_to_idx = {
                pool.submit(self._mm_embed_one, t): i
                for i, t in enumerate(texts)
            }
            results: List[Optional[List[float]]] = [None] * len(texts)
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results[idx] = fut.result()
        return results  # type: ignore

    def _mm_embed_one(self, text: str) -> List[float]:
        """单文多模态 embedding"""
        import urllib.request, json, ssl
        ctx = ssl.create_default_context()
        body = json.dumps({
            "model": self.model,
            "input": [{"type": "text", "text": text}],
        }).encode()
        req = urllib.request.Request(
            f"{ARK_BASE_URL}/embeddings/multimodal",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        result = json.loads(resp.read())
        # 单输入返回 data 为 dict: {"embedding": [...]}
        return result["data"]["embedding"]
