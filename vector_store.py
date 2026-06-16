"""VectorStore — ChromaDB 向量存储模块

职责：存储文档向量和元数据，提供增删检索
隔离边界：只认向量和元数据 dict，不知道「文档」是什么

Collection 命名规则：ChromaDB 要求英文/数字/._- 开头结尾字母数字
知识库中文名通过 slug 映射到合法 collection 名
"""
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.batch_utils import create_batches

from config import CHROMA_DATA_PATH

logger = logging.getLogger(__name__)

# ChromaDB 合法名：3-64 位 [a-zA-Z0-9._-]，首尾不能是 .-
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,62}[a-zA-Z0-9]$")


def slugify(name: str) -> str:
    """把知识库名转成合法 collection 名"""
    # \w 在 Python 3 中匹配 Unicode 字，显式只保留 ASCII 字母数字
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_")
    # 确保长度 >= 3
    if len(slug) < 3:
        slug = f"kb_{slug}_{hashlib.md5(name.encode()).hexdigest()[:4]}"
    # 确保首尾合法
    slug = re.sub(r"^[^a-zA-Z0-9]", "kb_", slug)
    slug = re.sub(r"[^a-zA-Z0-9]$", "_kb", slug)
    return slug[:63]


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add(self, collection: str, texts: List[str],
            embeddings: List[List[float]], metadatas: List[dict]) -> List[str]:
        ...

    @abstractmethod
    def search(self, collection: str, query_embedding: List[float],
               top_k: int, filter: Optional[dict] = None) -> List[dict]:
        ...

    @abstractmethod
    def delete(self, collection: str, ids: Optional[List[str]] = None):
        ...

    @abstractmethod
    def collection_exists(self, collection: str) -> bool:
        ...

    @abstractmethod
    def list_collections(self) -> List[str]:
        ...

    @abstractmethod
    def delete_collection(self, collection: str):
        ...

    @abstractmethod
    def create_collection(self, name: str):
        ...

    @staticmethod
    def validate_collection_name(name: str):
        if not _COLLECTION_NAME_RE.match(name):
            raise ValueError(
                f"Invalid collection name: {name!r}. "
                f"Must match pattern: {_COLLECTION_NAME_RE.pattern}"
            )


class ChromaVectorStore(VectorStore):
    """ChromaDB 实现"""

    def __init__(self, persist_dir: str = CHROMA_DATA_PATH):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    # ── 工具方法 ──

    def _get_or_create(self, name: str):
        col_name = slugify(name)
        self.validate_collection_name(col_name)
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── 接口实现 ──

    def create_collection(self, name: str):
        col_name = slugify(name)
        self.validate_collection_name(col_name)
        self.client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _normalize_distances(
            self, distances: List[float],
    ) -> List[float]:
        """ChromaDB 返回余弦距离 (0~2, 越小越近)，转为 0~1 相似度 (越大越好)"""
        return [(2.0 - d) / 2.0 for d in distances]

    # ── 接口实现 ──

    def add(self, collection: str, texts: List[str],
            embeddings: List[List[float]], metadatas: List[dict]) -> List[str]:
        """分批写入 ChromaDB"""
        col_name = slugify(collection)
        col = self._get_or_create(col_name)

        ids = [f"{col_name}_{i}" for i in range(
            col.count() if col.count() else 0,
            col.count() + len(texts),
        )]

        # ChromaDB 内置分批
        batches = create_batches(
            api=self.client,
            ids=ids,
            embeddings=embeddings,  # type: ignore
            metadatas=metadatas,  # type: ignore
            documents=texts,
        )
        for batch in batches:
            col.add(*batch)
        return ids

    def search(self, collection: str, query_embedding: List[float],
               top_k: int = 15,
               filter: Optional[dict] = None) -> List[dict]:
        """检索，返回 [{id, text, metadata, score}, ...]"""
        col_name = slugify(collection)
        col = self._get_or_create(col_name)
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter,
        )
        if not result["ids"]:
            return []

        # 整理结果
        items = []
        for i in range(len(result["ids"][0])):
            items.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": self._normalize_distances(
                    [result["distances"][0][i]]
                )[0],
            })
        return items

    def delete(self, collection: str, ids: Optional[List[str]] = None):
        col_name = slugify(collection)
        if not self.collection_exists(col_name):
            return
        col = self.client.get_collection(name=col_name)
        if ids:
            col.delete(ids=ids)

    def collection_exists(self, collection: str) -> bool:
        col_name = slugify(collection)
        try:
            self.client.get_collection(name=col_name)
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        cols = self.client.list_collections()
        return [c.name for c in cols]

    def delete_collection(self, collection: str):
        col_name = slugify(collection)
        if self.collection_exists(col_name):
            self.client.delete_collection(name=col_name)
