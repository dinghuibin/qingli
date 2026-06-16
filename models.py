"""RAG 子系统 — 数据模型"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """文档单元：Loader 输出 → Splitter 输入/输出"""
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """切片单元：Splitter 输出 → Embedder 输入"""
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """来源 doc 的标识"""
        return self.metadata.get("doc_id", "")


@dataclass
class KnowledgeBase:
    """知识库元数据"""
    name: str            # collection 名
    description: str = ""
    doc_count: int = 0
    chunk_count: int = 0
