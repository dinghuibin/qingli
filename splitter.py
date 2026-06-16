"""Splitter — 文本切片模块

职责：长文本切成可检索的语义块
隔离边界：输入 Document -> 输出 Chunk 列表
"""
import logging
from typing import List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from models import Document, Chunk
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_SIZE_TARGET

logger = logging.getLogger(__name__)


class Splitter:
    """双层切片策略"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_size_target: int = CHUNK_MIN_SIZE_TARGET,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size_target = min_size_target

        # 第一层：按 Markdown 标题切
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
        )
        # 第二层：递归字符切分
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split(self, document: Document) -> List[Chunk]:
        """双层切分 + 碎片合并"""
        md_text = document.text

        # 第一层：按标题切
        header_sections = self.header_splitter.split_text(md_text)

        # 第二层：递归字符切分
        all_chunks: List[Chunk] = []
        for section in header_sections:
            sub_chunks = self.char_splitter.split_text(section.page_content)
            for sub in sub_chunks:
                metadata = dict(section.metadata)
                metadata.update(document.metadata)
                all_chunks.append(Chunk(text=sub, metadata=metadata))

        # 碎片合并
        merged = self._merge_small_chunks(all_chunks)

        logger.info(
            "Split %d chars into %d chunks (merged from %d)",
            len(document.text), len(merged), len(all_chunks),
        )
        return merged

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """前向合并小碎片（参考 Open WebUI 的 ChunkMinSizeTarget 算法）"""
        if not chunks:
            return []

        merged: List[Chunk] = []
        acc = Chunk(text=chunks[0].text, metadata=chunks[0].metadata.copy())

        for chunk in chunks[1:]:
            # 如果累计长度 < 目标值 且 合并后不超过 chunk_size
            if (
                len(acc.text) < self.min_size_target
                and len(acc.text) + len(chunk.text) + 2 <= self.chunk_size
            ):
                acc.text += "\n\n" + chunk.text
            else:
                merged.append(acc)
                acc = Chunk(text=chunk.text, metadata=chunk.metadata.copy())

        if acc.text:
            merged.append(acc)

        return merged
