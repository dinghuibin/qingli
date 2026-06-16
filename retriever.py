"""Retriever — 检索编排器

职责：组合 Loader → Splitter → Embedder → VectorStore
      提供入库、检索、知识库管理一站式 API
      内含 Query 多路改写 + RAG Template

隔离边界：仅编排，不关心各模块的实现细节
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import (
    ARK_BASE_URL, QUERY_REWRITE_MODEL, TOP_K, QUERY_EXPANSION_COUNT,
    get_ark_api_key,
)
from embedder import Embedder
from models import Chunk, Document, KnowledgeBase
from splitter import Splitter
from vector_store import ChromaVectorStore, VectorStore, slugify

logger = logging.getLogger(__name__)

# 显示名 ↔ slug 映射文件
_KB_MAP_PATH = Path(os.path.expanduser("~/.hermes/scripts/rag/kb_map.json"))


def _load_kb_map() -> dict:
    if _KB_MAP_PATH.exists():
        with open(_KB_MAP_PATH) as f:
            return json.load(f)
    return {}


def _save_kb_map(mapping: dict):
    _KB_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_KB_MAP_PATH, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


class Retriever:
    """检索编排器"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        splitter: Optional[Splitter] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or Embedder()
        self.splitter = splitter or Splitter()

    # ── 知识库管理 ──

    def create_knowledge_base(self, name: str):
        """创建知识库（实际就是创建 ChromaDB collection）"""
        self.vector_store.create_collection(name)
        # 保存显示名 ↔ slug 映射
        mapping = _load_kb_map()
        mapping[name] = slugify(name)
        _save_kb_map(mapping)

    def delete_knowledge_base(self, name: str):
        self.vector_store.delete_collection(name)
        mapping = _load_kb_map()
        mapping.pop(name, None)
        _save_kb_map(mapping)

    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        mapping = _load_kb_map()
        rev = {v: k for k, v in mapping.items()}
        slugs = self.vector_store.list_collections()
        result = []
        for slug in slugs:
            display = rev.get(slug, slug)
            result.append(KnowledgeBase(name=display))
        return result

    # ── 入库 ──

    def ingest(self, file_path: str, kb_name: str) -> int:
        """入库：Loader → Splitter → Embedder → VectorStore"""
        from loaders import Loader

        if not self.vector_store.collection_exists(kb_name):
            raise ValueError(f"知识库 '{kb_name}' 不存在，请先创建")

        # 1. Load
        loader = Loader.detect(file_path)
        docs = loader.load(file_path)
        if not docs:
            logger.warning("No content extracted from %s", file_path)
            return 0

        # 2. Split
        all_chunks: List[Chunk] = []
        for doc in docs:
            chunks = self.splitter.split(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # 3. Embed
        embeddings = self.embedder.embed_chunks(all_chunks)

        # 4. Store
        texts = [c.text for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        self.vector_store.add(kb_name, texts, embeddings, metadatas)

        return len(all_chunks)

    # ── 检索 ──

    def query(self, question: str, kb_name: str,
              top_k: int = TOP_K) -> List[dict]:
        """检索：Query 多路改写 → 向量检索 → 合并去重 → 返回"""
        # 1. Query 改写
        queries = self._expand_queries(question)

        # 2. 多路向量检索
        seen_ids = set()
        results: List[dict] = []
        per_query_k = max(top_k // len(queries), 3)

        for q in queries:
            q_emb = self.embedder.embed_query(q)
            hits = self.vector_store.search(kb_name, q_emb, top_k=per_query_k)
            for h in hits:
                if h["id"] not in seen_ids:
                    seen_ids.add(h["id"])
                    results.append(h)

        # 3. 按 score 降序排列，取全局 top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── RAG Template ──

    def format_context(self, results: List[dict], kb_name: str) -> str:
        """把检索结果拼成 RAG Template"""
        if not results:
            return ""

        parts = [f"以下是从「{kb_name}」中检索到的相关内容：\n"]
        for i, r in enumerate(results, 1):
            source_info = r["metadata"].get("title") or r["metadata"].get("source", "未知")
            chap = r["metadata"].get("chapter", "")
            if chap:
                source_info = f"{source_info} · {chap}"
            page = r["metadata"].get("page")
            if page:
                source_info += f" · 第 {page} 页"

            parts.append(
                f"--- 来源：{source_info} (相似度: {r['score']:.2f}) ---\n"
                f"{r['text']}\n"
            )

        parts.append(
            "\n请基于以上检索内容回答问题。"
            "如果检索内容不足以回答，请如实说明。"
        )
        return "\n".join(parts)

    # ── Query 多路改写 ──

    def _expand_queries(self, question: str) -> List[str]:
        """LLM 把用户问题改写成多个搜索 query"""
        if QUERY_EXPANSION_COUNT <= 1:
            return [question]

        api_key = get_ark_api_key()
        client = OpenAI(api_key=api_key, base_url=ARK_BASE_URL)

        prompt = (
            f"你是搜索 query 改写助手。请把以下用户问题改写成 {QUERY_EXPANSION_COUNT} 个不同的搜索 query，"
            "覆盖不同的关键词和句式，有助于提高召回率。\n"
            "只输出 query，每行一个，不要序号。\n\n"
            f"用户问题：{question}"
        )

        try:
            resp = client.chat.completions.create(
                model=QUERY_REWRITE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            content = resp.choices[0].message.content or ""
            queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
            # 过滤掉空行和可能的编号
            clean = []
            for q in queries:
                # 去掉行首 "1." "2." 等编号
                import re as _re
                q = _re.sub(r"^\d+[.、\)]\s*", "", q).strip()
                if q and len(q) > 3:
                    clean.append(q)
            if clean:
                # 原问题也在输出中
                if question not in clean:
                    clean.insert(0, question)
                logger.info("Query expansion: %s -> %s", question, clean)
                return clean[:QUERY_EXPANSION_COUNT]
        except Exception as e:
            logger.warning("Query expansion failed, fallback to original: %s", e)

        return [question]
