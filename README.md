# 青藜 (Qīnglí) — 个人知识库 RAG 引擎

青藜，典出汉代学者燃藜杖夜读之典故，取"勤学深思"之意。

Hermes Agent 的一个 Skill 子系统，将个人藏书（epub/PDF/Word）入库为向量知识库，对话时自动检索相关段落以增强回答。

## 架构

```
青藜                                       (Hermes Skill: rag)
├── Embedder — 豆包多模态 API 向量化         (embedder.py)
├── VectorStore — ChromaDB 持久化存储        (vector_store.py)
├── Loader — epub/PDF/Word 文档读取          (loaders/)
├── Splitter — Markdown 双层切片 + 碎片合并   (splitter.py)
├── Retriever — 编排器 + Query 改写 + RAG 模板 (retriever.py)
└── rag_api.py — CLI 入口                   (→ cli: rag)
```

## 快速开始

```bash
# 创建知识库
rag create-kb "毛选"

# 入库一本 epub
rag ingest /data/books/实践论.epub -k "毛选"

# 检索（含 Query 多路改写 + RAG 模板）
rag format "毛选里怎么评价孙中山" -k "毛选"

# 列出知识库
rag list-kb
```

## 依赖

- Python 3.14+
- ChromaDB 1.5 (向量数据库，本地持久化)
- 豆包多模态 embedding API（火山引擎 Ark）

## 项目结构

```
/data/rag/
├── config.py          # 配置
├── models.py          # 数据模型
├── embedder.py        # 向量化
├── vector_store.py    # ChromaDB 封装
├── splitter.py        # 切片
├── retriever.py       # 编排器
├── loaders/           # 文档加载器
│   └── __init__.py
├── rag_api.py          # CLI 入口
├── chroma/            # ChromaDB 数据（Git 忽略）
├── kb_map.json        # 知识库名字映射（Git 忽略）
└── .venv/             # Python 虚拟环境（Git 忽略）
```
