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

### 安装依赖

```bash
cd /data/rag
python3 -m venv .venv
.venv/bin/pip install chromadb langchain-text-splitters ebooklib beautifulsoup4 pymupdf python-docx openai
```

### 设置 CLI 命令

```bash
chmod +x /data/rag/rag_api.py
ln -sf /data/rag/rag_api.py ~/.local/bin/rag
```

### 使用

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

## 准备工作：获取 API Key 和 Endpoint

青藜使用火山引擎 Ark 的豆包多模态 embedding 模型进行向量化。需要先开通服务：

### 1. 开通模型

1. 打开 [火山引擎 Ark 控制台 → 模型列表](https://console.volcengine.com/ark/region:ark+cn-beijing/model)
2. 搜索 `doubao-embedding-vision`，点击「开通模型服务」

### 2. 创建推理接入点（Endpoint）

1. 进入 [推理接入点](https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint)
2. 点击「创建推理接入点」，选择已开通的 `doubao-embedding-vision` 模型
3. 创建完成后，复制 **Endpoint ID**（格式如 `ep-20260616150213-xxxxx`）

### 3. 获取 API Key

1. 进入 [API Key 管理](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)
2. 复制你的 API Key（格式如 `ea764f0f-****-****-************`）

### 4. 配置环境变量

将以下内容写入 `~/.hermes/.env`（或 export 到环境变量）：

```bash
# 豆包 API Key
DOUBAO_API_KEY=你的API_KEY

# Embedding Endpoint ID（刚才创建的）
RAG_EMBEDDING_ENDPOINT=你的Endpoint_ID
```

## 快速开始

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
