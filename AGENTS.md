# Agent instructions for 青藜

This file is the shared entry point for AI assistants working in this repository.

## Read first

1. `README.md`

## Key conventions

- Python 3.14+, `from __future__ import annotations`
- No external framework dependencies beyond: chromadb, langchain-text-splitters, ebooklib, pymupdf, python-docx, beautifulsoup4, openai
- All paths in config.py use `/data/rag/` as root
- Vector store is ChromaDB, persistent at `/data/rag/chroma/`
- Embedding via 豆包 multimodal API (Volcengine Ark), endpoint ID in config
- Query expansion generates 1-3 rewritten queries before search
