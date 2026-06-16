#! /home/ubuntu/.hermes/scripts/rag/.venv/bin/python3
"""RAG CLI 入口

用法：
  rag_api.py ingest <file_path> -k <kb_name>
  rag_api.py query <question> -k <kb_name> [-n <top_k>]
  rag_api.py create-kb <kb_name>
  rag_api.py delete-kb <kb_name>
  rag_api.py list-kb
  rag_api.py format <question> -k <kb_name>    # 只输出 RAG Template 不开 LLM
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加 RAG 模块路径
_RAG_DIR = "/data/rag"
sys.path.insert(0, _RAG_DIR)

from retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_api")


def main():
    parser = argparse.ArgumentParser(description="Hermes RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = sub.add_parser("ingest", help="入库文档")
    ingest_p.add_argument("file_path", help="文件路径")
    ingest_p.add_argument("-k", "--kb", required=True, help="知识库名")

    # query
    query_p = sub.add_parser("query", help="检索问答")
    query_p.add_argument("question", help="用户问题")
    query_p.add_argument("-k", "--kb", required=True, help="知识库名")
    query_p.add_argument("-n", "--top-k", type=int, default=15,
                         help="返回 top_k 条 (default: 15)")

    # format
    fmt_p = sub.add_parser("format", help="只输出 RAG Template 不开 LLM")
    fmt_p.add_argument("question", help="用户问题")
    fmt_p.add_argument("-k", "--kb", required=True, help="知识库名")
    fmt_p.add_argument("-n", "--top-k", type=int, default=15)

    # create-kb
    sub.add_parser("create-kb", help="创建知识库").add_argument("name")
    sub.add_parser("delete-kb", help="删除知识库").add_argument("name")
    sub.add_parser("list-kb", help="列出所有知识库")

    args = parser.parse_args()
    retriever = Retriever()

    try:
        if args.command == "create-kb":
            retriever.create_knowledge_base(args.name)
            print(f"✅ 知识库 '{args.name}' 创建成功")

        elif args.command == "delete-kb":
            retriever.delete_knowledge_base(args.name)
            print(f"✅ 知识库 '{args.name}' 已删除")

        elif args.command == "list-kb":
            kbs = retriever.list_knowledge_bases()
            if kbs:
                print("知识库列表：")
                for kb in kbs:
                    print(f"  📚 {kb.name}")
            else:
                print("暂无知识库")

        elif args.command == "ingest":
            count = retriever.ingest(args.file_path, args.kb)
            print(f"✅ 已入库 {count} 个片段到「{args.kb}」")

        elif args.command == "query":
            results = retriever.query(args.question, args.kb, top_k=args.top_k)
            context = retriever.format_context(results, args.kb)
            print(context)
            print("\n---\n共检索到 {} 个相关片段".format(len(results)))

        elif args.command == "format":
            results = retriever.query(args.question, args.kb, top_k=args.top_k)
            context = retriever.format_context(results, args.kb)
            print(context)

    except Exception as e:
        logger.error("操作失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
