"""
导出 ChromaDB collection 的完整数据

功能：
1. 完整导出指定 collection 的所有文档和元数据
2. 支持输出到控制台或文件
3. 支持 JSON 和文本格式

使用方法：
    # 输出到控制台
    python tests/rag/export_collection_data.py --collection kb_1_20251210_154655

    # 导出到 JSON 文件
    python tests/rag/export_collection_data.py --collection kb_a48da1b24be2_20260111_152544 --output data.json

    # 导出到文本文件（更易读）
    python tests/rag/export_collection_data.py --collection kb_1_20251210_154655 --output data.txt --format text
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb


def export_to_json(data: dict, output_file: Optional[Path] = None):
    """导出为 JSON 格式"""
    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    if output_file:
        output_file.write_text(json_str, encoding='utf-8')
        print(f"✓ 数据已导出到: {output_file.absolute()}")
        print(f"  • 总文档数: {len(data['documents'])}")
        print(f"  • 文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(json_str)


def export_to_text(data: dict, output_file: Optional[Path] = None):
    """导出为文本格式（更易读）"""
    lines = []
    lines.append("=" * 100)
    lines.append(f"Collection 数据导出")
    lines.append(f"总文档数: {len(data['documents'])}")
    lines.append("=" * 100)

    for i, (doc_id, document, metadata) in enumerate(zip(data['ids'], data['documents'], data['metadatas']), 1):
        lines.append("")
        lines.append(f"{'=' * 100}")
        lines.append(f"文档 #{i}")
        lines.append(f"{'=' * 100}")
        lines.append(f"\nID: {doc_id}")

        lines.append(f"\n元数据:")
        lines.append("-" * 100)
        if metadata:
            for key, value in sorted(metadata.items()):
                lines.append(f"  • {key}: {value}")
        else:
            lines.append("  (无元数据)")

        lines.append(f"\n文档内容:")
        lines.append("-" * 100)
        lines.append(document if document else "(空文档)")
        lines.append("")

    lines.append("\n" + "=" * 100)
    lines.append("导出完成")
    lines.append("=" * 100)

    output_text = "\n".join(lines)

    if output_file:
        output_file.write_text(output_text, encoding='utf-8')
        print(f"✓ 数据已导出到: {output_file.absolute()}")
        print(f"  • 总文档数: {len(data['documents'])}")
        print(f"  • 文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(output_text)


def export_collection_data(
    collection_name: str,
    output_file: Optional[str] = None,
    format: str = 'text',
    chroma_path: str = './rag_data/vector_store'
):
    """导出 collection 的完整数据"""

    # 连接到 ChromaDB
    persist_directory = Path(chroma_path)
    if not persist_directory.exists():
        print(f"✗ 错误: 向量存储目录不存在: {persist_directory}")
        return

    print(f"正在连接到 ChromaDB...")
    print(f"  • 存储路径: {persist_directory.absolute()}")

    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
    except Exception as e:
        print(f"✗ 错误: 无法连接到 ChromaDB: {str(e)}")
        return

    # 获取 collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"✗ 错误: 无法获取 collection '{collection_name}': {str(e)}")
        print(f"\n可用的 collections:")
        for coll in client.list_collections():
            print(f"  • {coll.name}")
        return

    # 获取文档总数
    total_count = collection.count()
    print(f"\n正在读取 collection: {collection_name}")
    print(f"  • 总文档数: {total_count:,}")

    if total_count == 0:
        print("✗ Collection 为空，无数据可导出")
        return

    # 分批获取所有数据
    print(f"\n正在获取数据...")
    batch_size = 1000
    all_data = {
        'collection_name': collection_name,
        'total_count': total_count,
        'ids': [],
        'documents': [],
        'metadatas': [],
    }

    for offset in range(0, total_count, batch_size):
        limit = min(batch_size, total_count - offset)
        print(f"  • 进度: {offset + limit}/{total_count} ({(offset + limit) / total_count * 100:.1f}%)")

        result = collection.get(
            limit=limit,
            offset=offset,
            include=['documents', 'metadatas']
        )

        all_data['ids'].extend(result['ids'])
        all_data['documents'].extend(result['documents'])
        all_data['metadatas'].extend(result['metadatas'])

    print(f"✓ 数据获取完成，共 {len(all_data['documents'])} 条")

    # 导出数据
    output_path = Path(output_file) if output_file else None

    print(f"\n正在导出数据...")
    if format == 'json':
        export_to_json(all_data, output_path)
    else:  # text
        export_to_text(all_data, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='导出 ChromaDB collection 的完整数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 输出到控制台（文本格式）
  %(prog)s --collection kb_1_20251210_154655

  # 导出到文本文件
  %(prog)s --collection kb_1_20251210_154655 --output export.txt

  # 导出到 JSON 文件
  %(prog)s --collection kb_1_20251210_154655 --output export.json --format json
        """
    )

    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        help='要导出的 collection 名称'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径（不指定则输出到控制台）'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='导出格式：text（易读文本）或 json（默认: text）'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='./rag_data/vector_store',
        help='ChromaDB 存储路径（默认: ./rag_data/vector_store）'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("ChromaDB Collection 数据导出工具")
    print("=" * 100)

    export_collection_data(
        collection_name=args.collection,
        output_file=args.output,
        format=args.format,
        chroma_path=args.path
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
