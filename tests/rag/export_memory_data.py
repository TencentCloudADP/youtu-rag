"""
导出 memory_data 目录下的记忆内容

功能：
1. 读取 ChromaDB 中存储的记忆数据
2. 支持按 collection 过滤
3. 显示记忆类型、用户ID、内容等信息
4. 支持输出到控制台或文件

使用方法：
    # 列出所有 collections
    python tests/rag/export_memory_data.py --list

    # 导出所有记忆到控制台
    python tests/rag/export_memory_data.py

    # 导出指定 collection 的记忆
    python tests/rag/export_memory_data.py --collection memory_user123

    # 导出到文件
    python tests/rag/export_memory_data.py --output memory_export.txt

    # 导出为 JSON 格式
    python tests/rag/export_memory_data.py --output memory_export.json --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb


def format_metadata(metadata: dict) -> str:
    """格式化元数据为易读的字符串"""
    lines = []

    # 按重要性排序显示字段
    priority_fields = [
        'memory_type', 'user_id', 'session_id', 'document_id',
        'importance_score', 'success_rate', 'created_at', 'updated_at'
    ]

    for field in priority_fields:
        if field in metadata:
            value = metadata[field]
            lines.append(f"  • {field}: {value}")

    # 显示其他字段
    for key, value in sorted(metadata.items()):
        if key not in priority_fields:
            # 处理 JSON 字符串
            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                try:
                    parsed = json.loads(value)
                    value = json.dumps(parsed, ensure_ascii=False, indent=4)
                    lines.append(f"  • {key}:")
                    for line in value.split('\n'):
                        lines.append(f"    {line}")
                except:
                    lines.append(f"  • {key}: {value}")
            else:
                lines.append(f"  • {key}: {value}")

    return '\n'.join(lines)


def export_to_text(collections_data: list[dict], output_file: Optional[Path] = None):
    """导出为文本格式（更易读）"""
    lines = []
    lines.append("=" * 100)
    lines.append(f"记忆数据导出")
    lines.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)

    total_memories = sum(len(data['documents']) for data in collections_data)
    lines.append(f"\n总记忆数: {total_memories}")
    lines.append(f"Collections 数量: {len(collections_data)}")

    for coll_data in collections_data:
        collection_name = coll_data['collection_name']
        count = len(coll_data['documents'])

        lines.append(f"\n\n{'#' * 100}")
        lines.append(f"Collection: {collection_name}")
        lines.append(f"记忆数量: {count}")
        lines.append(f"{'#' * 100}")

        for i, (doc_id, document, metadata) in enumerate(
            zip(coll_data['ids'], coll_data['documents'], coll_data['metadatas']), 1
        ):
            lines.append("")
            lines.append(f"{'=' * 100}")
            lines.append(f"记忆 #{i} (ID: {doc_id})")
            lines.append(f"{'=' * 100}")

            lines.append(f"\n元数据:")
            lines.append("-" * 100)
            if metadata:
                lines.append(format_metadata(metadata))
            else:
                lines.append("  (无元数据)")

            lines.append(f"\n记忆内容:")
            lines.append("-" * 100)
            lines.append(document if document else "(空内容)")
            lines.append("")

    lines.append("\n" + "=" * 100)
    lines.append("导出完成")
    lines.append("=" * 100)

    output_text = "\n".join(lines)

    if output_file:
        output_file.write_text(output_text, encoding='utf-8')
        print(f"✓ 数据已导出到: {output_file.absolute()}")
        print(f"  • 总记忆数: {total_memories}")
        print(f"  • 文件大小: {output_file.stat().st_size / 1024:.2f} KB")
    else:
        print(output_text)


def export_to_json(collections_data: list[dict], output_file: Optional[Path] = None):
    """导出为 JSON 格式"""
    export_data = {
        'export_time': datetime.now().isoformat(),
        'total_memories': sum(len(data['documents']) for data in collections_data),
        'collections': collections_data
    }

    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

    if output_file:
        output_file.write_text(json_str, encoding='utf-8')
        print(f"✓ 数据已导出到: {output_file.absolute()}")
        print(f"  • 总记忆数: {export_data['total_memories']}")
        print(f"  • 文件大小: {output_file.stat().st_size / 1024:.2f} KB")
    else:
        print(json_str)


def export_memory_data(
    collection_name: Optional[str] = None,
    output_file: Optional[str] = None,
    format: str = 'text',
    chroma_path: str = './rag_data/memory_data',
    list_only: bool = False
):
    """导出记忆数据"""

    # 连接到 ChromaDB
    persist_directory = Path(chroma_path)
    if not persist_directory.exists():
        print(f"✗ 错误: 记忆存储目录不存在: {persist_directory}")
        return

    print(f"正在连接到 ChromaDB...")
    print(f"  • 存储路径: {persist_directory.absolute()}")

    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
    except Exception as e:
        print(f"✗ 错误: 无法连接到 ChromaDB: {str(e)}")
        return

    # 列出所有 collections
    all_collections = client.list_collections()
    print(f"\n可用的 Collections ({len(all_collections)}):")
    for coll in all_collections:
        count = coll.count()
        print(f"  • {coll.name} ({count} 条记忆)")

    if list_only:
        return

    # 确定要导出的 collections
    if collection_name:
        try:
            collections_to_export = [client.get_collection(name=collection_name)]
        except Exception as e:
            print(f"✗ 错误: 无法获取 collection '{collection_name}': {str(e)}")
            return
    else:
        collections_to_export = all_collections

    if not collections_to_export:
        print("✗ 没有可导出的 collections")
        return

    # 导出数据
    print(f"\n正在导出 {len(collections_to_export)} 个 collection(s)...")
    collections_data = []

    for collection in collections_to_export:
        total_count = collection.count()
        print(f"\n正在读取 collection: {collection.name}")
        print(f"  • 记忆数量: {total_count:,}")

        if total_count == 0:
            print("  ⚠ Collection 为空，跳过")
            continue

        # 分批获取所有数据
        batch_size = 1000
        coll_data = {
            'collection_name': collection.name,
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

            coll_data['ids'].extend(result['ids'])
            coll_data['documents'].extend(result['documents'])
            coll_data['metadatas'].extend(result['metadatas'])

        collections_data.append(coll_data)
        print(f"  ✓ 完成")

    if not collections_data:
        print("✗ 没有数据可导出")
        return

    # 导出数据
    output_path = Path(output_file) if output_file else None

    print(f"\n正在导出数据...")
    if format == 'json':
        export_to_json(collections_data, output_path)
    else:  # text
        export_to_text(collections_data, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='导出 memory_data 目录下的记忆内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有 collections
  %(prog)s --list

  # 导出所有记忆到控制台（文本格式）
  %(prog)s

  # 导出指定 collection
  %(prog)s --collection memory_user123

  # 导出到文本文件
  %(prog)s --output memory_export.txt

  # 导出到 JSON 文件
  %(prog)s --output memory_export.json --format json
        """
    )

    parser.add_argument(
        '--collection',
        type=str,
        help='要导出的 collection 名称（不指定则导出所有）'
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
        default='./rag_data/memory_data',
        help='ChromaDB 存储路径（默认: ./rag_data/memory_data）'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='仅列出所有 collections，不导出数据'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("记忆数据导出工具")
    print("=" * 100)

    export_memory_data(
        collection_name=args.collection,
        output_file=args.output,
        format=args.format,
        chroma_path=args.path,
        list_only=args.list
    )

    print("\n完成！")


if __name__ == "__main__":
    main()

