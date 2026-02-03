"""
检查 ChromaDB 向量存储的详细信息

功能：
1. 列出所有 collection
2. 显示每个 collection 的统计信息
3. 查看文档的元数据详情
4. 检查元数据字段的分布情况

使用方法：
    uv run python tests/rag/inspect_vector_store.py
    uv run python tests/rag/inspect_vector_store.py --collection kb_1_20251207_145519
    uv run python tests/rag/inspect_vector_store.py --show-samples 5
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb


def format_size(num_bytes: int) -> str:
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def analyze_metadata_fields(metadatas: list[dict]) -> dict:
    """分析元数据字段分布"""
    field_counter = Counter()
    field_values = defaultdict(set)

    for metadata in metadatas:
        if metadata:
            for key, value in metadata.items():
                field_counter[key] += 1
                # 收集唯一值（限制每个字段最多收集10个示例）
                if len(field_values[key]) < 10:
                    field_values[key].add(str(value)[:100] if value else 'None')

    return {
        'field_counts': dict(field_counter),
        'field_samples': {k: list(v) for k, v in field_values.items()}
    }


def inspect_collection(client: chromadb.Client, collection_name: str, show_samples: int = 0):
    """检查单个 collection 的详细信息"""
    print(f"\n{'=' * 80}")
    print(f"Collection: {collection_name}")
    print(f"{'=' * 80}")

    try:
        collection = client.get_collection(name=collection_name)

        # 获取 collection 基本信息
        count = collection.count()
        print(f"\n📊 基本统计:")
        print(f"  • 总文档数: {count:,}")

        if count == 0:
            print("  ⚠️  Collection 为空")
            return

        # 获取所有数据（分页处理大数据集）
        batch_size = 1000
        all_data = {
            'ids': [],
            'metadatas': [],
            'documents': [],
        }

        for offset in range(0, count, batch_size):
            limit = min(batch_size, count - offset)
            result = collection.get(
                limit=limit,
                offset=offset,
                include=['metadatas', 'documents']
            )
            all_data['ids'].extend(result['ids'])
            all_data['metadatas'].extend(result['metadatas'])
            all_data['documents'].extend(result['documents'])

        # 分析元数据
        print(f"\n📋 元数据分析:")
        metadata_analysis = analyze_metadata_fields(all_data['metadatas'])

        print(f"\n  字段统计:")
        for field, count in sorted(metadata_analysis['field_counts'].items(), key=lambda x: -x[1]):
            coverage = (count / len(all_data['metadatas'])) * 100
            print(f"    • {field}: {count:,} 个文档 ({coverage:.1f}% 覆盖率)")

        print(f"\n  字段值示例:")
        for field, samples in metadata_analysis['field_samples'].items():
            print(f"    • {field}:")
            for sample in samples[:3]:  # 只显示前3个示例
                print(f"      - {sample}")

        # 统计文档长度
        doc_lengths = [len(doc) if doc else 0 for doc in all_data['documents']]
        if doc_lengths:
            print(f"\n📄 文档内容统计:")
            print(f"  • 平均长度: {sum(doc_lengths) / len(doc_lengths):.0f} 字符")
            print(f"  • 最短: {min(doc_lengths):,} 字符")
            print(f"  • 最长: {max(doc_lengths):,} 字符")

        # 检查特殊元数据字段
        print(f"\n🔍 特殊字段检查:")

        # 使用实际获取到的文档数量
        actual_count = len(all_data['metadatas'])

        # 检查 ETAG
        etag_count = sum(1 for m in all_data['metadatas'] if m and 'etag' in m)
        print(f"  • 包含 ETAG: {etag_count}/{actual_count} ({etag_count/actual_count*100:.1f}%)")

        # 检查元数据哈希（不应该存在）
        metadata_hash_count = sum(1 for m in all_data['metadatas'] if m and '_metadata_hash' in m)
        if metadata_hash_count > 0:
            print(f"  ⚠️  包含 _metadata_hash (应该移除): {metadata_hash_count}/{actual_count}")
        else:
            print(f"  ✓ 无 _metadata_hash 字段 (正确)")

        # 检查标准字段
        standard_fields = ['char_length', 'publish_date', 'key_timepoints', 'summary']
        for field in standard_fields:
            field_count = sum(1 for m in all_data['metadatas'] if m and field in m)
            if field_count > 0:
                print(f"  • {field}: {field_count}/{actual_count} ({field_count/actual_count*100:.1f}%)")

        # 按来源文件分组
        print(f"\n📁 来源文件统计:")
        source_counter = Counter()
        for metadata in all_data['metadatas']:
            if metadata and 'source' in metadata:
                source_counter[metadata['source']] += 1

        for source, count in source_counter.most_common(20):  # 显示前20个
            print(f"  • {source}: {count:,} chunks")

        if len(source_counter) > 20:
            print(f"  ... 以及其他 {len(source_counter) - 20} 个文件")

        # 显示样本数据
        if show_samples > 0:
            print(f"\n📝 样本数据 (前 {show_samples} 条):")
            actual_count = len(all_data['ids'])
            for i in range(min(show_samples, actual_count)):
                try:
                    print(f"\n  --- Sample {i + 1} ---")
                    print(f"  ID: {all_data['ids'][i]}")
                    print(f"  Metadata: {json.dumps(all_data['metadatas'][i], ensure_ascii=False, indent=2)}")
                    doc_preview = all_data['documents'][i][:200] if all_data['documents'][i] else 'None'
                    print(f"  Document (preview): {doc_preview}...")
                except Exception as e:
                    print(f"  ERROR in sample {i + 1}: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"  ✗ 错误: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='检查 ChromaDB 向量存储详情')
    parser.add_argument('--collection', type=str, help='指定要检查的 collection 名称')
    parser.add_argument('--show-samples', type=int, default=0, help='显示样本数据的数量')
    parser.add_argument('--path', type=str,
                       default='./rag_data/vector_store',
                       help='ChromaDB 存储路径')
    args = parser.parse_args()

    print("=" * 80)
    print("ChromaDB 向量存储检查工具")
    print("=" * 80)

    # 连接到 ChromaDB
    persist_directory = Path(args.path)
    if not persist_directory.exists():
        print(f"\n✗ 错误: 向量存储目录不存在: {persist_directory}")
        return

    print(f"\n📂 存储路径: {persist_directory.absolute()}")

    # 计算目录大小
    total_size = sum(f.stat().st_size for f in persist_directory.rglob('*') if f.is_file())
    print(f"💾 总大小: {format_size(total_size)}")

    # 初始化 ChromaDB 客户端
    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
    except Exception as e:
        print(f"\n✗ 错误: 无法连接到 ChromaDB: {str(e)}")
        return

    # 获取所有 collections
    collections = client.list_collections()
    print(f"\n📚 Collections 总数: {len(collections)}")

    if not collections:
        print("  ⚠️  没有找到任何 collection")
        return

    # 如果指定了 collection，只检查该 collection
    if args.collection:
        inspect_collection(client, args.collection, args.show_samples)
    else:
        # 列出所有 collections
        print(f"\n📋 Collection 列表:")
        collection_stats = []

        for collection in collections:
            try:
                count = collection.count()
                collection_stats.append((collection.name, count))
                print(f"  • {collection.name}: {count:,} documents")
            except Exception as e:
                print(f"  • {collection.name}: ✗ 错误 ({str(e)})")

        # 详细检查每个 collection
        for collection_name, count in collection_stats:
            inspect_collection(client, collection_name, args.show_samples)

    print(f"\n{'=' * 80}")
    print("检查完成！")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
