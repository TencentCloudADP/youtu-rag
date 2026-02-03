"""
打印 MinIO 中的文件和元数据

功能：
1. 列出 MINIO_BUCKET (ufile) 和 MINIO_BUCKET_SYS (sysfile) 中的所有文件
2. 打印每个文件的完整元数据
3. 输出到 minio_file_info.txt

使用方法：
    python tests/rag/export_minio_files_info.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utu.rag.api.minio_client import minio_client


def print_bucket_files(bucket_name: str, output_file):

    output_file.write("=" * 100 + "\n")
    output_file.write(f"Bucket: {bucket_name}\n")
    output_file.write("=" * 100 + "\n\n")

    try:
        files = minio_client.list_files(bucket_name=bucket_name)

        if not files:
            output_file.write(f"(Bucket '{bucket_name}' is empty)\n\n")
            return

        output_file.write(f"Total files: {len(files)}\n\n")

        for idx, file_obj in enumerate(files, 1):
            filename = file_obj.object_name

            output_file.write("-" * 100 + "\n")
            output_file.write(f"File #{idx}: {filename}\n")
            output_file.write("-" * 100 + "\n")
            try:
                stat = minio_client.get_file_stat(filename, bucket_name=bucket_name)
                output_file.write(f"  Size: {stat.size:,} bytes ({stat.size / 1024 / 1024:.2f} MB)\n")
                output_file.write(f"  ETag: {stat.etag}\n")
                output_file.write(f"  Last Modified: {stat.last_modified}\n")
                output_file.write(f"  Content Type: {stat.content_type}\n")
            except Exception as e:
                output_file.write(f"  (Failed to get file stats: {e})\n")

            output_file.write(f"\n  Metadata:\n")
            try:
                metadata = minio_client.get_file_metadata(filename, bucket_name=bucket_name)

                if metadata:
                    sorted_keys = sorted(metadata.keys())

                    for key in sorted_keys:
                        value = metadata[key]
                        if isinstance(value, str) and len(value) > 200:
                            display_value = value[:200] + "... (truncated)"
                        else:
                            display_value = value
                        output_file.write(f"    • {key}: {display_value}\n")
                else:
                    output_file.write("    (No metadata)\n")

            except Exception as e:
                output_file.write(f"    (Failed to get metadata: {e})\n")

            output_file.write("\n")

    except Exception as e:
        output_file.write(f"Error listing files in bucket '{bucket_name}': {e}\n\n")


def main():
    output_path = project_root / "minio_file_info.txt"

    print("=" * 100)
    print("MinIO 文件和元数据打印工具")
    print("=" * 100)
    print(f"输出文件: {output_path.absolute()}")
    print()

    if not minio_client.check_health():
        print("✗ 错误: 无法连接到 MinIO")
        return

    print("✓ MinIO 连接成功")
    print()

    ufile_bucket = os.getenv("MINIO_BUCKET", "ufile")
    sysfile_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")

    print(f"正在扫描 buckets:")
    print(f"  • {ufile_bucket} (用户文件)")
    print(f"  • {sysfile_bucket} (系统文件)")
    print()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MinIO 文件和元数据信息\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        print(f"正在扫描 {ufile_bucket}...")
        print_bucket_files(ufile_bucket, f)

        print(f"正在扫描 {sysfile_bucket}...")
        print_bucket_files(sysfile_bucket, f)

        f.write("=" * 100 + "\n")
        f.write("扫描完成\n")
        f.write("=" * 100 + "\n")

    print()
    print("=" * 100)
    print(f"✓ 完成！文件已保存到: {output_path.absolute()}")
    print(f"  文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    print("=" * 100)


if __name__ == "__main__":
    main()
