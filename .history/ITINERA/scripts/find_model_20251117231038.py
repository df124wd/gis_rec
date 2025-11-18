#!/usr/bin/env python3
"""
查找已下载的模型路径
"""

import os
from pathlib import Path

print("=" * 60)
print("查找本地模型")
print("=" * 60)

# HuggingFace缓存目录
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

print(f"\n缓存目录: {cache_dir}")

if not cache_dir.exists():
    print("❌ 缓存目录不存在")
    print("\n可能原因:")
    print("1. 模型还未下载")
    print("2. 使用了自定义缓存路径")
    exit(1)

# 查找bge模型
print("\n查找 bge-base-zh-v1.5 模型...")
model_dirs = list(cache_dir.glob("models--BAAI--bge-base-zh-v1.5"))

if not model_dirs:
    print("❌ 未找到模型")
    print("\n请先运行: python scripts/download_model.py")
    exit(1)

model_dir = model_dirs[0]
print(f"✓ 找到模型: {model_dir}")

# 查找snapshots
snapshots_dir = model_dir / "snapshots"
if snapshots_dir.exists():
    snapshots = list(snapshots_dir.iterdir())
    if snapshots:
        latest_snapshot = snapshots[0]
        print(f"✓ 模型路径: {latest_snapshot}")
        
        # 检查关键文件
        print("\n检查模型文件:")
        key_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "modules.json"
        ]
        
        all_exist = True
        for f in key_files:
            file_path = latest_snapshot / f
            if file_path.exists():
                size = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f} ({size:.1f} MB)")
            else:
                print(f"  ❌ {f} (缺失)")
                all_exist = False
        
        if all_exist:
            print("\n" + "=" * 60)
            print("✓ 模型完整，可以使用！")
            print("=" * 60)
            print("\n配置方法1（推荐）：使用模型名称")
            print('  "LOCAL_EMBEDDING_MODEL": "BAAI/bge-base-zh-v1.5"')
            print("\n配置方法2：使用本地路径")
            print(f'  "LOCAL_EMBEDDING_MODEL": "{latest_snapshot}"')
            print("\n下一步:")
            print("  python server.py")
        else:
            print("\n⚠️  模型文件不完整，建议重新下载")
            print("  python scripts/download_model.py")
    else:
        print("❌ 未找到模型快照")
else:
    print("❌ 未找到snapshots目录")
