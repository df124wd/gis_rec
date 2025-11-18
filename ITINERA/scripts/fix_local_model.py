#!/usr/bin/env python3
"""
修复本地模型：从HuggingFace缓存正确复制模型文件
"""

import os
import shutil
from pathlib import Path

print("=" * 60)
print("修复本地模型")
print("=" * 60)

# 源路径（HuggingFace缓存）
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
source_model = cache_dir / "models--BAAI--bge-base-zh-v1.5"

# 目标路径（项目目录）
project_root = Path(__file__).parent.parent
target_model = project_root / "model" / "llm_model" / "models--BAAI--bge-base-zh-v1.5"

print(f"\n源路径: {source_model}")
print(f"目标路径: {target_model}")

# 检查源是否存在
if not source_model.exists():
    print("\n❌ 源模型不存在！")
    print("\n请先运行: python scripts/download_model.py")
    exit(1)

print("\n✓ 源模型存在")

# 检查目标是否存在
if target_model.exists():
    print("\n⚠️  目标目录已存在")
    choice = input("是否删除并重新复制？(y/n): ").strip().lower()
    if choice == 'y':
        shutil.rmtree(target_model)
        print("✓ 已删除旧目录")
    else:
        print("取消操作")
        exit(0)

# 复制整个模型目录
print("\n开始复制模型文件...")
print("（这可能需要几分钟，模型约400MB）")

try:
    shutil.copytree(source_model, target_model)
    print("\n✓ 模型复制完成！")
    
    # 验证关键文件
    print("\n验证模型文件:")
    
    # 检查blobs目录
    blobs_dir = target_model / "blobs"
    if blobs_dir.exists():
        blob_files = list(blobs_dir.iterdir())
        print(f"  ✓ blobs目录: {len(blob_files)} 个文件")
    else:
        print("  ❌ blobs目录不存在")
    
    # 检查snapshots目录
    snapshots_dir = target_model / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = list(snapshots_dir.iterdir())
        print(f"  ✓ snapshots目录: {len(snapshot_dirs)} 个快照")
        
        # 找到最新的snapshot
        if snapshot_dirs:
            latest_snapshot = snapshot_dirs[0]
            key_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
            for f in key_files:
                file_path = latest_snapshot / f
                if file_path.exists():
                    size = file_path.stat().st_size / (1024 * 1024)
                    print(f"    ✓ {f} ({size:.1f} MB)")
                else:
                    print(f"    ❌ {f} (缺失)")
    else:
        print("  ❌ snapshots目录不存在")
    
    print("\n" + "=" * 60)
    print("✓ 修复完成！")
    print("=" * 60)
    
    # 获取snapshot路径
    if snapshot_dirs:
        snapshot_path = snapshot_dirs[0]
        relative_path = snapshot_path.relative_to(project_root)
        
        print("\n配置文件更新:")
        print(f'  "LOCAL_EMBEDDING_MODEL": "{relative_path.as_posix()}"')
        
        print("\n下一步:")
        print("1. 更新 config/app_config.json 中的 LOCAL_EMBEDDING_MODEL")
        print("2. 运行: python server.py")
    
except Exception as e:
    print(f"\n❌ 复制失败: {e}")
    exit(1)
