#!/usr/bin/env python3
"""
诊断脚本 - 检查所有配置和路径
"""

import os
import sys
import json
from pathlib import Path

print("=" * 60)
print("系统诊断")
print("=" * 60)

# 1. 检查配置文件
print("\n[1] 配置文件")
config_path = Path(__file__).parent.parent / "config" / "app_config.json"
print(f"路径: {config_path}")
print(f"存在: {config_path.exists()}")

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n关键配置:")
    print(f"  DEEPSEEK_API_KEY: {config.get('DEEPSEEK_API_KEY', '')[:10]}...")
    print(f"  EMBEDDING_PROVIDER: {config.get('EMBEDDING_PROVIDER')}")
    print(f"  LOCAL_EMBEDDING_MODEL: {config.get('LOCAL_EMBEDDING_MODEL')}")

# 2. 检查环境变量
print("\n[2] 环境变量")
env_keys = ['DEEPSEEK_API_KEY', 'EMBEDDING_PROVIDER', 'LOCAL_EMBEDDING_MODEL', 
            'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE', 'HF_ENDPOINT']
for key in env_keys:
    value = os.environ.get(key, '(未设置)')
    if 'KEY' in key and value != '(未设置)':
        value = value[:10] + '...'
    print(f"  {key}: {value}")

# 3. 检查模型路径
print("\n[3] 模型路径")
project_root = Path(__file__).parent.parent
model_rel_path = config.get('LOCAL_EMBEDDING_MODEL', '')

if model_rel_path:
    if '/' in model_rel_path or '\\' in model_rel_path:
        # 相对路径
        model_path = project_root / model_rel_path
        print(f"类型: 相对路径")
        print(f"完整路径: {model_path}")
        print(f"存在: {model_path.exists()}")
        
        if model_path.exists():
            # 检查关键文件
            key_files = ['config.json', 'pytorch_model.bin', 'modules.json']
            print("\n关键文件:")
            for f in key_files:
                fp = model_path / f
                if fp.exists():
                    size = fp.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {f} ({size:.1f} MB)")
                else:
                    print(f"  ❌ {f} (缺失)")
    else:
        # 模型名称
        print(f"类型: 模型名称")
        print(f"值: {model_rel_path}")
        print(f"将从HuggingFace缓存加载")
        
        # 检查缓存
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache = cache_dir / f"models--{model_rel_path.replace('/', '--')}"
        print(f"缓存路径: {model_cache}")
        print(f"缓存存在: {model_cache.exists()}")

# 4. 检查依赖
print("\n[4] Python依赖")
try:
    import sentence_transformers
    print(f"  ✓ sentence-transformers: {sentence_transformers.__version__}")
except ImportError:
    print(f"  ❌ sentence-transformers: 未安装")

try:
    import openai
    print(f"  ✓ openai: {openai.__version__}")
except ImportError:
    print(f"  ❌ openai: 未安装")

# 5. 建议
print("\n" + "=" * 60)
print("诊断结果")
print("=" * 60)

issues = []

if not config_path.exists():
    issues.append("配置文件不存在")

if config.get('EMBEDDING_PROVIDER') != 'local':
    issues.append("EMBEDDING_PROVIDER 未设置为 local")

if model_rel_path and ('/' in model_rel_path or '\\' in model_rel_path):
    model_path = project_root / model_rel_path
    if not model_path.exists():
        issues.append(f"本地模型路径不存在: {model_path}")

if issues:
    print("\n⚠️  发现问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n建议:")
    print("  1. 检查配置文件: config/app_config.json")
    print("  2. 确认模型已移动到项目目录")
    print("  3. 运行: python scripts/test_local_model.py")
else:
    print("\n✓ 所有检查通过！")
    print("\n下一步:")
    print("  python server.py")
