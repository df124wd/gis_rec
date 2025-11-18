#!/usr/bin/env python3
"""
手动下载Embedding模型
"""

import os
import sys

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("下载本地Embedding模型")
print("=" * 60)
print("\n使用镜像: https://hf-mirror.com")
print("模型: BAAI/bge-base-zh-v1.5")
print("大小: 约400MB")
print("\n开始下载...\n")

try:
    from sentence_transformers import SentenceTransformer
    
    # 下载模型
    model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    
    print("\n" + "=" * 60)
    print("✓ 模型下载成功！")
    print("=" * 60)
    
    # 测试模型
    print("\n测试模型...")
    embeddings = model.encode(["测试文本"], normalize_embeddings=True)
    print(f"✓ 模型工作正常，维度: {len(embeddings[0])}")
    
    print("\n下一步:")
    print("1. 运行: python server.py")
    print("2. 或运行: python scripts/test_config.py")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n解决方案:")
    print("1. 检查网络连接")
    print("2. 尝试使用VPN")
    print("3. 或使用更小的模型:")
    print("   修改 config/app_config.json:")
    print('   "LOCAL_EMBEDDING_MODEL": "shibing624/text2vec-base-chinese"')
    sys.exit(1)
