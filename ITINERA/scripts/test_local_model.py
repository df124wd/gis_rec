#!/usr/bin/env python3
"""
测试项目本地模型路径
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("测试项目本地模型")
print("=" * 60)

# 设置配置
project_root = Path(__file__).parent.parent
model_path = project_root / "model" / "llm_model" / "models--BAAI--bge-base-zh-v1.5" / "snapshots" / "f03589ceff5aac7111bd60cfc7d497ca17ecac65"

print(f"\n项目根目录: {project_root}")
print(f"模型路径: {model_path}")

# 检查路径是否存在
if not model_path.exists():
    print("\n❌ 模型路径不存在！")
    print("\n请确认模型已移动到:")
    print(f"  {model_path}")
    sys.exit(1)

print("\n✓ 模型路径存在")

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
    file_path = model_path / f
    if file_path.exists():
        size = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {f} ({size:.1f} MB)")
    else:
        print(f"  ❌ {f} (缺失)")
        all_exist = False

if not all_exist:
    print("\n❌ 模型文件不完整")
    sys.exit(1)

print("\n✓ 模型文件完整")

# 测试加载模型
print("\n测试加载模型...")
try:
    # 设置离线模式
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['EMBEDDING_PROVIDER'] = 'local'
    os.environ['LOCAL_EMBEDDING_MODEL'] = str(model_path)
    
    from model.utils.proxy_call import OpenaiCall
    
    proxy = OpenaiCall()
    result = proxy.embedding(input_data=["测试文本"])
    
    if hasattr(result, 'data') and len(result.data) > 0:
        emb = result.data[0].embedding
        print(f"✓ 模型加载成功")
        print(f"  维度: {len(emb)}")
    else:
        print("❌ Embedding格式错误")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n配置已更新为:")
print(f'  "LOCAL_EMBEDDING_MODEL": "model/llm_model/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"')
print("\n下一步:")
print("  python server.py")
