#!/usr/bin/env python3
"""
配置测试脚本 - 验证DeepSeek和本地Embedding配置是否正确
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config():
    """测试配置文件"""
    print("=" * 60)
    print("配置测试")
    print("=" * 60)
    
    config_path = Path(__file__).parent.parent / "config" / "app_config.json"
    
    if not config_path.exists():
        print("❌ 配置文件不存在:", config_path)
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n[1/4] 检查DeepSeek配置...")
    deepseek_key = config.get('DEEPSEEK_API_KEY', '')
    if deepseek_key and deepseek_key.startswith('sk-'):
        print(f"✓ DEEPSEEK_API_KEY: {deepseek_key[:10]}...")
        os.environ['DEEPSEEK_API_KEY'] = deepseek_key
    else:
        print("❌ DEEPSEEK_API_KEY 未配置或格式错误")
        return False
    
    print("\n[2/4] 检查Embedding配置...")
    embedding_provider = config.get('EMBEDDING_PROVIDER', 'openai')
    if embedding_provider == 'local':
        print(f"✓ EMBEDDING_PROVIDER: local")
        model_name = config.get('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-base-zh-v1.5')
        print(f"✓ LOCAL_EMBEDDING_MODEL: {model_name}")
        os.environ['EMBEDDING_PROVIDER'] = 'local'
        os.environ['LOCAL_EMBEDDING_MODEL'] = model_name
    else:
        print(f"⚠️  EMBEDDING_PROVIDER: {embedding_provider} (建议使用local)")
    
    print("\n[3/4] 测试LLM连接...")
    try:
        from model.utils.proxy_call import OpenaiCall
        proxy = OpenaiCall()
        
        # 测试简单对话
        response = proxy.chat(
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            temperature=0
        )
        
        if "测试成功" in response or "成功" in response:
            print(f"✓ LLM连接成功")
            print(f"  响应: {response[:50]}...")
        else:
            print(f"⚠️  LLM响应异常: {response[:100]}")
    except Exception as e:
        print(f"❌ LLM连接失败: {e}")
        return False
    
    print("\n[4/4] 测试Embedding...")
    try:
        if embedding_provider == 'local':
            print("  正在加载本地模型（首次需要下载，约400MB）...")
            result = proxy.embedding(input_data=["测试文本"])
            
            if hasattr(result, 'data') and len(result.data) > 0:
                emb = result.data[0].embedding
                print(f"✓ Embedding生成成功")
                print(f"  维度: {len(emb)}")
            else:
                print("❌ Embedding格式错误")
                return False
        else:
            print("⚠️  跳过本地Embedding测试（未启用）")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("  请运行: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Embedding测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行: python server.py")
    print("2. 访问: http://localhost:8000")
    print("3. 测试推荐功能")
    
    return True

def test_api_key():
    """测试API Key有效性"""
    print("\n[额外] 测试DeepSeek API Key有效性...")
    try:
        import requests
        
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("⚠️  未设置DEEPSEEK_API_KEY")
            return
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✓ DeepSeek API Key 有效")
            result = response.json()
            usage = result.get('usage', {})
            print(f"  本次消耗: {usage.get('total_tokens', 0)} tokens")
        elif response.status_code == 401:
            print("❌ API Key 无效或已过期")
        elif response.status_code == 429:
            print("⚠️  API 限流，但Key有效")
        else:
            print(f"⚠️  API返回异常: {response.status_code}")
            print(f"  {response.text[:200]}")
    except Exception as e:
        print(f"⚠️  API测试失败: {e}")

if __name__ == "__main__":
    try:
        success = test_config()
        if success:
            test_api_key()
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n测试中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
