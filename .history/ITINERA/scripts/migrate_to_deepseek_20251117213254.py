#!/usr/bin/env python3
"""
迁移脚本：从OpenAI切换到DeepSeek + 本地Embedding
节省成本，提升中文效果
"""

import os
import sys
import json
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def backup_config():
    """备份原配置"""
    config_path = Path(__file__).parent.parent / "config" / "app_config.json"
    backup_path = config_path.with_suffix('.json.backup')
    
    if config_path.exists():
        shutil.copy(config_path, backup_path)
        print(f"✓ 已备份配置到: {backup_path}")
    return config_path

def update_config(config_path):
    """更新配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 清空OpenAI配置（节省余额）
    config['OPENAI_API_KEY'] = ""
    config['OPENAI_BASE_URL'] = ""
    
    # 确保DeepSeek配置存在
    if not config.get('DEEPSEEK_API_KEY'):
        print("⚠️  警告：未找到DEEPSEEK_API_KEY，请手动配置")
    
    # 切换到本地Embedding
    config['EMBEDDING_PROVIDER'] = "local"
    config['LOCAL_EMBEDDING_MODEL'] = "BAAI/bge-base-zh-v1.5"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✓ 已更新配置文件")

def clean_old_embeddings():
    """清理旧的embedding文件"""
    data_dir = Path(__file__).parent.parent / "model" / "data"
    npy_files = list(data_dir.glob("*.npy"))
    
    if npy_files:
        print(f"\n发现 {len(npy_files)} 个旧的embedding文件:")
        for f in npy_files:
            print(f"  - {f.name}")
        
        choice = input("\n是否删除？(y/n): ").strip().lower()
        if choice == 'y':
            for f in npy_files:
                f.unlink()
                print(f"✓ 已删除: {f.name}")
            print("\n下次启动服务时会自动重新生成embedding")
        else:
            print("⚠️  保留旧文件，可能导致维度不匹配错误")

def install_dependencies():
    """安装依赖"""
    print("\n正在检查依赖...")
    try:
        import sentence_transformers
        print("✓ sentence-transformers 已安装")
    except ImportError:
        print("⚠️  未安装 sentence-transformers")
        choice = input("是否现在安装？(y/n): ").strip().lower()
        if choice == 'y':
            os.system("pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")

def download_model():
    """预下载模型"""
    print("\n正在下载本地Embedding模型...")
    print("模型: BAAI/bge-base-zh-v1.5 (约400MB)")
    
    choice = input("是否现在下载？(y/n): ").strip().lower()
    if choice == 'y':
        try:
            from sentence_transformers import SentenceTransformer
            print("下载中，请稍候...")
            model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
            print("✓ 模型下载完成")
        except Exception as e:
            print(f"⚠️  下载失败: {e}")
            print("提示：可以设置镜像源后重试:")
            print("  export HF_ENDPOINT=https://hf-mirror.com")

def main():
    print("=" * 60)
    print("迁移到DeepSeek + 本地Embedding")
    print("=" * 60)
    
    # 1. 备份配置
    print("\n[1/5] 备份配置文件...")
    config_path = backup_config()
    
    # 2. 更新配置
    print("\n[2/5] 更新配置...")
    update_config(config_path)
    
    # 3. 清理旧embedding
    print("\n[3/5] 清理旧embedding...")
    clean_old_embeddings()
    
    # 4. 安装依赖
    print("\n[4/5] 检查依赖...")
    install_dependencies()
    
    # 5. 下载模型
    print("\n[5/5] 下载模型...")
    download_model()
    
    print("\n" + "=" * 60)
    print("✓ 迁移完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查 config/app_config.json 中的 DEEPSEEK_API_KEY")
    print("2. 运行: python server.py")
    print("3. 首次启动会自动生成新的embedding文件")
    print("\n预计节省成本: 100% (Embedding) + 90% (Chat)")
    print("=" * 60)

if __name__ == "__main__":
    main()
