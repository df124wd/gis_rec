import requests
import json
import logging
import os

from openai import OpenAI

class OpenaiCall:
    """
    统一的LLM调用代理，支持OpenAI和DeepSeek
    优先使用DeepSeek（成本更低），Embedding可配置使用本地模型
    """
    def __init__(self, api_key=None, base_url=None, provider=None):
        """
        Args:
            api_key: API密钥，优先级：传入参数 > 环境变量
            base_url: API地址
            provider: 'deepseek' 或 'openai'，默认自动检测
        """
        # 自动检测provider
        if provider is None:
            # 优先使用DeepSeek（成本低）
            if os.getenv("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "deepseek"  # 默认
        
        self.provider = provider
        
        # 根据provider设置API Key和Base URL
        if self.provider == "deepseek":
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
            self.default_chat_model = "deepseek-chat"
            self.default_embedding_model = "deepseek-chat"  # DeepSeek暂无专用embedding模型
            print(f"[LLM] 使用DeepSeek API (base_url={base_url})")
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                base_url
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_PROXY_BASE")
            )
            self.default_chat_model = "gpt-3.5-turbo-1106"
            self.default_embedding_model = "text-embedding-ada-002"
            print(f"[LLM] 使用OpenAI API (base_url={base_url or 'default'})")
        
        if not api_key:
            raise ValueError(f"未找到{self.provider.upper()}_API_KEY，请在环境变量或配置文件中设置")
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, messages, model=None, temperature=0):
        """
        Chat补全接口
        """
        # 使用默认模型（根据provider自动选择）
        if model is None:
            model = self.default_chat_model
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def stream_chat(self, messages, model=None, temperature=0):
        """
        流式Chat接口
        """
        if model is None:
            model = self.default_chat_model
        
        for chunk in self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        ): 
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content is not None:
                    yield content
    
    def embedding(self, input_data, model=None):
        """
        Embedding接口
        支持：
        1. OpenAI text-embedding-ada-002
        2. DeepSeek (不支持，需使用本地模型)
        3. 本地模型 (通过EMBEDDING_PROVIDER=local配置)
        """
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        
        # 使用本地Embedding模型
        if embedding_provider == "local":
            return self._local_embedding(input_data)
        
        # DeepSeek没有专用embedding接口，强制使用本地模型
        if self.provider == "deepseek":
            print("[警告] DeepSeek不支持Embedding API，自动切换到本地模型")
            return self._local_embedding(input_data)
        
        # 使用OpenAI Embedding
        if model is None:
            model = self.default_embedding_model
        
        response = self.client.embeddings.create(
            input=input_data,
            model=model
        )
        return response
    
    def _local_embedding(self, input_data):
        """
        使用本地Embedding模型（需要安装sentence-transformers）
        推荐模型：
        - BAAI/bge-large-zh-v1.5 (中文最佳)
        - BAAI/bge-base-zh-v1.5 (中文平衡)
        - shibing624/text2vec-base-chinese (轻量)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "使用本地Embedding需要安装sentence-transformers:\n"
                "pip install sentence-transformers"
            )
        
        # 设置HuggingFace镜像（国内加速）
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print("[Embedding] 已设置HuggingFace镜像: https://hf-mirror.com")
        
        # 获取模型路径
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        
        # 缓存模型实例
        if not hasattr(self, '_local_model'):
            print(f"[Embedding] 加载本地模型: {model_name}")
            self._local_model = SentenceTransformer(model_name)
        
        # 处理输入格式
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list):
            texts = input_data
        else:
            texts = [str(input_data)]
        
        # 生成embeddings
        embeddings = self._local_model.encode(texts, normalize_embeddings=True)
        
        # 返回OpenAI兼容格式
        class EmbeddingResponse:
            def __init__(self, embeddings):
                self.data = [
                    type('obj', (object,), {'embedding': emb.tolist()})()
                    for emb in embeddings
                ]
        
        return EmbeddingResponse(embeddings)