import os
import numpy as np
import pandas as pd


class SearchEngine:
    def __init__(self, embedding: np.ndarray = None, emb_path: str = "", file_path: str = "", proxy = None):
        # 先设置代理，再决定是否生成embedding，避免在get_embeddings中访问未设置的self.proxy
        self.proxy = proxy
        self.emb_path = emb_path
        self.file_path = file_path
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = self.get_embeddings(emb_path=emb_path, file_path=file_path)

    def top_k_cosine_similarity(self, A: np.ndarray = None, B: np.ndarray = None, k: int = None, indices: list = None):
        """
        Calculate the top-k cosine similarities between vectors in set A and set B.

        Args:
            A (np.ndarray, optional): An array of vectors of shape (m, emb_dim) representing set A.
            B (np.ndarray, optional): An array of vectors of shape (n, emb_dim) representing set B.
            k (int, optional): The number of top similarities to return.
            indices (list, optional): A list of indices to consider for top-k cosine similarities.

        Returns:
            tuple: A tuple containing two elements:
                - np.ndarray: An array of shape (k,) containing the indices of the top-k cosine similarities.
                - np.ndarray: An array of shape (k,) containing the top-k cosine similarity scores.
        """
        # Normalize the vectors
        A_norm = A / np.linalg.norm(A)
        B_norm = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

        # Compute the cosine similarity
        cosine_similarities = np.dot(A_norm, B_norm.T)

        # If indices are provided, replace all other cosine similarities with negative infinity so that they don't get considered
        if indices is not None:
            mask = np.ones(cosine_similarities.shape, dtype=bool)
            mask[0][indices] = False
            cosine_similarities[0][mask[0]] = -np.inf

        # Get the top-k indices
        top_k_indices = np.argsort(cosine_similarities[0])[::-1][:k]

        # Get the top-k cosine similarities
        top_k_similarities = cosine_similarities[0][top_k_indices]

        return top_k_indices, top_k_similarities


    def get_embeddings(self, emb_path: str = "", file_path: str = "", force: bool = False) -> None: 
        """
        Retrieves embeddings from the specified 'emb_path'
        if no embedding exists, then load context from 'file_path' and compute embedding though openai api calls.

        Args:
            emb_path (str): The path to the embeddings file (numpy array).
            file_path (str, optional): The path to the original context (pandas dataframe).
            save (bool, optional, default to True): save the computed embeddings to the emb_path (numpy array).

        Returns:
            None
        """
        # 当 force=True 时忽略已有文件，按当前提供商重新生成，确保维度对齐
        if os.path.exists(emb_path) and not force:
            embedding = np.load(emb_path)
        else:
            data = pd.read_csv(file_path)
            if self.proxy is None:
                raise RuntimeError("SearchEngine.proxy 未设置，无法生成embedding。请传入有效的proxy或提供现有的embedding/emb_path。")
            # 若已有context列，优先使用
            if 'context' in data.columns:
                context = data['context'].astype(str)
            else:
                # 兼容真实数据的列名，生成简洁文本描述（避免三元表达式跨行导致语法错误）
                if 'name' in data.columns:
                    name = data['name'].astype(str)
                elif '宗地坐落' in data.columns:
                    name = data['宗地坐落'].astype(str)
                else:
                    name = data.index.astype(str)
                if 'address' in data.columns:
                    address = data['address'].astype(str)
                elif '宗地坐落' in data.columns:
                    address = data['宗地坐落'].astype(str)
                else:
                    address = name
                # 缺失列使用空Series，保证长度与数据一致
                usage = (data['土地用途'].astype(str) if '土地用途' in data.columns else pd.Series([''] * len(data)))
                area = (data['宗地面积(平方米)'].astype(str) if '宗地面积(平方米)' in data.columns else pd.Series([''] * len(data)))
                price = (data['挂牌起始价(万元)'].astype(str) if '挂牌起始价(万元)' in data.columns else pd.Series([''] * len(data)))
                desc = ("用途:" + usage + "，面积:" + area + "㎡，起始价:" + price + "万元").str.strip()
                context = name + "，地址是" + address + "，" + desc
            res_records = self.proxy.embedding(input_data=context.tolist())
            try:
                embedding = np.array([np.array(record["embedding"]) for record in res_records["data"]])
            except Exception:
                # 兼容不同proxy返回结构
                embedding = np.array([np.array(record.embedding) for record in res_records.data])
            # 始终覆盖写入最新embedding
            try:
                np.save(emb_path, embedding)
            except Exception:
                pass

        return embedding

    def query(self, desc: tuple = None, top_k: int = None):
        """
        query the existing vector database and return the top_k ids and similarity scores

        Args:
            desc (tuple): The user pos reqs and neg reqs.
            top_k (int)

        Returns:
            numpy array with shape (top_k, 2)
            The first column indicates the queried ids and the second column indicates the similarity scores.
        """
        try:
            pos_desc, neg_desc = desc

            pos_res = self.proxy.embedding(input_data=f"{pos_desc}")
            try:
                pos_embedding = np.array([np.array(record["embedding"]) for record in pos_res["data"]])
            except:
                pos_embedding = np.array([np.array(record.embedding) for record in pos_res.data])

            # 若维度不一致，自动重算数据集embedding以对齐当前提供商维度
            try:
                if self.embedding is None or (self.embedding.size > 0 and self.embedding.shape[1] != pos_embedding.shape[1]):
                    self.embedding = self.get_embeddings(emb_path=self.emb_path, file_path=self.file_path, force=True)
            except Exception as _e:
                pass

            indices, similarities = self.top_k_cosine_similarity(pos_embedding, self.embedding, k=100000000)
            
            # 确保indices和similarities是一维数组
            if indices.ndim > 1:
                indices = indices.flatten()
            if similarities.ndim > 1:
                similarities = similarities.flatten()
                
            if neg_desc not in [None, ""]:
                sorted_indices = np.argsort(indices)
                indices = indices[sorted_indices]
                similarities = similarities[sorted_indices]

                neg_res = self.proxy.embedding(input_data=f"{neg_desc}")
                try:
                    neg_embedding = np.array([np.array(record["embedding"]) for record in neg_res["data"]])
                except:
                    neg_embedding = np.array([np.array(record.embedding) for record in neg_res.data])
                neg_indices, neg_similarities = self.top_k_cosine_similarity(neg_embedding, self.embedding, k=100000000, indices=indices)
                
                # 确保neg_similarities是一维数组
                if neg_similarities.ndim > 1:
                    neg_similarities = neg_similarities.flatten()
                
                sorted_indices = np.argsort(neg_indices)
                neg_indices = neg_indices[sorted_indices]
                neg_similarities = neg_similarities[sorted_indices]

                mean_similarity = np.mean(similarities)
                
                for i in range(len(neg_similarities)):
                    similarities[i] -= neg_similarities[i]
                similarities += (mean_similarity - np.mean(similarities)) # back to original similarities.

                sorted_indices = np.argsort(similarities)[::-1]
                indices = indices[sorted_indices]
                similarities = similarities[sorted_indices]
            
            # 限制返回结果数量
            if top_k is not None and top_k > 0:
                indices = indices[:top_k]
                similarities = similarities[:top_k]
            
            # 确保返回的数组形状正确
            result = np.column_stack((indices, similarities))
            
            return result
            
        except Exception as e:
            print(f"SearchEngine.query出错: {e}")
            return np.array([]).reshape(0, 2)  # 返回空的二维数组

