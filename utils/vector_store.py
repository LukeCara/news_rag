'''
文件说明：此模块实现了基于FAISS的向量存储功能

负责管理文档向量的存储、检索和持久化，使用FAISS库进行
高效的相似度搜索
'''

import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
from utils.schema import Document

class VectorStore:
    '''
    向量存储类
    
    使用FAISS实现高效的向量存储和检索功能，支持文档的添加、
    搜索以及向量存储的保存和加载
    '''
    
    def __init__(self, dimension: int):
        '''
        初始化向量存储
        
        参数:
            dimension: int - 向量维度
        '''
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        #self.index=
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        '''
        添加文档及其对应的向量表示到存储中
        
        参数:
            documents: List[Document] - 待添加的文档列表
            embeddings: np.ndarray - 文档对应的向量表示
            
        异常:
            ValueError: 文档数量与向量数量不匹配时抛出
            Exception: 添加过程中的其他错误
        '''
        try:
            if len(documents) != embeddings.shape[0]:
                raise ValueError("文档数量与向量数量不匹配")
            
            self.documents.extend(documents)
            self.index.add(embeddings)
        except Exception as e:
            raise Exception(f"向量存储添加文档时出错：{str(e)}")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        '''
        使用查询向量搜索相似文档
        
        参数:
            query_embedding: np.ndarray - 查询文本的向量表示
            k: int - 返回的最相似文档数量
            
        返回:
            List[Tuple[Document, float]] - 文档和相似度分数的列表
            
        异常:
            Exception: 搜索过程中的错误
        '''
        try:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1:  # FAISS返回-1表示空槽位
                    results.append((self.documents[idx], float(distance)))
            
            return results
        except Exception as e:
            raise Exception(f"向量存储搜索时出错：{str(e)}")

    def save(self, path: str):
        '''
        保存向量存储到磁盘
        
        参数:
            path: str - 保存路径
            
        异常:
            Exception: 保存过程中的错误
        '''
        try:
            faiss.write_index(self.index, f"{path}.index")
            with open(f"{path}.docs", 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            raise Exception(f"保存向量存储时出错：{str(e)}")

    def load(self, path: str):
        '''
        从磁盘加载向量存储
        
        参数:
            path: str - 加载路径
            
        异常:
            Exception: 加载过程中的错误
        '''
        try:
            self.index = faiss.read_index(f"{path}.index")
            with open(f"{path}.docs", 'rb') as f:
                self.documents = pickle.load(f)
        except Exception as e:
            raise Exception(f"加载向量存储时出错：{str(e)}")

    def add_vectors(self, vectors):
        self.index.add(vectors)  # Add vectors to the index

    def search_vectors(self, query_vector, k=10):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices