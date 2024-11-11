'''
文件说明：此模块提供文本向量化功能

使用BAAI的BGE模型将文本转换为向量表示，支持批量处理
和本地模型加载
'''

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union
import numpy as np

class EmbeddingModel:
    '''
    文本向量化模型类
    
    使用BGE模型将文本转换为向量表示，支持GPU加速
    和本地模型加载
    '''
    
    def __init__(self, model_path: str = None):
        '''
        初始化向量化模型
        
        参数:
            model_path: Optional[str] - 本地模型路径，若为None则使用在线模型
        '''
        self.model_name = model_path if model_path else "BAAI/bge-large-zh"
        #print(f"model名称是：{self.model_name}")
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.model = AutoModel.from_pretrained(self.model_name)
        #self.word_embedding_model = self.models.Transformer(self.model_name, tokenizer=self.tokenizer, model=self.model)
        #self.model = SentenceTransformer(modules=[self.word_embedding_model])
        #self.model = SentenceTransformer(self.model_name,local_files_only=True)
        self.model = SentenceTransformer(self.model_name)
        #('vector model loaded')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(self.device)
        self.model.to(self.device)
        #print('all loaded successfully')

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        '''
        生成文本的向量表示
        
        参数:
            texts: Union[str, List[str]] - 输入文本或文本列表
            
        返回:
            np.ndarray - 文本的向量表示
            
        异常:
            Exception: 生成向量时的错误
        '''
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # BGE模型推荐的输入格式
            texts = [f"为这个句子生成表示：{text}" for text in texts]
            
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            raise Exception(f"生成嵌入向量时出错：{str(e)}")

    def get_embedding_dimension(self) -> int:
        '''
        获取向量维度
        
        返回:
            int - 向量维度
        '''
        return self.model.get_sentence_embedding_dimension()