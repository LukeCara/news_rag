'''
文件说明：此模块定义了系统中使用的核心数据模型

包含以下主要模型：
- Question: 用户查询模型
- Response: 系统响应模型
- Document: 文档块模型
'''

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Question(BaseModel):
    '''
    用户查询模型类
    
    用于封装用户的查询请求，包括查询文本和可选的上下文信息
    '''
    text: str = Field(..., min_length=1, max_length=1000)
    context: Optional[List[str]] = None

class Response(BaseModel):
    '''
    系统响应模型类
    
    封装系统对用户查询的响应，包括答案文本、参考来源、
    置信度和工具调用结果
    '''
    answer: str
    sources: Optional[List[str]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    tool_calls: Optional[List[Dict[str, Any]]] = None

class Document(BaseModel):
    '''
    文档块模型类
    
    表示经过分割的文档片段，包含文本内容和元数据信息
    '''
    content: str
    metadata: Dict[str, Any]