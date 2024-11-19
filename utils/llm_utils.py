# -*- coding: utf-8 -*-
'''
文件说明：此模块实现了LLM处理和响应生成功能

负责管理与本地LLM的交互，包括提示词构建、响应生成
和工具调用等核心功能
'''

from typing import Dict, Any, List, Optional
import json
from openai import OpenAI
from utils import chat_history
from utils.tools import Tools
from utils.schema import Response
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMHandler:
    '''
    LLM处理器类
    
    管理与本地LLM的交互，包括模型加载、提示词构建、
    响应生成和工具调用等功能
    '''
    
    def __init__(self, model_path: str):
        '''
        初始化LLM处理器
        
        参数:
            model_path: str - 本地模型路径
            
        异常:
            Exception: 模型加载失败时抛出
        '''
        try:
            #self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",local_files_only=True)
            #需要加上local_files_only=True,速度会变快
            #self.model= AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",device_map="auto")
            #self.model = AutoModelForCausalLM.from_pretrained(model_path,)

                #model_type="qwen",
                #gpu_layers=35  # 适用于Qwen模型
            #)
            self.tools = Tools()
            self.tool_descriptions = self.tools.get_tool_descriptions()
        except Exception as e:
            raise Exception(f"初始化LLM失败：{str(e)}")

    def generate_response(
        self, 
        question: str, 
        context: Optional[List[str]] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> Response:
        '''
        生成对用户问题的响应
        
        参数:
            question: str - 用户问题
            context: Optional[List[str]] - 相关文档上下文
            chat_history: Optional[List[Dict]] - 对话历史记录
            
        返回:
            Response - 生成的响应对象
            
        异常:
            处理错误时返回错误响应
        '''
        try:
            # 准备带有上下文和对话历史的提示词
            prompt = self._prepare_prompt(question, context, chat_history) #提供最初的提示词，包含上下文、历史对话、工具描述等
            #print(prompt)
            # 生成中间结果,大模型挑选工具（前提是大模型）
            response = self._generate_text(prompt)
            # 解析响应和处理工具调用
            parsed_response = self._parse_response(response)
            print(f"parsed response is{parsed_response}")
            # 执行工具调用并生成最终响应
            if "tool_calls" in parsed_response:
                print("开始调用工具")
                tool_results = self._execute_tool_calls(parsed_response["tool_calls"])
                final_response = self._generate_text(
                    self._prepare_followup_prompt(question, parsed_response, tool_results)
                )
                parsed_response = self._parse_response(final_response)
            
            return Response(
                answer=parsed_response["answer"],
                sources=parsed_response.get("sources", []),
                confidence=parsed_response.get("confidence", 0.8),
                tool_calls=parsed_response.get("tool_calls", [])
            )
            
        except Exception as e:
            return Response(
                answer=f"抱歉，我遇到了一个错误：{str(e)}",
                confidence=0.0
            )

    def _prepare_prompt(
        self, 
        question: str, 
        context: Optional[List[str]] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        '''
        准备发送给LLM的提示词
        
        参数:
            question: str - 用户问题
            context: Optional[List[str]] - 相关文档上下文
            chat_history: Optional[List[Dict]] - 对话历史记录
            
        返回:
            str - 格式化的提示词
        '''
        prompt = "你是一个有帮助的AI助手。"
        
        # 添加对话历史上下文
        if chat_history:
            prompt += "\n之前的对话历史：\n"
            for msg in chat_history[-3:]:  # 只包含最近3轮对话
                role = "用户" if msg["role"] == "user" else "助手"
                prompt += f"{role}：{msg['content']}\n"
        
        # 添加文档上下文
        if context:
            prompt += "\n参考以下上下文：\n"
            prompt += "\n".join(context) + "\n"
        
        prompt += f"\n当前问题：{question}\n"
        prompt += "可用工具：\n"
        prompt += json.dumps(self.tool_descriptions, ensure_ascii=False,indent=2) + "\n" #确保输入的描述为中文
        #prompt += json.dumps(self.tool_descriptions, indent=2) + "\n"

        print(f"1----输入问题后，_prepare_prompt为----\n：{prompt}\n ------------")
        return prompt

    def _generate_text(self, prompt: str) -> str:
        '''
        使用LLM生成文本,该函数生成的为中间结果以及一次性结果（非流式），还需要进一步进行解析
        
        参数:
            prompt: str - 输入提示词
            
        返回:
            str - 生成的文本
            
        异常:
            Exception: 生成过程中的错误
        '''
        try:
            response=chat_history.chat(prompt, chat_history.history)
            return response
        except Exception as e:
            raise Exception(f"生成文本时出错：{str(e)}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        '''
        解析LLM的响应文本
        
        参数:
            response: str - LLM的原始响应文本
            
        返回:
            Dict[str, Any] - 解析后的响应字典
        '''
        try:
            # 尝试解析为JSON
            print(f"解析为json格式为{json.loads(response)}")
            return json.loads(response)
        except:
            #print(f"response.strip()为{response.strip()}")
            # 降级为基础文本响应
            return {
                "answer": response.strip(),
                "confidence": 0.8
            }

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        执行工具调用
        
        参数:
            tool_calls: List[Dict[str, Any]] - 工具调用请求列表
            
        返回:
            List[Dict[str, Any]] - 工具调用结果列表
        '''
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("arguments", {})
            
            tool_method = getattr(self.tools, tool_name, None)
            if tool_method:
                try:
                    result = tool_method(**tool_args)
                    results.append({
                        "name": tool_name,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "name": tool_name,
                        "error": str(e)
                    })
        return results

    def _prepare_followup_prompt(
        self, 
        question: str, 
        initial_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]]
    ) -> str:
        '''
        准备工具调用后的跟进提示词d
        
        参数:
            question: str - 原始问题
            initial_response: Dict[str, Any] - 初始响应
            tool_results: List[Dict[str, Any]] - 工具调用结果
            
        返回:
            str - 格式化的跟进提示词
        '''
        prompt = f"原始问题：{question}\n"
        prompt += "工具执行结果：\n"
        prompt += json.dumps(tool_results, indent=2) + "\n"
        prompt += "请根据工具执行结果提供最终答案。"
        print(f"_prepare follow up 的提示为：{prompt}")
        return prompt