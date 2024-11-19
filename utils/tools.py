# -*- coding: utf-8 -*-
'''
文件说明：此模块提供系统的扩展工具功能

实现了一系列辅助功能，如时间查询、文本格式化和计算等，
供LLM调用以增强其能力
'''

from typing import Dict, Any, List
import datetime
import requests
import json

class Tools:
    '''
    工具类
    
    提供一系列可供LLM调用的工具函数，包括时间查询、
    计算和文本处理等功能
    '''
    
    @staticmethod
    def get_current_time() -> Dict[str, Any]:
        '''
        获取当前时间
        
        返回:
            Dict[str, Any] - 包含当前时间和日期的字典
        '''
        current_time = datetime.datetime.now()
        return {
            "time": current_time.strftime("%H:%M:%S"),
            "date": current_time.strftime("%Y-%m-%d")
        }

    @staticmethod
    def calculate(expression: str) -> Dict[str, Any]:
        '''
        执行基础数学计算
        
        参数:
            expression: str - 数学表达式字符串
            
        返回:
            Dict[str, Any] - 包含计算结果或错误信息的字典
            
        安全说明:
            使用受限的eval环境，只允许基本数学运算
        '''
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def format_text(text: str, format_type: str) -> Dict[str, Any]:
        '''
        格式化文本
        
        参数:
            text: str - 待格式化的文本
            format_type: str - 格式化类型（upper/lower/title/capitalize）
            
        返回:
            Dict[str, Any] - 包含格式化后文本的字典
        '''
        formats = {
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
            "capitalize": text.capitalize()
        }
        return {"formatted_text": formats.get(format_type, text)}

    @staticmethod
    def get_tool_descriptions() -> List[Dict[str, Any]]:
        '''
        获取工具描述列表
        
        返回:
            List[Dict[str, Any]] - 工具函数的描述列表，用于函数调用
        '''
        return [
            {
                "name": "get_current_time",
                "description": "获取当前时间和日期",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "calculate",
                "description": "执行基本数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "format_text",
                "description": "文本格式化处理",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "待格式化的文本"
                        },
                        "format_type": {
                            "type": "string",
                            "enum": ["upper", "lower", "title", "capitalize"],
                            "description": "格式化类型"
                        }
                    },
                    "required": ["text", "format_type"]
                }
            }
        ]

