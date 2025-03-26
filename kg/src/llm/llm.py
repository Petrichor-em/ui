import os
import sys
import yaml
import json
import hashlib
from pathlib import Path
from langchain_openai import ChatOpenAI
import requests
from typing import Any, Dict, List
from dataclasses import dataclass

"""获取当前 Python 脚本所在项目的根目录路径"""
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CachedChatOpenAI:
	def __init__(self, base_llm):
		self.llm = base_llm
		self.cache_dir = Path(f"{project_dir}/.cache/llm_responses")
		self.cache_dir.mkdir(parents=True, exist_ok=True)

	def invoke(self, *args, **kwargs):
		# 生成缓存键
		input_str = json.dumps({"args": str(args), "kwargs": str(kwargs)}, sort_keys=True)
		cache_key = hashlib.md5(input_str.encode()).hexdigest()
		cache_file = self.cache_dir / f"{cache_key}.json"

		# 检查缓存
		if cache_file.exists():
			with open(cache_file, "r", encoding="utf-8") as f:
				cached_data = json.load(f)
				return cached_data["content"]

		# 调用API获取新响应
		try:
			result = self.llm.invoke(*args, **kwargs)
			if not hasattr(result, 'content'):
				raise ValueError("API response has no content")
			
			# 缓存响应
			cache_data = {"content": result.content}
			with open(cache_file, "w", encoding="utf-8") as f:
				json.dump(cache_data, f, ensure_ascii=False, indent=2)
			
			return result.content
		except Exception as e:
			raise ValueError(f"API调用失败: {str(e)}")

def get_llm(config_path: str):
    """
    基于配置文件获取语言模型

    参数：
        配置文件的路径

    返回值：
        具备缓存能力的语言模型
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    llm_name = config['llm']['use']
    llm_config = config["llm"].get(llm_name)

    if llm_name is None:
        raise ValueError("invalid llm use in config")
    
    """
    ChatOpenAI 的返回值通常是一个可以调用的实例，
    允许你使用输入的文本（如用户的消息）来获取模型生成的回复。
    调用这个实例时，你会传入用户的输入，模型会返回生成的文本。
    """
    base_llm = ChatOpenAI(
		model=llm_config["model"],
		openai_api_key=llm_config.get("api_key"),
		openai_api_base=llm_config["api_base"],
		temperature=llm_config.get("temperature"),
		max_tokens=llm_config.get("max_tokens"),
		timeout=llm_config.get("timeout")
	)

    return CachedChatOpenAI(base_llm)