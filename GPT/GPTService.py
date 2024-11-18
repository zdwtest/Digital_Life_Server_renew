import json
import logging
import os
import time

import openai  # 确保安装：pip install openai

# 假设这些模块在正确的路径下
import GPT.machine_id
import GPT.tune as tune

# 配置日志记录，格式更易读
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

class GPTService:
    def __init__(self, args):
        logging.info('正在初始化通义千问服务...')
        self.tune = tune.get_tune(args.character, args.model)  # 获取个性化设置

        # 改进的 API 密钥和基础 URL 处理
        self.api_model = args.model or os.getenv("DASHSCOPE_API_MODEL") # Store api_model as an instance variable
        api_key = args.APIKey or os.getenv("DASHSCOPE_API_KEY")  # 优先使用命令行参数，否则使用环境变量
        base_url = args.baseUrl or os.getenv("DASHSCOPE_API_BASE") # 优先使用命令行参数，否则使用环境变量

        if not api_key:
            logging.error("API 密钥未找到。请设置 DASHSCOPE_API_KEY 环境变量或通过 --APIKey 命令行参数提供。")
            raise ValueError("API 密钥未找到。")

        openai.api_key = api_key  # 设置 API 密钥
        openai.api_base = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 设置 API 基础 URL，优先使用自定义的，否则使用默认的

        # 记录正在使用的 URL
        logging.info(f"正在使用 OpenAI API 基础 URL：{openai.api_base}")
        if base_url:
            logging.info("基础 URL 来自命令行参数或环境变量。")
        else:
            logging.info("正在使用默认基础 URL。")

        if not self.api_model:
            logging.error("模型未指定。请设置 DASHSCOPE_API_MODEL 环境变量或通过 --model 命令行参数提供。")
            raise ValueError("模型未指定。")

        logging.info(f'已指定模型: {self.api_model}') # Log the model being used
        logging.info('通义千问 API 机器人已初始化。')

    def ask(self, text):
        stime = time.time()  # 记录开始时间
        try:
            logging.debug(f'正在向通义千问发送请求：{self.tune + "\n" + text}')  # 日志记录请求内容
            completion = openai.ChatCompletion.create(
                model=self.api_model,  # Access api_model using self
                messages=[{'role': 'system', 'content': self.tune},  # 系统指令
                          {'role': 'user', 'content': text}],  # 用户输入
                stream=False  # 非流式响应
            )
            response = completion.choices[0].message.content  # 获取响应内容
            logging.info('通义千问响应：%s，耗时 %.2f 秒' % (response, time.time() - stime))  # 日志记录响应和耗时
            return response  # 返回响应
        except openai.error.OpenAIError as e:
            logging.exception(f'通义千问 API 错误：{e}')  # Log the exception for more details
            return f"错误：通义千问 API 请求失败: {e}"  # 返回错误信息
        except Exception as e:
            logging.exception(f'未知错误: {e}')  # 日志记录其他异常
            return "错误：发生未知错误。"  # 返回错误信息

    def ask_stream(self, text):
        stime = time.time()  # 记录开始时间
        try:
            completion = openai.ChatCompletion.create(
                model=self.api_model,  # Access api_model using self
                messages=[{'role': 'system', 'content': self.tune}, {'role': 'user', 'content': text}],  # 系统指令和用户输入
                stream=True,  # 流式响应
            )
            for chunk in completion:  # 循环处理流式响应
                response = chunk.choices[0].delta.content  # 获取响应片段
                if response:
                    yield response  # 生成器，逐段返回响应
                time.sleep(0.1)  # 暂停一小段时间
        except openai.error.OpenAIError as e:
            logging.exception(f'通义千问 API 流式传输错误：{e}') # Log the exception for more details
            yield f"错误：通义千问 API 流式传输请求失败: {e}"  # 返回错误信息
        except Exception as e:
            logging.exception(f'未知错误: {e}')  # 日志记录其他异常
            yield "错误：发生未知错误。"  # 返回错误信息