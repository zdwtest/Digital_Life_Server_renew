from llama_cpp import Llama
from typing import Optional

class LLamaService:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        初始化LLama服务
        
        Args:
            model_path: llama模型文件的路径 (.gguf 格式)
            n_ctx: 上下文窗口大小
            n_threads: 使用的CPU线程数，默认None会自动选择
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: int = 512, 
                         temperature: float = 0.7,
                         top_p: float = 0.95) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            top_p: 采样参数
            
        Returns:
            生成的回复文本
        """
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return output["choices"][0]["text"]

# # 实例化服务
# llm_service = LLamaService(
#     model_path="/path/to/your/model.gguf",  # 替换为你的模型路径
#     n_ctx=2048  # 根据需要调整上下文窗口大小
# )

# # 生成回复
# response = llm_service.generate_response(
#     prompt="你好,请介绍一下自己",
#     max_tokens=512
# )
# print(response)
