# example_service.py
from LCS.LangchainService import LangchainService

class ExampleService:
    async def initialize(self):
        # 创建 LangchainService 实例
        self.lc_service = await LangchainService.create(
            llm_type="openai",  # 可选: "openai", "chatgpt", "llama"
            model_config={
                "temperature": 0.7,
                "model_name": "gpt-3.5-turbo"  # 对于 OpenAI 模型
            },
            embedding_type="openai",
            docs_dir="your_docs_path",
            db_dir="your_db_path",
            use_existing_db=True,  # 如果已有向量数据库则设为 True
            enable_agent=True,  # 如果需要使用 agent 功能则设为 True
            openai_api_base="https://your-api-endpoint"  # 可选，如果使用自定义 API 端点
        )
    
    async def ask_question(self, question: str):
        # 使用 LangchainService 进行查询
        response = await self.lc_service.query(question)
        return response

# 使用示例
async def main():
    service = ExampleService()
    await service.initialize()
    
    # 进行查询
    answer = await service.ask_question("你的问题")
    print(answer)
    
# 如果你需要在同步环境中使用，可以这样包装：
# import asyncio

# def sync_query(question: str):
#     async def run():
#         service = ExampleService()
#         await service.initialize()
#         return await service.ask_question(question)
    
#     return asyncio.run(run())