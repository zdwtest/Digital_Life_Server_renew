import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from LCS.LangchainService import LangchainService

class ProgressVisualizer:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )
        self.task = None

    async def progress_callback(self, message: str):
        """进度回调函数"""
        if self.task:
            self.task.description = message
        else:
            self.task = self.progress.add_task(message, total=None)

async def test_langchain_service():
    # 创建进度可视化器
    visualizer = ProgressVisualizer()
    
    # 初始化 LangchainService
    with visualizer.progress:
        lc_service = await LangchainService.create(
            llm_type="openai",
            model_config={
                "temperature": 0.7,
                "model_name": "gpt-3.5-turbo"
            },
            embedding_type="openai",
            docs_dir="your_docs_path",
            db_dir="your_db_path",
            use_existing_db=True,
            enable_agent=True,
            callback=visualizer.progress_callback,  # 注入进度回调
            openai_api_key="1111111",
            openai_api_base="1111111"
        )

        # 测试查询
        question = "请解释什么是机器学习？"
        response = await lc_service.query(question)
        
        # 打印结果
        visualizer.console.print("\n[bold green]回答:[/bold green]")
        visualizer.console.print(response)

if __name__ == "__main__":
    asyncio.run(test_langchain_service()) 