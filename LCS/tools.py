from langchain_community.tools import BaseTool, DuckDuckGoSearchRun
from langchain.agents import Tool
from datetime import datetime
from LCS.rag import RAGService
from typing import Optional
from pydantic import Field

class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base"
    description: str = """
    当需要查询历史文档、公司内部资料、技术文档等本地知识库中的信息时使用此工具。
    输入应该是具体的查询问题。
    不要用于查询实时或最新信息。
    """
    rag_service: RAGService
    
    def __init__(self, rag_service: RAGService, **data):
        super().__init__(rag_service=rag_service, **data)
    
    async def _arun(self, query: str) -> str:
        result = await self.rag_service.query(query)
        response = f"答案：{result['answer']}\n\n来源信息：\n"
        for i, source in enumerate(result['sources'], 1):
            response += f"{i}. {source[:200]}...\n"
        return response
    
    def _run(self, query: str) -> str:
        raise NotImplementedError("请使用异步版本")

class CurrentTimeTool(BaseTool):
    name: str = "获取当前时间"
    description: str = """
    当需要获取当前时间、日期信息时使用此工具。
    不需要任何输入参数。
    """
    
    async def _arun(self, query: Optional[str] = None) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _run(self, query: Optional[str] = None) -> str:
        raise NotImplementedError("请使用异步版本")

class CalculatorTool(BaseTool):
    name: str = "数学计算器"
    description: str = """
    用于执行数学计算，如加减乘除、平方根、指数等。
    输入应该是数学表达式。
    示例输入: '2 + 2' 或 'sqrt(16)' 或 '2^3'
    """
    
    async def _arun(self, query: str) -> str:
        try:
            return str(eval(query))
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _run(self, query: str) -> str:
        raise NotImplementedError("请使用异步版本")

async def create_tools(rag_service: RAGService):
    """创建完整的工具集"""
    tools = []
    
    # 添加知识库搜索
    tools.append(KnowledgeBaseTool(rag_service))
    
    # 添加网络搜索
    search = DuckDuckGoSearchRun()
    tools.append(Tool(
        name="网络搜索",
        func=search.run,
        coroutine=search.arun,  # 添加异步支持
        description="""
        用于搜索最新的互联网信息。
        输入应该是具体的搜索查询。
        """
    ))
    
    # 添加时间工具
    tools.append(CurrentTimeTool())
    
    # 添加计算器
    tools.append(CalculatorTool())
    
    return tools 