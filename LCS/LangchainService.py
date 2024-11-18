from typing import Union, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.base_language import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel

from LCS.rag import RAGService
from LCS.tools import create_tools

class LangchainService:
    async def _init_llm(self, llm_type: str, config: dict) -> Union[BaseLanguageModel, BaseChatModel]:
        """异步初始化LLM模型"""
        if llm_type in ["openai", "chatgpt"]:
            if self.openai_api_base:
                config['base_url'] = self.openai_api_base
            if self.openai_api_key:
                config['api_key'] = self.openai_api_key
            return ChatOpenAI(**config)
        elif llm_type == "llama":
            default_config = {
                "n_ctx": 4096,
                "temperature": 0.7,
                "max_tokens": 2048,
                "n_batch": 512,
            }
            if self.llama_model_path:
                default_config["model_path"] = self.llama_model_path
            default_config.update(config)
            return LlamaCpp(**default_config)
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")

    def __init__(self, 
                llm_type: str = "openai",
                model_config: dict = None,
                embedding_type: str = "openai",
                docs_dir: str = "docs",
                db_dir: str = "db",
                use_existing_db: bool = False,
                openai_api_base: Optional[str] = None,
                openai_api_key: Optional[str] = None,
                llama_model_path: Optional[str] = None,
                enable_agent: bool = False,
                callback=None):
        """初始化服务的基本属性"""
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key
        self.llama_model_path = llama_model_path
        self.callback = callback
        self.llm = None
        self.rag_service = None
        self.agent = None
        
        # 保存初始化参数供后续异步初始化使用
        self.init_params = {
            'llm_type': llm_type,
            'model_config': model_config or {},
            'embedding_type': embedding_type,
            'docs_dir': docs_dir,
            'db_dir': db_dir,
            'use_existing_db': use_existing_db,
            'enable_agent': enable_agent
        }

    async def async_init(self):
        """异步初始化方法"""
        self.llm = await self._init_llm(
            self.init_params['llm_type'], 
            self.init_params['model_config']
        )
        
        self.rag_service = await RAGService(
            embedding_type=self.init_params['embedding_type'],
            docs_dir=self.init_params['docs_dir'],
            db_dir=self.init_params['db_dir'],
            openai_api_base=self.openai_api_base,
            llm=self.llm
        ).async_init()
        
        if self.init_params['use_existing_db']:
            await self.rag_service.async_load_vector_store()
            await self.rag_service.async_init_qa_chain()
        
        if self.init_params['enable_agent']:
            await self._init_agent()
            
        return self

    @classmethod
    async def create(cls, *args, **kwargs):
        """工厂方法用于创建实例"""
        instance = cls(*args, **kwargs)
        return await instance.async_init()

    async def _init_agent(self):
        """异步初始化agent，添加回调"""
        tools = await create_tools(self.rag_service)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        class ProgressCallback(BaseCallbackHandler):
            def __init__(self, callback):
                self.callback = callback
                super().__init__()

            async def on_llm_start(self, serialized, prompts, **kwargs):
                """当 LLM 开始生成时触发"""
                if self.callback:
                    await self.callback("开始生成回答...")

            async def on_llm_end(self, response, **kwargs):
                """当 LLM 完成生成时触发"""
                if self.callback:
                    await self.callback("回答生成完成")

            async def on_tool_start(self, serialized, input_str, **kwargs):
                """当工具开始执行时触发"""
                tool_name = serialized.get("name", "unknown tool")
                if self.callback:
                    await self.callback(f"正在使用{tool_name}搜索相关信息...")

            async def on_tool_end(self, output, **kwargs):
                """当工具执行完成时触发"""
                if self.callback:
                    await self.callback("信息搜索完成")

            async def on_chain_start(self, serialized, inputs, **kwargs):
                """当链开始执行时触发"""
                chain_name = serialized.get("name", "unknown chain")
                if self.callback:
                    await self.callback(f"正在使用{chain_name}处理信息...")

            async def on_chain_end(self, outputs, **kwargs):
                """当链执行完成时触发"""
                if self.callback:
                    await self.callback("信息处理完成")

            async def on_agent_action(self, action, **kwargs):
                """当 Agent 决定采取行动时触发"""
                if self.callback:
                    await self.callback(f"Agent决定执行: {action.tool}")

            async def on_agent_finish(self, finish, **kwargs):
                """当 Agent 完成所有操作时触发"""
                if self.callback:
                    await self.callback("Agent完成所有操作")

        callbacks = [ProgressCallback(self.callback)] if self.callback else None
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            callbacks=callbacks
        )

    async def query(self, question: str) -> str:
        """异步查询方法"""
        if self.agent:
            return await self.agent.arun(question)
        else:
            return await self.rag_service.query(question)