from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PDFMinerLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader, 
    CSVLoader
)
from typing import Optional, List
import os
from langchain_core.runnables import RunnablePassthrough

class RAGService:
    def __init__(self,
                 embedding_type: str = "openai",
                 docs_dir: str = "docs",
                 db_dir: str = "db",
                 openai_api_base: Optional[str] = None,
                 llm = None):
        """
        初始化RAG服务
        
        Args:
            embedding_type: 嵌入模型类型 ("openai", "huggingface")
            docs_dir: 知识库文档目录
            db_dir: 向量数据库存储目录
            openai_api_base: OpenAI API的基础地址
            llm: 使用的LLM模型实例
        """
        self.embeddings = self._init_embeddings(embedding_type, openai_api_base)
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.vector_store = None
        self.llm = llm
        self.qa_chain = None
    
    def _init_embeddings(self, embedding_type: str, openai_api_base: Optional[str] = None):
        """初始化嵌入模型"""
        kwargs = {}
        if openai_api_base:
            kwargs["openai_api_base"] = openai_api_base
        if embedding_type == "openai":
            kwargs["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
            return OpenAIEmbeddings(**kwargs)
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"不支持的嵌入模型类型: {embedding_type}")
    
    def load_documents(self, file_pattern: str = "**/*.*") -> List:
        """加载多种格式的文档"""
        # 为不同文件类型配置加载器
        loaders = {
            ".txt": (TextLoader, {"encoding": "utf-8"}),
            ".pdf": (PDFMinerLoader, {}),
            ".docx": (Docx2txtLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {"encoding": "utf-8"}),
            ".csv": (CSVLoader, {"encoding": "utf-8"})
        }
        
        documents = []
        for ext, (loader_cls, loader_kwargs) in loaders.items():
            try:
                loader = DirectoryLoader(
                    self.docs_dir,
                    glob=f"**/*{ext}",
                    loader_cls=loader_cls,
                    loader_kwargs=loader_kwargs
                )
                documents.extend(loader.load())
            except Exception as e:
                print(f"加载 {ext} 文件时出错: {str(e)}")
        
        # 使用 RecursiveCharacterTextSplitter 进行文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        return texts
    
    def create_vector_store(self, texts: List) -> None:
        """创建向量存储"""
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
    
    def load_vector_store(self) -> None:
        """加载已存在的向量存储"""
        if os.path.exists(self.db_dir):
            self.vector_store = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
    
    async def similarity_search(self, query: str, k: int = 4) -> List:
        """执行异步相似性搜索"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        return await self.vector_store.asimilarity_search(query, k=k)

    async def get_relevant_context(self, question: str) -> str:
        """异步获取相关上下文"""
        docs = await self.similarity_search(question)
        return "\n".join([doc.page_content for doc in docs])

    def init_qa_chain(self) -> None:
        """初始化异步问答链"""
        if not self.vector_store:
            raise ValueError("请先初始化或加载向量存储")
        
        retriever = self.vector_store.as_retriever()
        
        self.qa_chain = {
            "context": retriever,
            "question": RunnablePassthrough()
        } | self.llm

    async def query(self, question: str) -> dict:
        """执行异步问答查询
        
        Returns:
            dict: 包含答案和源文档的字典
        """
        if not self.qa_chain:
            raise ValueError("问答链未初始化")
        
        result = await self.qa_chain.ainvoke({
            "query": question
        })
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
        
        # LCEL (LangChain Expression Language) 是 LangChain 提供的一种声明式的链式操作语法，它让我们能够更灵活地组合和构建 AI 应用程序的各个组件。