from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader, Docx2txtLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp, OpenAI
from langchain_community.chat_models import ChatOpenAI
from typing import Optional, List, Union
import os
from langchain.chains.question_answering import load_qa_chain

class LangchainService:
    def __init__(self, 
                 llm_type: str = "openai",
                 model_config: dict = None,
                 embedding_type: str = "openai",
                 docs_dir: str = "docs",
                 db_dir: str = "db",
                 use_existing_db: bool = False):
        """
        初始化LangChain服务
        
        Args:
            llm_type: LLM类型 ("openai", "llama", "chatgpt")
            model_config: 模型配置参数
            embedding_type: 嵌入模型类型 ("openai", "huggingface")
            docs_dir: 知识库文档目录
            db_dir: 向量数据库存储目录
            use_existing_db: 是否使用已存在的向量库
        """
        self.llm = self._init_llm(llm_type, model_config or {})
        self.embeddings = self._init_embeddings(embedding_type)
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.vector_store = None
        
        if use_existing_db:
            self.load_vector_store()
        
        # 添加 qa_chain 的初始化
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")
    
    def _init_llm(self, llm_type: str, config: dict) -> Union[OpenAI, LlamaCpp, ChatOpenAI]:
        """初始化LLM模型"""
        if llm_type == "openai":
            return OpenAI(**config)
        elif llm_type == "chatgpt":
            return ChatOpenAI(**config)
        elif llm_type == "llama":
            # 确保设置合适的context_window大小
            default_config = {
                "n_ctx": 4096,  # 增加上下文窗口大小
                "temperature": 0.7,
                "max_tokens": 2048,
                "n_batch": 512,
            }
            # 合并用户配置和默认配置
            default_config.update(config)
            return LlamaCpp(**default_config)
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
            
    def _init_embeddings(self, embedding_type: str):
        """初始化嵌入模型"""
        if embedding_type == "openai":
            return OpenAIEmbeddings()
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"  # 显式指定模型名称
            )
        else:
            raise ValueError(f"不支持��嵌入模型类型: {embedding_type}")
    
    def load_documents(self, file_pattern: str = "**/*.*") -> List:
        """加载多种格式的文档
        
        支持的格式:
        - .txt (文本文件)
        - .pdf (PDF文件)
        - .docx (Word文档)
        - .md (Markdown文件)
        - .csv (CSV文件)
        """
        # 为不同文件类型配置加载器
        loaders = {
            ".txt": (TextLoader, {"encoding": "utf-8"}),
            ".pdf": (PDFMinerLoader, {}),
            ".docx": (Docx2txtLoader, {}),  # 移除 encoding 参数
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
        
        # 使用 RecursiveCharacterTextSplitter 进行更智能的文本分割
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
        # 移除 persist() 调用，因为新版本的 Chroma 会自动持久化
    
    def load_vector_store(self) -> None:
        """加载已存在的向量存储"""
        if os.path.exists(self.db_dir):
            self.vector_store = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
    
    def query(self, question: str) -> str:
        """查询知识库并返回答案"""
        # 确保使用向量存储进行相似性搜索
        docs = self.vector_store.similarity_search(question, k=4)
        
        # 构建提示，强调使用检索到的文档内容
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下参考文档内容回答问题。如果无法从参考文档中找到相关信息，请说明无法回答。

参考文档：
{context}

问题：{question}

回答："""
        
        # 使用 LLM 生成答案
        return self.qa_chain.run(input_documents=docs, question=prompt)
      
      
#       # 使用OpenAI API
# service = LangchainService(
#     llm_type="openai",
#     model_config={
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo"  # 或其他OpenAI模型
#     }
# )

# # 使用本地Llama模型
# service = LangchainService(
#     llm_type="llama",
#     model_config={
#         "model_path": "/path/to/model.gguf",
#         "temperature": 0.7,
#         "max_tokens": 512
#     }
# )

# # 使用ChatGPT
# service = LangchainService(
#     llm_type="chatgpt",
#     model_config={
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo"
#     }
# )

# # 加载文档并创建向量存储
# texts = service.load_documents()
# service.create_vector_store(texts)

# # 查询
# answer = service.query("你的问题")
# print(answer)

# 1. 首先运行构建知识库脚本
# python tools/build_knowledge_base.py

# 2. 然后在应用中使用已构建好的知识库
# from LCS.LangchainService import LangchainService

# # 初始化服务,使用已存在的向量库
# service = LangchainService(
#     llm_type="openai",
#     model_config={
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo"
#     },
#     embedding_type="openai",
#     db_dir="db",  # 指定向量库目录
#     use_existing_db=True  # 使用已存在的向量库
# )

# # 直接查询,无需重新加载文档
# answer = service.query("你的问题")
# print(answer)