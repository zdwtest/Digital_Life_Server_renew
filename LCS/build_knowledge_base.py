from LCS.LangchainService import LangchainService

def build_knowledge_base(
    docs_dir: str = "docs",
    db_dir: str = "db",
    embedding_type: str = "huggingface"
):
    """
    构建知识库
    
    Args:
        docs_dir: 文档目录
        db_dir: 向量数据库存储目录
        embedding_type: 嵌入模型类型
    """
    # 仅初始化 embedding 模型,不初始化 LLM
    service = LangchainService(
        llm_type="llama",  # 这里的llm_type可以随意指定,因为不会被使用
        model_config={
                "model_path": "llama/llama-2-7b-chat.Q5_K_M.gguf",  # 替换为你的模型路径
                "temperature": 0.7,
                "max_tokens": 512,
                "n_ctx": 2048,  # 上下文窗口大小
                "n_threads": 4   # CPU线程数
            },
        embedding_type=embedding_type,
        docs_dir=docs_dir,
        db_dir=db_dir
    )
    
    print("开始加载文档...")
    texts = service.load_documents()
    print(f"成功加载 {len(texts)} 个文档片段")
    
    print("开始构建向量存储...")
    service.create_vector_store(texts)
    print(f"向量知识库已保存到: {db_dir}")

if __name__ == "__main__":
    build_knowledge_base() 