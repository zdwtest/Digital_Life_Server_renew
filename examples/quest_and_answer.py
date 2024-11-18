from LCS.LangchainService import LangchainService

# 初始化服务，使用已存在的向量库
service = LangchainService(
    llm_type="llama",
    model_config={
        "model_path": "llama/llama-2-7b-chat.Q5_K_M.gguf",  # 保持 model_path
        "temperature": 0.7,
        "max_tokens": 4096,
        "threads": 16,            # 使用 threads
        "n_ctx": 10000,  # 可以根据需要调整,但不要超过模型的最大支持长度
    },
    embedding_type="huggingface",
    db_dir="db",
    use_existing_db=True  # 确保使用现有向量库
)

# 进行查询时，添加提示让模型基于知识库回答
answer = service.query("基于已有的知识库资料，请简单介绍一下使用对象关系映射的优点,使用中文回答")
print(answer)
