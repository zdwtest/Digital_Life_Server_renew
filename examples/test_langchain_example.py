import os
from LCS.LangchainService import LangchainService

def test_openai_service():
    """测试OpenAI服务"""
    # 设置OpenAI API密钥
    os.environ["OPENAI_API_KEY"] = "你的API密钥"
    
    # 初始化服务
    service = LangchainService(
        llm_type="openai",
        model_config={
            "temperature": 0.7,
            "model_name": "gpt-3.5-turbo"
        },
        docs_dir="example_docs"
    )
    
    # 准备测试文档
    os.makedirs("example_docs", exist_ok=True)
    with open("example_docs/python_intro.txt", "w", encoding="utf-8") as f:
        f.write("""
        Python是一种面向对象的高级编程语言。
        它的语法简单易学，具有丰富的标准库。
        Python支持多种编程范式，包括面向对象、命令式和函数式编程。
        """)
    
    try:
        # 加载文档
        texts = service.load_documents()
        print(f"加载了 {len(texts)} 个文档片段")
        
        # 创建向量存储
        service.create_vector_store(texts)
        print("创建向量存储成功")
        
        # 测试查询
        question = "Python有什么特点？"
        answer = service.query(question)
        print(f"问题: {question}")
        print(f"回答: {answer}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists("example_docs"):
            shutil.rmtree("example_docs")

if __name__ == "__main__":
    test_openai_service() 