import unittest
import os
import shutil
from LCS.LangchainService import LangchainService
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLocalLLM:
    def __init__(self, model_path: str):
        """
        初始化测试类
        
        Args:
            model_path: 本地模型路径
        """
        self.model_path = model_path
        self.test_docs_dir = "test_docs"
        self.test_db_dir = "test_vector_db"
        self.service = None
        
    def setup(self):
        """准备测试环境"""
        logger.info("开始设置测试环境...")
        
        # 创建测试文档目录
        os.makedirs(self.test_docs_dir, exist_ok=True)
        
        # 创建测试文档
        self._create_test_documents()
        
        # 初始化LangChain服务
        self.service = LangchainService(
            llm_type="llama",
            model_config={
                "model_path": self.model_path,
                "temperature": 0.7,
                "max_tokens": 512,
                "n_ctx": 2048,  # 上下文窗口大小
                "n_threads": 4   # CPU线程数
            },
            embedding_type="huggingface",  # 使用HuggingFace的嵌入模型
            docs_dir=self.test_docs_dir,
            db_dir=self.test_db_dir
        )
        
        logger.info("测试环境设置完成")
        
    def _create_test_documents(self):
        """创建测试文档"""
        test_docs = {
            "python_intro.txt": """
            Python是一种面向对象的高级编程语言。
            Python的设计哲学强调代码的可读性和简洁的语法。
            它支持多种编程范式，包括面向对象、命令式和函数式编程。
            """,
            
            "python_features.txt": """
            Python的主要特点包括：
            1. 简单易学的语法
            2. 丰富的标准库
            3. 跨平台兼容性
            4. 大量的第三方库支持
            5. 活跃的社区支持
            """
        }
        
        for filename, content in test_docs.items():
            with open(f"{self.test_docs_dir}/{filename}", "w", encoding="utf-8") as f:
                f.write(content)
        
        logger.info(f"创建了 {len(test_docs)} 个测试文档")
        
    def cleanup(self):
        """清理测试环境"""
        logger.info("开始清理测试环境...")
        
        # 先关闭向量存储的数据库连接
        if self.service and self.service.vector_store:
            self.service.vector_store._client.close()
        
        # 删除测试目录，添加重试机制
        max_retries = 3
        for dir_path in [self.test_docs_dir, self.test_db_dir]:
            if os.path.exists(dir_path):
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(dir_path)
                        break
                    except PermissionError:
                        if attempt == max_retries - 1:
                            logger.error(f"无法删除目录 {dir_path}，已达到最大重试次数")
                            raise
                        time.sleep(1)  # 等待1秒后重试
            
        logger.info("测试环境清理完成")
        
    def test_document_loading(self):
        """测试文档加载"""
        try:
            docs = self.service.load_documents()
            logger.info(f"成功加载了 {len(docs)} 个文档片段")
            return docs
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
            
    def test_vector_store_creation(self, docs):
        """测试向量存储创建"""
        try:
            self.service.create_vector_store(docs)
            logger.info("向量存储创建成功")
        except Exception as e:
            logger.error(f"向量存储创建失败: {e}")
            raise
            
    def test_query(self):
        """测试查询功能"""
        test_questions = [
            "Python是什么样的编程语言？",
            "Python有哪些主要特点？",
            "为什么选择Python作为编程语言？"
        ]
        
        for question in test_questions:
            try:
                logger.info(f"\n问题: {question}")
                answer = self.service.query(question)
                logger.info(f"回答: {answer}\n")
            except Exception as e:
                logger.error(f"查询失败: {e}")
                raise
                
    def run_all_tests(self):
        """运行所有测试"""
        try:
            logger.info("开始运行测试...")
            
            # 设置测试环境
            self.setup()
            
            # 测试文档加载
            docs = self.test_document_loading()
            
            # 测试向量存储创建
            self.test_vector_store_creation(docs)
            
            # 测试查询
            self.test_query()
            
            logger.info("所有测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中出现错误: {e}")
            raise
            
        finally:
            self.cleanup() 