import unittest
import os
import shutil
from unittest.mock import Mock, patch
from LCS.LangchainService import LangchainService

class TestLangchainService(unittest.TestCase):
    def setUp(self):
        """测试前准备工作"""
        self.test_docs_dir = "test_docs"
        self.test_db_dir = "test_db"
        
        # 创建测试文档目录
        os.makedirs(self.test_docs_dir, exist_ok=True)
        
        # 创建测试文档
        self._create_test_documents()
        
    def tearDown(self):
        """测试后清理工作"""
        # 删除测试目录
        if os.path.exists(self.test_docs_dir):
            shutil.rmtree(self.test_docs_dir)
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
            
    def _create_test_documents(self):
        """创建测试文档"""
        test_content = """
        Python是一种广泛使用的解释型、高级和通用的编程语言。
        Python的设计哲学强调代码的可读性和简洁的语法。
        """
        
        with open(f"{self.test_docs_dir}/test1.txt", "w", encoding="utf-8") as f:
            f.write(test_content)
            
    def test_init_openai(self):
        """测试OpenAI初始化"""
        service = LangchainService(
            llm_type="openai",
            model_config={"temperature": 0.7},
            docs_dir=self.test_docs_dir,
            db_dir=self.test_db_dir
        )
        self.assertIsNotNone(service.llm)
        
    def test_init_llama(self):
        """测试Llama初始化"""
        with self.assertRaises(ValueError):
            # 没有提供模型路径应该抛出错误
            LangchainService(
                llm_type="llama",
                docs_dir=self.test_docs_dir,
                db_dir=self.test_db_dir
            )
            
    def test_load_documents(self):
        """测试文档加载"""
        service = LangchainService(
            llm_type="openai",
            docs_dir=self.test_docs_dir,
            db_dir=self.test_db_dir
        )
        docs = service.load_documents()
        self.assertTrue(len(docs) > 0)
        
    @patch('LLMS.LangchainService.Chroma')
    def test_vector_store(self, mock_chroma):
        """测试向量存储"""
        # 模拟Chroma
        mock_chroma.from_documents.return_value = Mock()
        
        service = LangchainService(
            llm_type="openai",
            docs_dir=self.test_docs_dir,
            db_dir=self.test_db_dir
        )
        
        docs = service.load_documents()
        service.create_vector_store(docs)
        
        # 验证是否调用了向量存储
        mock_chroma.from_documents.assert_called_once()
        
    @patch('LLMS.LangchainService.RetrievalQA')
    def test_query(self, mock_qa):
        """测试查询功能"""
        # 模拟问答链
        mock_qa.from_chain_type.return_value = Mock()
        mock_qa.from_chain_type.return_value.run.return_value = "测试回答"
        
        service = LangchainService(
            llm_type="openai",
            docs_dir=self.test_docs_dir,
            db_dir=self.test_db_dir
        )
        
        # 模拟向量存储
        service.vector_store = Mock()
        service.vector_store.as_retriever.return_value = Mock()
        
        answer = service.query("什么是Python？")
        self.assertEqual(answer, "测试回答") 