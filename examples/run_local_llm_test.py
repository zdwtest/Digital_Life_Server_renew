from tests.test_local_llm import TestLocalLLM

def main():
    # 指定本地模型路径
    model_path = "llama/llama-2-7b-chat.Q5_K_M.gguf"  # 替换为你的模型路径
    
    # 创建测试实例
    tester = TestLocalLLM(model_path)
    
    # 运行测试
    tester.run_all_tests()

if __name__ == "__main__":
    main() 