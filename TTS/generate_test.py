import os
from pathlib import Path
from tools.llama.generate import generate_text
import time

# 获取当前脚本的目录
current_dir = os.path.dirname(__file__)

# 构建相对路径
relative_checkpoint_path = os.path.join(current_dir, 'fishspeech', 'checkpoints', 'fish-speech-1.4' )
relative_prompt_tokens = Path("tmp/fake.npy")

# 测试参数设置
text = "哈哈哈哈哈哈哈哈哈"
num_samples = 1  # 设置为多个样本，以便重复测试
num_iterations = 1  # 设定测试迭代次数

if __name__ == "__main__":
    for i in range(num_iterations):
        print(f"\nIteration {i + 1}/{num_iterations}:")
        
        # 调用生成函数
        result = generate_text(
            text=text,
            num_samples=num_samples,
            prompt_text=["The text corresponding to reference audio"],
            prompt_tokens=[str(relative_prompt_tokens)],  # 将路径作为列表传递
            checkpoint_path=relative_checkpoint_path,
        )

        # 打印生成的结果
        for idx, generated in enumerate(result):
            print(f"Sample {idx + 1}: {generated}")

        # 等待一段时间再进行下一次迭代
        time.sleep(2)  # 你可以根据需要调整这个延迟时间