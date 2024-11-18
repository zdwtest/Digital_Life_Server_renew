import os
from pathlib import Path
import logging
from tools.llama.generate import generate_text
from tools.vqgan.inference import inference

# 假设已经导入或定义了 generate_text 和 inference 函数
# from fishspeech import generate_text, inference

class TTService:
    def __init__(self, checkpoint_path, prompt_tokens_path, num_samples, config_name="firefly_gan_vq"):
        logging.info('Initializing TTS Service...')
        
        # 设置检查点路径和提示令牌路径
        self.checkpoint_path = checkpoint_path
        self.prompt_tokens_path = prompt_tokens_path
        self.config_name = config_name
        self.num_samples = num_samples
        # 创建输出目录
        self.output_dir = Path("tmp")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_audio(self, text):
        logging.info('Generating audio for text: %s', text)
        
        # 调用生成函数
        result = generate_text(
            text=text,
            num_samples=self.num_samples,
            prompt_text=["The text corresponding to reference audio"],
            prompt_tokens=[str(self.prompt_tokens_path)],
            checkpoint_path=self.checkpoint_path,
        )
        
        # 假设 result 包含生成的音频数据，具体根据实际情况调整
        # 这里可以返回 result 或进行其他处理

        return result

    def save_audio(self, output_file):
        logging.info('Saving audio to %s', output_file)
        
        # 调用推理函数
        inference(
            checkpoint_path=self.checkpoint_path / 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
            input_path=Path("tmp/codes_0.npy"),
            device="cuda",  # 如果没有 CUDA 设备，改为 "cpu"
            output_path=output_file,
            config_name=self.config_name,
        )

# 示例使用
if __name__ == "__main__":
    # 获取当前脚本的目录
    current_dir = os.path.dirname(__file__)

    # 构建相对路径
    relative_checkpoint_path = os.path.join(current_dir, 'fishspeech', 'checkpoints', 'fish-speech-1.4')
    relative_prompt_tokens = Path("tmp/fake.npy")

    # 创建 TTS 服务实例
    tts_service = TTService(
        checkpoint_path=relative_checkpoint_path,
        prompt_tokens_path=relative_prompt_tokens,
        num_samples=1,
    )

    # 生成音频
    text = "哈哈哈哈哈哈哈哈哈"
    audio_result = tts_service.generate_audio(text)

    # 保存音频
    tts_service.save_audio(output_file="tmp/output.wav")
