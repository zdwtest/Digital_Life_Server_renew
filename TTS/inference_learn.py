import os
from pathlib import Path

from tools.vqgan.inference import inference

output_dir = Path("tmp")
output_dir.mkdir(parents=True, exist_ok=True)

# 获取当前脚本的目录
current_dir = os.path.dirname(__file__)

relative_checkpoint_path = os.path.join(current_dir, 'fishspeech', 'checkpoints', 'fish-speech-1.4', 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth' )

inference(
    checkpoint_path=relative_checkpoint_path,
    input_path=Path("F:/node/Digital_Life_Server_renew/TTS/fishspeech/custom/丁真.wav"),
    device="cuda",
    output_path=output_dir / "fake.wav",  # 确保路径存在并正确
    config_name="firefly_gan_vq",
)