import os
import torch
from dataclasses import dataclass


@dataclass
class VAEConfig:
    # 数据路径 - 使用EN4数据集进行测试
    data_dir: str = "/mnt/nas/SZLbackup/ReconMOST-testdata-alignwith-FIO"
    # data_dir: str = "../../../data/ReconMOST_train_processed_en4grid/fgoals-f3-l-align"
    output_dir: str = "./outputs/vae-pretrained-test-old"
    train_data_dir: str = "/hdd/SunZL/data/nearest_processed_FIO"
    
    # 数据参数
    input_channels: int = 42  # 海温层数
    height: int = 173
    width: int = 360
    
    # Padding参数 - 确保尺寸是64的倍数
    padded_height: int = 192  # 最接近173的64的倍数
    padded_width: int = 384   # 最接近360的64的倍数
    
    # 归一化参数
    temp_min: float = -5.0
    temp_max: float = 40.0
    temp_range: float = 45.0
    
    # 模型参数
    pretrained_model_name: str = "stabilityai/stable-diffusion-2-1"
    # pretrained_model_name: str = "stabilityai/sd-vae-ft-mse"
    num_channel_groups: int = 14  # 将42通道分成14组，每组3通道
    channels_per_group: int = 3
    
    # 测试参数
    batch_size: int = 1
    num_test_samples: int = 100  # 测试样本数量
    
    # 其他参数
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)