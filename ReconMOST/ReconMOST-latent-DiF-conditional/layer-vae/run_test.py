#!/usr/bin/env python3
"""
运行预训练VAE测试的脚本
使用方法：
    python run_test.py
    
可选参数：
    --data_dir: EN4数据目录路径
    --output_dir: 输出目录路径
    --num_samples: 测试样本数量
"""

import argparse
import sys
from config import VAEConfig
from test_pretrained_vae import PretrainedVAETester


def main():
    parser = argparse.ArgumentParser(description='Test pretrained VAE on ocean temperature data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to EN4 test data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    # 创建配置
    config = VAEConfig()
    
    # 覆盖配置参数
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_samples:
        config.num_test_samples = args.num_samples
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print("="*60)
    print("Pretrained VAE Test Configuration")
    print("="*60)
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of test samples: {config.num_test_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Model: {config.pretrained_model_name}")
    print(f"Input shape: {config.input_channels}x{config.height}x{config.width}")
    # print(f"Padded shape: {config.input_channels}x{config.padded_height}x{config.padded_width}")
    print("="*60)
    
    try:
        # 创建测试器并运行测试
        tester = PretrainedVAETester(config)
        results = tester.test()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()