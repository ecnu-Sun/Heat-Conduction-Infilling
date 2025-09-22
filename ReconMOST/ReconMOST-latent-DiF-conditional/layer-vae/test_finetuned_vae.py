# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import torch
# import torch.nn.functional as F
# import numpy as np
# from diffusers import AutoencoderKL
# from diffusers.image_processor import VaeImageProcessor
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from datetime import datetime

# from config import VAEConfig
# from dataset import create_dataloader
# from utils import visualize_reconstruction


# class FinetunedVAETester:
#     def __init__(self, config: VAEConfig, use_best_model: bool = True):
#         self.config = config
#         self.device = torch.device(config.device)
#         torch.manual_seed(config.seed)
        
#         # 检查微调模型目录
#         self.checkpoint_dir = os.path.join(config.output_dir, "finetuned_models")
#         if not os.path.exists(self.checkpoint_dir):
#             raise ValueError(f"Finetuned models directory not found: {self.checkpoint_dir}")
        
#         # 加载微调的VAE模型
#         self.vae = self._load_finetuned_vae(use_best_model)
        
#         # 初始化VaeImageProcessor
#         self.vae_processor = VaeImageProcessor(
#             vae_scale_factor=8,
#             do_resize=True,
#             do_normalize=True,
#             do_convert_rgb=False,
#             do_convert_grayscale=False
#         )
        
#         # 创建数据加载器
#         self.test_dataloader = create_dataloader(
#             config.data_dir,
#             batch_size=config.batch_size,
#             num_workers=config.num_workers,
#             shuffle=False,
#             temp_min=config.temp_min,
#             temp_range=config.temp_range
#         )
        
#         # 创建输出目录
#         self.output_dir = os.path.join(config.output_dir, "finetuned_test_results")
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         print(f"Initialized tester with device: {self.device}")
#         print(f"Test output directory: {self.output_dir}")
    
#     def _load_finetuned_vae(self, use_best_model: bool = True):
#         """
#         加载微调的VAE模型
#         """
#         # 选择加载best还是final模型
#         if use_best_model:
#             model_path = os.path.join(self.checkpoint_dir, "vae_best.pt")
#             if not os.path.exists(model_path):
#                 print("Best model not found, loading final model instead")
#                 model_path = os.path.join(self.checkpoint_dir, "vae_final.pt")
#         else:
#             model_path = os.path.join(self.checkpoint_dir, "vae_final.pt")
        
#         if not os.path.exists(model_path):
#             raise ValueError(f"No finetuned model found at {model_path}")
        
#         # 加载预训练VAE结构
#         vae = AutoencoderKL.from_pretrained(
#             self.config.pretrained_model_name,
#             subfolder="vae"
#         ).to(self.device)
        
#         # 加载微调的权重
#         checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
#         vae.load_state_dict(checkpoint['model_state_dict'])
#         vae.eval()
        
#         print(f"Loaded finetuned VAE from {model_path}")
#         print(f"Model trained for {checkpoint['epoch']} epochs with loss {checkpoint['loss']:.4f}")
        
#         return vae
    
#     def encode_decode_channels(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         使用微调的VAE模型对每个通道组进行编码解码
#         Args:
#             image: [B, 42, H, W] 输入图像（[0,1]范围）
#         Returns:
#             reconstructed: [B, 42, H, W] 重建图像（温度值）
#         """
#         print(f"Encoding and decoding image with shape {image.shape}...")
#         batch_size = image.shape[0]
#         original_height = image.shape[2]
#         original_width = image.shape[3]
#         reconstructed_channels = []
        
#         # 分组处理
#         for group_idx in range(self.config.num_channel_groups):
#             start_ch = group_idx * self.config.channels_per_group
#             end_ch = start_ch + self.config.channels_per_group
            
#             # 提取3个通道
#             group_channels = image[:, start_ch:end_ch, :, :]  # [B, 3, H, W] 已经在[0,1]范围内
            
#             # 使用processor预处理（直接传入[0,1]范围的数据）
#             group_channels_processed = self.vae_processor.preprocess(
#                 group_channels.cpu()
#             ).to(self.device)
            
#             # 编码解码
#             with torch.no_grad():
#                 posterior = self.vae.encode(group_channels_processed).latent_dist
#                 z = posterior.mean  # 使用均值确保确定性
#                 decoded = self.vae.decode(z).sample
            
#             # 使用processor后处理
#             decoded_denormalized = self.vae_processor.postprocess(
#                 decoded,
#                 output_type="pt",
#                 do_denormalize=[True] * batch_size
#             ).to(self.device)
            
#             # 将[0,1]范围转换回原始温度范围
#             decoded_temp = decoded_denormalized * self.config.temp_range + self.config.temp_min
            
#             # 调整回原始尺寸
#             decoded_resized = F.interpolate(
#                 decoded_temp, 
#                 size=(original_height, original_width),
#                 mode='bilinear',
#                 align_corners=False
#             )
            
#             reconstructed_channels.append(decoded_resized)
        
#         # 堆叠所有重建的通道
#         reconstructed = torch.cat(reconstructed_channels, dim=1)  # [B, 42, H, W]
        
#         return reconstructed
    
#     def compute_mse(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
#         """
#         计算MSE指标
#         """
#         # 将原始图像从[0,1]转换回温度值进行比较
#         original_temp = original * self.config.temp_range + self.config.temp_min
        
#         # 整体MSE
#         mse = F.mse_loss(reconstructed, original_temp, reduction='mean').item()
        
#         # 每个通道的MSE
#         channel_mse = []
#         for ch in range(self.config.input_channels):
#             ch_mse = F.mse_loss(
#                 reconstructed[:, ch, :, :], 
#                 original_temp[:, ch, :, :], 
#                 reduction='mean'
#             ).item()
#             channel_mse.append(ch_mse)
        
#         return {
#             'mse': mse,
#             'mse_temp': mse,
#             'channel_mse': channel_mse,
#             'rmse': np.sqrt(mse),
#             'rmse_temp': np.sqrt(mse)
#         }
    
#     def test(self):
#         """
#         执行测试
#         """
#         print("\nStarting test on EN4 dataset with finetuned VAE...")
#         print(f"Original image size: {self.config.height}x{self.config.width}")
        
#         all_mse = []
#         all_mse_temp = []
#         all_channel_mse = []
        
#         num_samples = min(self.config.num_test_samples, len(self.test_dataloader))
        
#         with torch.no_grad():
#             for i, (images, filepaths) in enumerate(tqdm(self.test_dataloader, desc="Testing")):
#                 if i >= num_samples:
#                     break
                
#                 images = images.to(self.device)
                
#                 # 编码解码
#                 reconstructed = self.encode_decode_channels(images)
                
#                 # 计算MSE
#                 metrics = self.compute_mse(images, reconstructed)
                
#                 all_mse.append(metrics['mse'])
#                 all_mse_temp.append(metrics['mse_temp'])
#                 all_channel_mse.append(metrics['channel_mse'])
                
#                 # 保存一些可视化结果
#                 if i < 10:  # 保存前10个样本的可视化
#                     original_temp = images[0].cpu() * self.config.temp_range + self.config.temp_min
#                     save_path = os.path.join(
#                         self.output_dir, 
#                         f'reconstruction_sample_{i+1}.png'
#                     )
#                     visualize_reconstruction(
#                         original_temp, 
#                         reconstructed[0].cpu(), 
#                         layer_idx=0,  # 可视化第一层
#                         save_path=save_path,
#                         vmin=self.config.temp_min,
#                         vmax=self.config.temp_max
#                     )
                
#                 # 打印单个样本结果（只在前几个样本时打印）
#                 if i < 5:
#                     print(f"\nSample {i+1}: {os.path.basename(filepaths[0])}")
#                     print(f"  MSE (temperature): {metrics['mse_temp']:.4f} °C²")
#                     print(f"  RMSE (temperature): {metrics['rmse_temp']:.4f} °C")
        
#         # 计算统计结果
#         all_channel_mse = np.array(all_channel_mse)  # [num_samples, 42]
        
#         results = {
#             'avg_mse': np.mean(all_mse),
#             'std_mse': np.std(all_mse),
#             'avg_mse_temp': np.mean(all_mse_temp),
#             'std_mse_temp': np.std(all_mse_temp),
#             'avg_rmse_temp': np.sqrt(np.mean(all_mse_temp)),
#             'channel_avg_mse': np.mean(all_channel_mse, axis=0),
#             'channel_std_mse': np.std(all_channel_mse, axis=0),
#             'all_mse': all_mse,
#             'all_mse_temp': all_mse_temp
#         }
        
#         # 打印总体结果
#         print("\n" + "="*60)
#         print("Test Results Summary (Finetuned VAE):")
#         print("="*60)
#         print(f"Average MSE (temperature): {results['avg_mse_temp']:.4f} ± {results['std_mse_temp']:.4f} °C²")
#         print(f"Average RMSE (temperature): {results['avg_rmse_temp']:.4f} °C")
#         print("="*60)
        
#         # 保存结果
#         self.save_results(results)
        
#         # 绘制通道MSE分布
#         self.plot_channel_mse(results)
        
#         return results
    
#     def save_results(self, results: dict):
#         """
#         保存测试结果
#         """
#         import json
        
#         # 保存数值结果
#         save_path = os.path.join(self.output_dir, 'test_results.json')
        
#         # 转换numpy数组为列表以便JSON序列化
#         json_results = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'config': {
#                 'model': self.config.pretrained_model_name + " (finetuned)",
#                 'num_samples': len(results['all_mse']),
#                 'input_shape': f"{self.config.input_channels}x{self.config.height}x{self.config.width}",
#                 'vae_scale_factor': 8,
#                 'note': 'Results from finetuned VAE model'
#             },
#             'results': {
#                 'avg_mse': float(results['avg_mse']),
#                 'std_mse': float(results['std_mse']),
#                 'avg_mse_temp': float(results['avg_mse_temp']),
#                 'std_mse_temp': float(results['std_mse_temp']),
#                 'avg_rmse_temp': float(results['avg_rmse_temp']),
#                 'channel_avg_mse': results['channel_avg_mse'].tolist(),
#                 'channel_std_mse': results['channel_std_mse'].tolist()
#             }
#         }
        
#         with open(save_path, 'w') as f:
#             json.dump(json_results, f, indent=2)
        
#         print(f"\nResults saved to {save_path}")
    
#     def plot_channel_mse(self, results: dict):
#         """
#         绘制每个通道的MSE分布
#         """
#         plt.figure(figsize=(12, 6))
        
#         channels = np.arange(self.config.input_channels)
#         avg_mse = results['channel_avg_mse']
#         std_mse = results['channel_std_mse']
        
#         plt.errorbar(channels, avg_mse, yerr=std_mse, fmt='o-', capsize=5)
#         plt.xlabel('Channel (Depth Layer)')
#         plt.ylabel('MSE (°C²)')
#         plt.title('MSE Distribution Across Depth Channels (Finetuned VAE)')
#         plt.grid(True, alpha=0.3)
        
#         save_path = os.path.join(self.output_dir, 'channel_mse_distribution.png')
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print(f"Channel MSE plot saved to {save_path}")


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Test finetuned VAE on ocean temperature data')
#     parser.add_argument('--data_dir', type=str, default=None,
#                         help='Path to test data directory')
#     parser.add_argument('--num_samples', type=int, default=None,
#                         help='Number of test samples')
#     parser.add_argument('--batch_size', type=int, default=None,
#                         help='Batch size for testing')
#     parser.add_argument('--use_final', action='store_true',
#                         help='Use final model instead of best model')
    
#     args = parser.parse_args()
    
#     # 创建配置
#     config = VAEConfig()
    
#     # 覆盖配置参数
#     if args.data_dir:
#         config.data_dir = args.data_dir
#     if args.num_samples:
#         config.num_test_samples = args.num_samples
#     if args.batch_size:
#         config.batch_size = args.batch_size
    
#     print("="*60)
#     print("Finetuned VAE Test Configuration")
#     print("="*60)
#     print(f"Data directory: {config.data_dir}")
#     print(f"Output directory: {config.output_dir}")
#     print(f"Number of test samples: {config.num_test_samples}")
#     print(f"Batch size: {config.batch_size}")
#     print(f"Using {'final' if args.use_final else 'best'} model")
#     print("="*60)
    
#     try:
#         # 创建测试器并运行测试
#         tester = FinetunedVAETester(config, use_best_model=not args.use_final)
#         results = tester.test()
        
#         print("\n✅ Test completed successfully!")
        
#     except Exception as e:
#         print(f"\n❌ Error during testing: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from config import VAEConfig
from dataset import create_dataloader
from utils import visualize_reconstruction


class FinetunedVAETester:
    def __init__(self, config: VAEConfig, use_model: str):
        self.config = config
        self.device = torch.device(config.device)
        self.use_model = use_model
        torch.manual_seed(config.seed)
        
        # 检查微调模型目录
        self.checkpoint_dir = os.path.join(config.output_dir, "finetuned_models")
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError(f"Finetuned models directory not found: {self.checkpoint_dir}")
        
        # 加载微调的VAE模型
        self.vae = self._load_finetuned_vae(use_model)
        
        # 初始化VaeImageProcessor
        self.vae_processor = VaeImageProcessor(
            # vae_scale_factor=8,
            do_resize=False,
            do_normalize=True,
            do_convert_rgb=False,
            do_convert_grayscale=False
        )
        
        # 创建数据加载器
        self.test_dataloader = create_dataloader(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            temp_min=config.temp_min,
            temp_range=config.temp_range
        )
        
        # 创建输出目录
        self.output_dir = os.path.join(config.output_dir, "finetuned_test_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Initialized tester with device: {self.device}")
        print(f"Test output directory: {self.output_dir}")
    
    def _load_finetuned_vae(self, use_model: str):
        """
        加载微调的VAE模型
        """
        # 选择加载best还是final模型
        model_path = os.path.join(self.checkpoint_dir, use_model)
        if not os.path.exists(model_path):
            print("model not found")
            raise ValueError(f"No finetuned model found at {model_path}")

        # 加载预训练VAE结构
        vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="vae"
        ).to(self.device)
        
        # 加载微调的权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.eval()
        
        print(f"Loaded finetuned VAE from {model_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs with loss {checkpoint['loss']:.4f}")
        
        return vae
    
    def encode_decode_channels(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用微调的VAE模型对每个通道组进行编码解码
        Args:
            image: [B, 42, H, W] 输入图像（[0,1]范围）
        Returns:
            original_processed: [B, 42, H, W] 处理后的原始图像（温度值）
            reconstructed: [B, 42, H, W] 重建图像（温度值）
        """
        print(f"Encoding and decoding image with shape {image.shape}...")
        batch_size = image.shape[0]
        
        original_processed_channels = []
        reconstructed_channels = []
        
        # 分组处理
        for group_idx in range(self.config.num_channel_groups):
            start_ch = group_idx * self.config.channels_per_group
            end_ch = start_ch + self.config.channels_per_group
            
            # 提取3个通道
            group_channels = image[:, start_ch:end_ch, :, :]  # [B, 3, H, W] 已经在[0,1]范围内
            # 计算padding并应用反射padding，填充顶部和右侧到8的倍数
            h, w = group_channels.shape[-2:]
            group_channels = F.pad(group_channels, (0, (8 - w % 8) % 8, (8 - h % 8) % 8, 0), mode='reflect')
            # 使用processor预处理（直接传入[0,1]范围的数据）
            group_channels_processed = self.vae_processor.preprocess(
                group_channels.cpu()
            ).to(self.device)
            
            # 保存处理后的原始图像并反归一化到温度值
            original_denormalized = self.vae_processor.postprocess(
                group_channels_processed,
                output_type="pt",
                do_denormalize=[True] * batch_size
            ).to(self.device)
            original_temp = original_denormalized * self.config.temp_range + self.config.temp_min
            original_temp = original_temp[:, :, -h:, :w]
            original_processed_channels.append(original_temp)
            
            # 编码解码
            with torch.no_grad():
                posterior = self.vae.encode(group_channels_processed).latent_dist
                z = posterior.mean  # 使用均值确保确定性
                decoded = self.vae.decode(z).sample
            
            # 使用processor后处理
            decoded_denormalized = self.vae_processor.postprocess(
                decoded,
                output_type="pt",
                do_denormalize=[True] * batch_size
            ).to(self.device)
            
            # 将[0,1]范围转换回原始温度范围
            decoded_temp = decoded_denormalized * self.config.temp_range + self.config.temp_min
            
            decoded_temp=decoded_temp[:, :, -h:, :w]
            reconstructed_channels.append(decoded_temp)
        
        # 堆叠所有通道
        original_processed = torch.cat(original_processed_channels, dim=1)  # [B, 42, H, W]
        reconstructed = torch.cat(reconstructed_channels, dim=1)  # [B, 42, H, W]
        
        return original_processed, reconstructed
    
    def compute_mse(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """
        计算MSE指标
        Args:
            original: 处理后的原始图像（温度值）
            reconstructed: 重建图像（温度值）
        """
        # 整体MSE（现在两个输入都已经是温度值了）
        mse = F.mse_loss(reconstructed, original, reduction='mean').item()
        
        # 每个通道的MSE
        channel_mse = []
        for ch in range(self.config.input_channels):
            ch_mse = F.mse_loss(
                reconstructed[:, ch, :, :], 
                original[:, ch, :, :], 
                reduction='mean'
            ).item()
            channel_mse.append(ch_mse)
        
        return {
            'mse': mse,
            'mse_temp': mse,
            'channel_mse': channel_mse,
            'rmse': np.sqrt(mse),
            'rmse_temp': np.sqrt(mse)
        }
    
    def test(self):
        """
        执行测试
        """
        print("\nStarting test on EN4 dataset with finetuned VAE...")
        print(f"Original image size: {self.config.height}x{self.config.width}")
        print(f"VAE processed size: {self.config.height//8}x{self.config.width//8}")
        
        all_mse = []
        all_mse_temp = []
        all_channel_mse = []
        
        num_samples = min(self.config.num_test_samples, len(self.test_dataloader))
        
        with torch.no_grad():
            for i, (images, filepaths) in enumerate(tqdm(self.test_dataloader, desc="Testing")):
                if i >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # 编码解码
                original_processed, reconstructed = self.encode_decode_channels(images)
                
                # 计算MSE
                metrics = self.compute_mse(original_processed, reconstructed)
                
                all_mse.append(metrics['mse'])
                all_mse_temp.append(metrics['mse_temp'])
                all_channel_mse.append(metrics['channel_mse'])
                
                # 保存一些可视化结果
                if i < 10:  # 保存前10个样本的可视化
                    save_path = os.path.join(
                        self.output_dir, 
                        f'reconstruction_sample_{i+1}.png'
                    )
                    visualize_reconstruction(
                        original_processed[0].cpu(), 
                        reconstructed[0].cpu(), 
                        layer_idx=0,  # 可视化第一层
                        save_path=save_path,
                        vmin=self.config.temp_min,
                        vmax=self.config.temp_max
                    )
                
                # 打印单个样本结果（只在前几个样本时打印）
                if i < 5:
                    print(f"\nSample {i+1}: {os.path.basename(filepaths[0])}")
                    print(f"  MSE (temperature): {metrics['mse_temp']:.4f} °C²")
                    print(f"  RMSE (temperature): {metrics['rmse_temp']:.4f} °C")
        
        # 计算统计结果
        all_channel_mse = np.array(all_channel_mse)  # [num_samples, 42]
        
        results = {
            'avg_mse': np.mean(all_mse),
            'std_mse': np.std(all_mse),
            'avg_mse_temp': np.mean(all_mse_temp),
            'std_mse_temp': np.std(all_mse_temp),
            'avg_rmse_temp': np.sqrt(np.mean(all_mse_temp)),
            'channel_avg_mse': np.mean(all_channel_mse, axis=0),
            'channel_std_mse': np.std(all_channel_mse, axis=0),
            'all_mse': all_mse,
            'all_mse_temp': all_mse_temp
        }
        
        # 打印总体结果
        print("\n" + "="*60)
        print("Test Results Summary (Finetuned VAE):")
        print("="*60)
        print(f"Average MSE (temperature): {results['avg_mse_temp']:.4f} ± {results['std_mse_temp']:.4f} °C²")
        print(f"Average RMSE (temperature): {results['avg_rmse_temp']:.4f} °C")
        print("="*60)
        
        # 保存结果
        self.save_results(results)
        
        # 绘制通道MSE分布
        self.plot_channel_mse(results)
        
        return results
    
    def save_results(self, results: dict):
        """
        保存测试结果
        """
        import json
        
        # 保存数值结果
        save_path = os.path.join(self.output_dir, 'test_results.json')
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'model': self.config.pretrained_model_name + " (finetuned)",
                'num_samples': len(results['all_mse']),
                'input_shape': f"{self.config.input_channels}x{self.config.height}x{self.config.width}",
                'vae_scale_factor': 8,
                'processed_shape': f"{self.config.input_channels}x{self.config.height//8}x{self.config.width//8}",
                'note': 'Results from finetuned VAE model (comparing at VAE resolution)'
            },
            'results': {
                'avg_mse': float(results['avg_mse']),
                'std_mse': float(results['std_mse']),
                'avg_mse_temp': float(results['avg_mse_temp']),
                'std_mse_temp': float(results['std_mse_temp']),
                'avg_rmse_temp': float(results['avg_rmse_temp']),
                'channel_avg_mse': results['channel_avg_mse'].tolist(),
                'channel_std_mse': results['channel_std_mse'].tolist()
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {save_path}")
    
    def plot_channel_mse(self, results: dict):
        """
        绘制每个通道的MSE分布
        """
        plt.figure(figsize=(12, 6))
        
        channels = np.arange(self.config.input_channels)
        avg_mse = results['channel_avg_mse']
        std_mse = results['channel_std_mse']
        
        plt.errorbar(channels, avg_mse, yerr=std_mse, fmt='o-', capsize=5)
        plt.xlabel('Channel (Depth Layer)')
        plt.ylabel('MSE (°C²)')
        plt.title('MSE Distribution Across Depth Channels (Finetuned VAE)')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, 'channel_mse_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Channel MSE plot saved to {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test finetuned VAE on ocean temperature data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to test data directory')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for testing')
    parser.add_argument('--use_model', type=str, default='vae_epoch_2.pt',
                        help='Use which vae model ')
    
    args = parser.parse_args()
    
    # 创建配置
    config = VAEConfig()
    
    # 覆盖配置参数
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.num_samples:
        config.num_test_samples = args.num_samples
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print("="*60)
    print("Finetuned VAE Test Configuration")
    print("="*60)
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of test samples: {config.num_test_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Using {args.use_model} model")
    print("="*60)
    
    try:
        # 创建测试器并运行测试
        tester = FinetunedVAETester(config, use_model=args.use_model)
        results = tester.test()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()