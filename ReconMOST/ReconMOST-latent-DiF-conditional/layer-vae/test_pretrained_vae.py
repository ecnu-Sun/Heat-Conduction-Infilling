# import torch
# import torch.nn.functional as F
# import numpy as np
# from diffusers import AutoencoderKL
# from tqdm import tqdm
# import os
# import matplotlib.pyplot as plt
# from datetime import datetime

# from config import VAEConfig
# from dataset import create_dataloader
# from utils import denormalize_temperature, normalize_temperature, visualize_reconstruction
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# class PretrainedVAETester:
#     def __init__(self, config: VAEConfig):
#         self.config = config
        
#         # 设置设备
#         self.device = torch.device(config.device)
#         torch.manual_seed(config.seed)
        
#         # 加载预训练VAE
#         print(f"Loading pretrained VAE from {config.pretrained_model_name}...")
#         self.vae = AutoencoderKL.from_pretrained(
#             config.pretrained_model_name,
#             subfolder="vae"
#         ).to(self.device)
#         self.vae.eval()  # 设置为评估模式
        
#         # 创建数据加载器
#         self.test_dataloader = create_dataloader(
#             config.data_dir,
#             batch_size=config.batch_size,
#             num_workers=config.num_workers,
#             shuffle=False,
#             temp_min=config.temp_min,
#             temp_range=config.temp_range
#         )
        
#         print(f"Initialized tester with device: {self.device}")
    
#     def pad_image(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         将图像padding到64的倍数
#         Args:
#             image: [B, C, H, W] 输入图像
#         Returns:
#             padded_image: [B, C, padded_H, padded_W]
#         """
#         _, _, h, w = image.shape
#         pad_h = self.config.padded_height - h
#         pad_w = self.config.padded_width - w
        
#         # 计算padding (left, right, top, bottom)
#         pad_left = pad_w // 2
#         pad_right = pad_w - pad_left
#         pad_top = pad_h // 2
#         pad_bottom = pad_h - pad_top
        
#         # 使用反射padding
#         padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
#         return padded
    
#     def unpad_image(self, padded_image: torch.Tensor) -> torch.Tensor:
#         """
#         移除padding恢复原始尺寸
#         """
#         _, _, h_padded, w_padded = padded_image.shape
#         h, w = self.config.height, self.config.width
        
#         pad_h = h_padded - h
#         pad_w = w_padded - w
        
#         pad_top = pad_h // 2
#         pad_left = pad_w // 2
        
#         # 提取原始区域
#         unpadded = padded_image[:, :, pad_top:pad_top+h, pad_left:pad_left+w]
#         return unpadded
    
#     def encode_decode_channels(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         将42通道图像拆分成14个3通道图像分别编码解码
#         Args:
#             image: [B, 42, H, W] 输入图像（已归一化到[-1, 1]）
#         Returns:
#             reconstructed: [B, 42, H, W] 重建图像
#         """
#         batch_size = image.shape[0]
#         reconstructed_channels = []
        
#         # Padding到64的倍数
#         padded_image = self.pad_image(image)
        
#         # 分组处理
#         for group_idx in range(self.config.num_channel_groups):
#             start_ch = group_idx * self.config.channels_per_group
#             end_ch = start_ch + self.config.channels_per_group
            
#             # 提取3个通道
#             group_channels = padded_image[:, start_ch:end_ch, :, :]  # [B, 3, H_pad, W_pad]
            
#             # 编码
#             with torch.no_grad():
#                 posterior = self.vae.encode(group_channels).latent_dist
#                 z = posterior.mean  # 使用均值而不是采样，确保确定性
                
#                 # 解码
#                 decoded = self.vae.decode(z).sample
            
#             # 移除padding
#             decoded_unpadded = self.unpad_image(decoded)
#             reconstructed_channels.append(decoded_unpadded)
        
#         # 堆叠所有重建的通道
#         reconstructed = torch.cat(reconstructed_channels, dim=1)  # [B, 42, H, W]
        
#         return reconstructed
    
#     def compute_mse(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
#         """
#         计算MSE指标
#         """
#         # 整体MSE
#         mse = F.mse_loss(reconstructed, original, reduction='mean').item()
        
#         # 每个通道的MSE
#         channel_mse = []
#         for ch in range(self.config.input_channels):
#             ch_mse = F.mse_loss(
#                 reconstructed[:, ch, :, :], 
#                 original[:, ch, :, :], 
#                 reduction='mean'
#             ).item()
#             channel_mse.append(ch_mse)
        
#         # 计算反归一化后的MSE（实际温度单位）
#         original_temp = denormalize_temperature(original, self.config.temp_min, self.config.temp_range)
#         reconstructed_temp = denormalize_temperature(reconstructed, self.config.temp_min, self.config.temp_range)
#         mse_temp = F.mse_loss(reconstructed_temp, original_temp, reduction='mean').item()
        
#         return {
#             'mse': mse,
#             'mse_temp': mse_temp,
#             'channel_mse': channel_mse,
#             'rmse': np.sqrt(mse),
#             'rmse_temp': np.sqrt(mse_temp)
#         }
    
#     def test(self):
#         """
#         执行测试
#         """
#         print("Starting test on EN4 dataset...")
        
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
#                     save_path = os.path.join(
#                         self.config.output_dir, 
#                         f'reconstruction_sample_{i+1}.png'
#                     )
#                     # 反归一化后可视化
#                     orig_temp = denormalize_temperature(
#                         images[0].cpu(), 
#                         self.config.temp_min, 
#                         self.config.temp_range
#                     )
#                     recon_temp = denormalize_temperature(
#                         reconstructed[0].cpu(), 
#                         self.config.temp_min, 
#                         self.config.temp_range
#                     )
#                     visualize_reconstruction(
#                         orig_temp, 
#                         recon_temp, 
#                         layer_idx=0,  # 可视化第一层
#                         save_path=save_path,
#                         vmin=self.config.temp_min,
#                         vmax=self.config.temp_max
#                     )
                
#                 # 打印单个样本结果
#                 print(f"\nSample {i+1}: {os.path.basename(filepaths[0])}")
#                 print(f"  MSE (normalized): {metrics['mse']:.6f}")
#                 print(f"  MSE (temperature): {metrics['mse_temp']:.4f} °C²")
#                 print(f"  RMSE (temperature): {metrics['rmse_temp']:.4f} °C")
        
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
#         print("Test Results Summary:")
#         print("="*60)
#         print(f"Average MSE (normalized): {results['avg_mse']:.6f} ± {results['std_mse']:.6f}")
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
#         save_path = os.path.join(self.config.output_dir, 'test_results.json')
        
#         # 转换numpy数组为列表以便JSON序列化
#         json_results = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'config': {
#                 'model': self.config.pretrained_model_name,
#                 'num_samples': len(results['all_mse']),
#                 'input_shape': f"{self.config.input_channels}x{self.config.height}x{self.config.width}",
#                 'padded_shape': f"{self.config.input_channels}x{self.config.padded_height}x{self.config.padded_width}"
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
#         plt.ylabel('MSE (normalized)')
#         plt.title('MSE Distribution Across Depth Channels')
#         plt.grid(True, alpha=0.3)
        
#         save_path = os.path.join(self.config.output_dir, 'channel_mse_distribution.png')
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print(f"Channel MSE plot saved to {save_path}")


# def main():
#     config = VAEConfig()
#     tester = PretrainedVAETester(config)
#     results = tester.test()


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
import os
import matplotlib.pyplot as plt
from datetime import datetime

from config import VAEConfig
from dataset import create_dataloader
from utils import visualize_reconstruction



class PretrainedVAETester:
    def __init__(self, config: VAEConfig):
        self.config = config
        
        # 设置设备
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        
        # 加载预训练VAE
        print(f"Loading pretrained VAE from {config.pretrained_model_name}...")
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name,
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()  # 设置为评估模式
        
        # 初始化VaeImageProcessor
        self.vae_processor = VaeImageProcessor(
            vae_scale_factor=8,  # VAE的缩放因子
            do_resize=True,  # 会缩放到8的倍数（173x360 -> 168x360）
            do_normalize=True,  # 使用processor的归一化
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
        
        print(f"Initialized tester with device: {self.device}")
        print(f"Note: Images will be resized from {config.height}x{config.width} to nearest multiple of 8")
    
    def encode_decode_channels(self, image: torch.Tensor) -> torch.Tensor:
        """
        将42通道图像拆分成14个3通道图像分别编码解码
        Args:
            image: [B, 42, H, W] 输入图像（原始温度值）
        Returns:
            reconstructed: [B, 42, H, W] 重建图像
        """
        print(f"Processing image with shape: {image.shape}")
        batch_size = image.shape[0]
        original_height = image.shape[2]
        original_width = image.shape[3]
        reconstructed_channels = []
        
        # 分组处理
        for group_idx in range(self.config.num_channel_groups):
            start_ch = group_idx * self.config.channels_per_group
            end_ch = start_ch + self.config.channels_per_group
            
            # 提取3个通道
            group_channels = image[:, start_ch:end_ch, :, :]  # [B, 3, H, W]
            
            # 使用processor预处理
            # 注意：这会将 173x360 缩放到 168x360
            group_channels_processed = self.vae_processor.preprocess(
                group_channels.cpu()
            ).to(self.device)
            
            # 记录处理后的尺寸（第一次迭代时）
            if group_idx == 0:
                processed_height = group_channels_processed.shape[2]
                processed_width = group_channels_processed.shape[3]
                print(f"Processor resized images from {original_height}x{original_width} to {processed_height}x{processed_width}")
            
            # 编码
            with torch.no_grad():
                posterior = self.vae.encode(group_channels_processed).latent_dist
                z = posterior.mean  # 使用均值而不是采样，确保确定性
                
                # 解码
                decoded = self.vae.decode(z).sample
            
            # 使用processor后处理（反归一化到[0,1]）
            decoded_denormalized = self.vae_processor.postprocess(
                decoded,
                output_type="pt",
                do_denormalize=[True] * batch_size
            ).to(self.device)
            
            # 将[0,1]范围转换回原始温度范围
            decoded_temp = decoded_denormalized * self.config.temp_range + self.config.temp_min
            
            # 调整回原始尺寸
            decoded_resized = F.interpolate(
                decoded_temp, 
                size=(original_height, original_width),
                mode='bilinear',
                align_corners=False
            )
            
            reconstructed_channels.append(decoded_resized)
        
        # 堆叠所有重建的通道
        reconstructed = torch.cat(reconstructed_channels, dim=1)  # [B, 42, H, W]
        
        return reconstructed
    
    # def encode_decode_channels(self, image: torch.Tensor) -> torch.Tensor:
    #     """
    #     将42通道图像拆分成42个单通道图像，每个复制成3通道后分别编码解码
    #     Args:
    #         image: [B, 42, H, W] 输入图像（原始温度值）
    #     Returns:
    #         reconstructed: [B, 42, H, W] 重建图像
    #     """
    #     batch_size = image.shape[0]
    #     original_height = image.shape[2]
    #     original_width = image.shape[3]
    #     reconstructed_channels = []
        
    #     # 分组处理 - 现在是42组，每组1个通道
    #     for channel_idx in range(self.config.input_channels):  # 42个通道
    #         # 提取单个通道
    #         single_channel = image[:, channel_idx:channel_idx+1, :, :]  # [B, 1, H, W]
            
    #         # 将单通道复制成3通道
    #         three_channel = single_channel.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            
    #         # 使用processor预处理
    #         # 注意：这会将 173x360 缩放到 168x360
    #         three_channel_processed = self.vae_processor.preprocess(
    #             three_channel.cpu()
    #         ).to(self.device)
            
    #         # 记录处理后的尺寸（第一次迭代时）
    #         if channel_idx == 0:
    #             processed_height = three_channel_processed.shape[2]
    #             processed_width = three_channel_processed.shape[3]
    #             print(f"Processor resized images from {original_height}x{original_width} to {processed_height}x{processed_width}")
            
    #         # 编码
    #         with torch.no_grad():
    #             posterior = self.vae.encode(three_channel_processed).latent_dist
    #             z = posterior.mean  # 使用均值而不是采样，确保确定性
                
    #             # 解码
    #             decoded = self.vae.decode(z).sample
            
    #         # 使用processor后处理（反归一化到[0,1]）
    #         decoded_denormalized = self.vae_processor.postprocess(
    #             decoded,
    #             output_type="pt",
    #             do_denormalize=[True] * batch_size
    #         ).to(self.device)
            
    #         # 将[0,1]范围转换回原始温度范围
    #         decoded_temp = decoded_denormalized * self.config.temp_range + self.config.temp_min
            
    #         # 取三个通道的平均值作为重建的单通道
    #         decoded_single = decoded_temp.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            
    #         # 调整回原始尺寸
    #         decoded_resized = F.interpolate(
    #             decoded_single, 
    #             size=(original_height, original_width),
    #             mode='bilinear',
    #             align_corners=False
    #         )
            
    #         reconstructed_channels.append(decoded_resized)
        
    #     # 堆叠所有重建的通道
    #     reconstructed = torch.cat(reconstructed_channels, dim=1)  # [B, 42, H, W]
        
    #     return reconstructed
    
    def compute_mse(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """
        计算MSE指标
        """
        # 首先将原始图像从[0,1]转换回温度值进行比较
        original_temp = original * self.config.temp_range + self.config.temp_min
        # 整体MSE（使用原始温度值）
        mse = F.mse_loss(reconstructed, original_temp, reduction='mean').item()

        # 每个通道的MSE
        channel_mse = []
        for ch in range(self.config.input_channels):
            ch_mse = F.mse_loss(
                reconstructed[:, ch, :, :], 
                original_temp[:, ch, :, :], 
                reduction='mean'
            ).item()
            channel_mse.append(ch_mse)
        
        return {
            'mse': mse,
            'mse_temp': mse,  # 现在直接就是温度单位的MSE
            'channel_mse': channel_mse,
            'rmse': np.sqrt(mse),
            'rmse_temp': np.sqrt(mse)
        }
    
    def test(self):
        """
        执行测试
        """
        print("Starting test on EN4 dataset...")
        print(f"Original image size: {self.config.height}x{self.config.width}")
        
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
                reconstructed = self.encode_decode_channels(images)
                
                # 计算MSE
                metrics = self.compute_mse(images, reconstructed)
                
                all_mse.append(metrics['mse'])
                all_mse_temp.append(metrics['mse_temp'])
                all_channel_mse.append(metrics['channel_mse'])
                
                # 保存一些可视化结果
                if i < 10:  # 保存前10个样本的可视化
                    original_temp = images[0].cpu() * self.config.temp_range + self.config.temp_min
                    save_path = os.path.join(
                        self.config.output_dir, 
                        f'reconstruction_sample_{i+1}.png'
                    )
                    # 可视化（现在数据已经是原始温度值）
                    visualize_reconstruction(
                        original_temp, 
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
        print("Test Results Summary:")
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
        save_path = os.path.join(self.config.output_dir, 'test_results.json')
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'model': self.config.pretrained_model_name,
                'num_samples': len(results['all_mse']),
                'input_shape': f"{self.config.input_channels}x{self.config.height}x{self.config.width}",
                'vae_scale_factor': 8,
                'note': 'Images are resized by processor to nearest multiple of 8'
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
        plt.title('MSE Distribution Across Depth Channels')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.config.output_dir, 'channel_mse_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Channel MSE plot saved to {save_path}")


def main():
    config = VAEConfig()
    tester = PretrainedVAETester(config)
    results = tester.test()


if __name__ == "__main__":
    main()