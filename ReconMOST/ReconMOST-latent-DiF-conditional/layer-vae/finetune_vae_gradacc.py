import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import VAEConfig
from dataset import create_dataloader


class VAEFinetuner:
    def __init__(self, config: VAEConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        
        # 创建输出目录
        self.checkpoint_dir = os.path.join(config.output_dir, "finetuned_models")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 加载预训练VAE
        print(f"Loading pretrained VAE from {config.pretrained_model_name}...")
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name,
            subfolder="vae"
        ).to(self.device)
        
        # 初始化VaeImageProcessor
        self.vae_processor = VaeImageProcessor(
            # vae_scale_factor=8,
            do_resize=False,
            do_normalize=True,
            do_convert_rgb=False,
            do_convert_grayscale=False
        )
        
        # 创建训练数据加载器
        self.train_dataloader = create_dataloader(
            config.train_data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            temp_min=config.temp_min,
            temp_range=config.temp_range
        )
        
        print(f"Initialized finetuner with device: {self.device}")
        print(f"Training data directory: {config.train_data_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def finetune(self, num_epochs: int = 10, learning_rate: float = 1e-4):
        """
        微调VAE模型
        Args:
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}")
        
        # 设置为训练模式
        self.vae.train()
        
        # 设置优化器
        optimizer = AdamW(self.vae.parameters(), lr=learning_rate)
        eval_interval = 10
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1)
        # 训练循环
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(self.device)
                
                # ---- 【修改开始】 ----
                # 在每个batch开始前清空梯度
                optimizer.zero_grad()
                
                # 用于记录一个batch内所有通道组的累积损失值（非张量）
                batch_total_loss_val = 0
                batch_total_recon_loss_val = 0
                batch_total_kl_loss_val = 0
                
                # 对每个通道组进行处理
                for group_idx in range(self.config.num_channel_groups):
                    start_ch = group_idx * self.config.channels_per_group
                    end_ch = start_ch + self.config.channels_per_group
                    
                    # 提取对应的3个通道
                    group_channels = images[:, start_ch:end_ch, :, :]
                    
                    # 计算padding并应用反射padding，填充顶部和右侧到8的倍数
                    h, w = group_channels.shape[-2:]
                    group_channels = F.pad(group_channels, (0, (8 - w % 8) % 8, (8 - h % 8) % 8, 0), mode='reflect')
                    # 使用processor预处理
                    group_channels_processed = self.vae_processor.preprocess(
                        group_channels.cpu()
                    ).to(self.device)
                    
                    # 前向传播
                    posterior = self.vae.encode(group_channels_processed).latent_dist
                    z = posterior.sample()
                    decoded = self.vae.decode(z).sample
                    
                    # 计算损失
                    error = decoded - group_channels_processed
                    recon_loss = torch.mean(torch.pow(error, 2))
                    kl_loss = posterior.kl().mean()
                    group_loss = recon_loss + 0.000000000000001 * kl_loss

                    # 将损失根据通道组数量进行缩放，以实现梯度平均
                    scaled_loss = group_loss / self.config.num_channel_groups
                    
                    # 反向传播，梯度会累积
                    scaled_loss.backward()

                    # 记录损失值用于显示和统计
                    batch_total_loss_val += scaled_loss.item() * self.config.num_channel_groups
                    batch_total_recon_loss_val += recon_loss.item()
                    batch_total_kl_loss_val += kl_loss.item()
                
                # 在处理完一个batch的所有通道组后，执行一次优化器步骤
                optimizer.step()
                # ---- 【修改结束】 ----

                # 计算并记录整个batch的平均损失
                avg_loss_val = batch_total_loss_val / self.config.num_channel_groups
                avg_recon_loss_val = batch_total_recon_loss_val / self.config.num_channel_groups
                avg_kl_loss_val = batch_total_kl_loss_val / self.config.num_channel_groups

                # 记录损失 (使用 .item() 后的值)
                epoch_losses.append(avg_loss_val)
                epoch_recon_losses.append(avg_recon_loss_val)
                epoch_kl_losses.append(avg_kl_loss_val)
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{avg_loss_val:.7f}",
                    'recon': f"{avg_recon_loss_val:.7f}",
                    'kl': f"{avg_kl_loss_val:.7f}"
                })

            # 计算epoch平均损失
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_recon = np.mean(epoch_recon_losses)
            avg_epoch_kl = np.mean(epoch_kl_losses)
            
            # 学习率调度器步骤 (基于epoch的平均损失)
            scheduler.step(avg_epoch_loss)
            if (epoch + 1) % eval_interval == 0:
                 print(f"New learning rate: {optimizer.param_groups[0]['lr']}")

            print(f"Epoch {epoch+1} - Average loss: {avg_epoch_loss:.7f} "
                  f"(Recon: {avg_epoch_recon:.7f}, KL: {avg_epoch_kl:.7f})")
            
            # 保存checkpoint
            if (epoch + 1) % 1 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"vae_epoch_{epoch+1}.pt")
                torch.save({
                    'model_state_dict': self.vae.state_dict(),
                    'epoch': epoch + 1,
                    'loss': avg_epoch_loss,
                    'global_step': global_step,
                    'config': self.config
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_path = os.path.join(self.checkpoint_dir, "vae_best.pt")
                torch.save({
                    'model_state_dict': self.vae.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'global_step': global_step,
                    'config': self.config
                }, save_path)
                print(f"Saved best model with loss {best_loss:.7f}")
        
        # (后面的代码保持不变)
        # ...
        final_save_path = os.path.join(self.checkpoint_dir, "vae_final.pt")
        torch.save({
            'model_state_dict': self.vae.state_dict(),
            'epoch': num_epochs,
            'loss': avg_epoch_loss,
            'global_step': global_step,
            'config': self.config
        }, final_save_path)
        
        import json
        summary_path = os.path.join(self.checkpoint_dir, "finetuning_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'pretrained_model': self.config.pretrained_model_name,
                    'num_channel_groups': self.config.num_channel_groups,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': self.config.batch_size
                },
                'results': {
                    'best_loss': float(best_loss),
                    'final_loss': float(avg_epoch_loss),
                    'total_steps': global_step
                }
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Finetuning completed!")
        print(f"Best loss: {best_loss:.7f}")
        print(f"Models saved in: {self.checkpoint_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Finetune VAE on ocean temperature data')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # 创建配置
    config = VAEConfig()
    
    # 覆盖批次大小
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # 创建微调器并执行微调
    finetuner = VAEFinetuner(config)
    finetuner.finetune(args.num_epochs, args.learning_rate)


if __name__ == "__main__":
    main()