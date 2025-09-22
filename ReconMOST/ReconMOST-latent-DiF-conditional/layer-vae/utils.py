import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import xarray as xr


def normalize_temperature(temp: torch.Tensor, temp_min: float = -5.0, temp_range: float = 45.0) -> torch.Tensor:
    """
    将温度从[temp_min, temp_max]归一化到[-1, 1]
    temp_max = temp_min + temp_range = -5 + 45 = 40
    """
    # 归一化到[0, 1]
    normalized = (temp - temp_min) / temp_range
    # 转换到[-1, 1]
    normalized = 2 * normalized - 1
    return normalized


def denormalize_temperature(normalized: torch.Tensor, temp_min: float = -5.0, temp_range: float = 45.0) -> torch.Tensor:
    """
    将归一化的温度从[-1, 1]还原到[temp_min, temp_max]
    """
    # 从[-1, 1]转换到[0, 1]
    normalized_01 = (normalized + 1) / 2
    # 还原到原始范围
    temp = normalized_01 * temp_range + temp_min
    return temp


def load_nc_file(filepath: str, var_names: List[str] = None) -> np.ndarray:
    """
    加载.nc文件并提取海温数据
    支持多种变量名和数据格式
    """
    if var_names is None:
        var_names = ['thetao', 'temperature', 'temp', 'TEMP', 'sst', 'SST', 'to']
    
    try:
        ds = xr.open_dataset(filepath)
        
        # 打印数据集信息用于调试
        # print(f"Dataset variables: {list(ds.variables.keys())}")
        # print(f"Dataset dimensions: {dict(ds.dims)}")
        
        # 尝试不同的变量名
        data = None
        used_var = None
        
        for var_name in var_names:
            if var_name in ds.variables:
                data = ds[var_name].values
                used_var = var_name
                break
        
        if data is None:
            # 如果没有找到预定义的变量名，尝试找到可能的温度变量
            # 查找包含'temp'或'sst'的变量名
            for var in ds.variables:
                if 'temp' in var.lower() or 'sst' in var.lower() or 'thetao' in var.lower():
                    data = ds[var].values
                    used_var = var
                    break
        
        ds.close()
        
        if data is None:
            raise ValueError(f"No temperature variable found in {filepath}. Available variables: {list(ds.variables.keys())}")
        
        # print(f"Loaded variable '{used_var}' with shape {data.shape}")
        return data
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算均方误差 (Mean Squared Error)
    """
    return torch.mean((img1 - img2) ** 2).item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 2.0) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    假设输入已经归一化到[-1, 1]，所以max_val=2.0
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def visualize_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor,
                           layer_idx: int = 0, save_path: Optional[str] = None,
                           vmin: float = -5, vmax: float = 40):
    """
    可视化原始图像和重建图像的对比
    输入应该是实际温度值（摄氏度）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 选择一个层进行可视化
    orig_layer = original[layer_idx].detach().cpu().numpy()
    recon_layer = reconstructed[layer_idx].detach().cpu().numpy()
    diff = orig_layer - recon_layer
    
    # 原始图像
    im1 = axes[0].imshow(orig_layer, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Original (Layer {layer_idx})')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')
    
    # 重建图像
    im2 = axes[1].imshow(recon_layer, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Reconstructed (Layer {layer_idx})')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Temperature (°C)')
    
    # 差异图
    diff_max = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    axes[2].set_title('Difference (Original - Reconstructed)')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='Temperature Difference (°C)')
    
    # 添加MSE和RMSE信息
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    fig.suptitle(f'MSE: {mse:.4f} °C², RMSE: {rmse:.4f} °C', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_depth_profile(original: torch.Tensor, reconstructed: torch.Tensor,
                      lat_idx: int, lon_idx: int, save_path: Optional[str] = None):
    """
    绘制某个位置的深度剖面对比
    """
    # 提取特定位置的深度剖面
    orig_profile = original[:, lat_idx, lon_idx].detach().cpu().numpy()
    recon_profile = reconstructed[:, lat_idx, lon_idx].detach().cpu().numpy()
    
    depths = np.arange(len(orig_profile))
    
    plt.figure(figsize=(8, 10))
    plt.plot(orig_profile, depths, 'b-', label='Original', linewidth=2)
    plt.plot(recon_profile, depths, 'r--', label='Reconstructed', linewidth=2)
    
    plt.gca().invert_yaxis()  # 深度向下
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth Layer')
    plt.title(f'Temperature Profile at Lat={lat_idx}, Lon={lon_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()