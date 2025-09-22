import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def calculate_and_plot_2d_psd(ax, image_data, title):
    """
    计算2D PSD并在指定的matplotlib axis上绘图。
    """
    # 1. 确保处理NaN值
    nan_mask = np.isnan(image_data)
    if np.any(nan_mask):
        image_data[nan_mask] = 0

    # 2. 计算2D傅里叶变换并移位
    fft_image = np.fft.fft2(image_data)
    fft_shifted = np.fft.fftshift(fft_image)

    # 3. 计算功率谱密度 (PSD)
    psd_2d = np.abs(fft_shifted)**2
    
    # 使用对数色阶来可视化，因为能量差异巨大
    # 添加一个很小的值以避免log(0)
    im = ax.imshow(psd_2d + 1e-9, cmap='viridis', norm=LogNorm())
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return im

# --- 主程序 ---

# 1. 文件路径
file_before = '/hdd/SunZL/data/FIO12_regrided/FIO1_regrided/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc'
file_after = '/hdd/SunZL/data/UNION_DATA_LAPLACE/FIO12_regrided_LAPLACE/FIO1_regrided_LAPLACE/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc'

# 2. 读取数据 (只取表层)
ds_before = xr.open_dataset(file_before)
ds_after = xr.open_dataset(file_after)

sst_before = ds_before['thetao'].isel(time=0, lev=0).values
sst_after = ds_after['thetao'].isel(time=0, lev=0).values

# 3. 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.style.use('default')

# 4. 分别绘制填充前后的2D PSD图
im1 = calculate_and_plot_2d_psd(axes[0], sst_before.copy(), 'Before Infilling (Original)')
im2 = calculate_and_plot_2d_psd(axes[1], sst_after.copy(), 'After Infilling (Laplace)')

# 5. 添加整体标题和颜色条
fig.suptitle('2D Power Spectral Density Comparison', fontsize=18, fontweight='bold')
fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label='Power (Log Scale)')

# 6. 保存并显示图像
# 推荐使用矢量图格式 .pdf
output_filename = 'sst_2d_psd_comparison.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"图像已保存为: {output_filename}")