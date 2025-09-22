# 安装必要的库 (如果尚未安装)
# pip install numpy xarray matplotlib scipy netcdf4

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def calculate_radial_psd(image_data):
    """
    计算2D图像的径向平均功率谱密度。

    参数:
    image_data (np.ndarray): 输入的二维图像数据。

    返回:
    freqs (np.ndarray): 频率（波数）数组。
    psd_1d (np.ndarray): 径向平均后的功率谱密度。
    """
    # 1. 处理可能的NaN值，这里我们用0填充
    # 对于填充前的数据，这是必须的步骤
    nan_mask = np.isnan(image_data)
    if np.any(nan_mask):
        print("发现NaN值，使用0进行填充...")
        image_data[nan_mask] = 0

    # 2. 计算2D傅里叶变换
    fft_image = np.fft.fft2(image_data)
    fft_shifted = np.fft.fftshift(fft_image) # 将零频率分量移到中心

    # 3. 计算功率谱密度 (PSD)
    psd_2d = np.abs(fft_shifted)**2

    # 4. 计算径向平均
    # 创建频率坐标
    h, w = image_data.shape
    center_h, center_w = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)

    # 使用scipy.ndimage进行快速的径向求和
    psd_sum_by_radius = ndimage.sum(psd_2d, labels=r, index=np.arange(r.max() + 1))
    counts_by_radius = ndimage.sum(np.ones_like(psd_2d), labels=r, index=np.arange(r.max() + 1))
    
    # 避免除以零
    valid_indices = counts_by_radius > 0
    psd_1d = psd_sum_by_radius[valid_indices] / counts_by_radius[valid_indices]
    
    # 定义频率轴（波数）
    # 频率单位是 1/像素，最大频率是图像尺寸的一半
    freqs = np.arange(r.max() + 1)[valid_indices]
    
    return freqs, psd_1d

# --- 主程序 ---

# 1. 文件路径
file_before = '/hdd/SunZL/data/FIO12_regrided/FIO1_regrided/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc'
file_after = '/hdd/SunZL/data/UNION_DATA_LAPLACE/FIO12_regrided_LAPLACE/FIO1_regrided_LAPLACE/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc'

# 2. 读取数据 (只取表层)
ds_before = xr.open_dataset(file_before)
ds_after = xr.open_dataset(file_after)

# 提取表层 (lev=0) 的海温数据，并去掉多余的维度
sst_before = ds_before['thetao'].isel(time=0, lev=0).values
sst_after = ds_after['thetao'].isel(time=0, lev=0).values

print(f"数据维度: {sst_before.shape}")

# 3. 计算填充前后的径向PSD
freqs_before, psd_before = calculate_radial_psd(sst_before.copy())
freqs_after, psd_after = calculate_radial_psd(sst_after.copy())

# 4. 可视化对比
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# 使用对数-对数坐标 (log-log plot) 更能看清细节
ax.loglog(freqs_before, psd_before, label='Before Infilling (Original)', color='dodgerblue', linewidth=2)
ax.loglog(freqs_after, psd_after, label='After Infilling (Laplace)', color='orangered', linewidth=2, linestyle='--')

# 添加图表元素
ax.set_title('Power Spectral Density Comparison of SST Field', fontsize=16, fontweight='bold')
ax.set_xlabel('Spatial Frequency (Wavenumber)', fontsize=12)
ax.set_ylabel('Radially Averaged Power', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, which="both", ls="-", alpha=0.5)

# 优化坐标轴显示
ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig("fuliye.png")