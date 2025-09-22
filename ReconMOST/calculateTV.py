import numpy as np
import xarray as xr
import os
import gzip

def total_variation(image_data):
    """计算图像的总变差"""
    grad_y, grad_x = np.gradient(image_data)
    tv = np.sum(np.sqrt(grad_x**2 + grad_y**2))
    return tv

def fractal_dimension_box_counting(image_data, threshold=0.01):
    """使用Box-Counting方法估算图像的内在维度"""
    grad_x = np.gradient(image_data, axis=1)
    grad_y = np.gradient(image_data, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    # 防止空梯度图报错
    if np.max(edges) == 0:
        return 1.0 # 一个完全平坦的图像，其边缘分形维数接近1

    binary_image = edges > threshold * np.max(edges)
    pixels = np.where(binary_image)
    
    if len(pixels[0]) == 0:
        return 1.0 # 没有检测到边缘

    Lx, Ly = binary_image.shape
    scales = np.logspace(0.01, np.log10(min(Lx, Ly)/2), num=10, endpoint=False, base=10)
    
    counts = []
    for scale in scales:
        scale = int(np.ceil(scale))
        H, _, _ = np.histogram2d(pixels[0], pixels[1], bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        counts.append(np.sum(H > 0))
    
    # 过滤掉counts为0的情况，避免log(0)
    valid_indices = np.array(counts) > 0
    if np.sum(valid_indices) < 2: # 需要至少两个点来进行线性回归
        return 1.0 
        
    log_scales = np.log(scales[valid_indices])
    log_counts = np.log(np.array(counts)[valid_indices])
    
    coeffs = np.polyfit(log_scales, log_counts, 1)
    return -coeffs[0]

def calculate_compressibility(image_data, filename):
    """计算给定numpy数组的gzip压缩大小"""
    temp_filename = filename + '.npy.gz'
    with gzip.open(temp_filename, 'wb') as f:
        np.save(f, image_data)
    size = os.path.getsize(temp_filename)
    os.remove(temp_filename)
    return size

# --- 主程序 ---

# 1. 加载数据
print("正在加载数据...")
ds_before = xr.open_dataset('/hdd/SunZL/data/FIO12_regrided/FIO1_regrided/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc')
sst_before_raw = ds_before['thetao'].isel(time=0, lev=0).values

ds_after = xr.open_dataset('/hdd/SunZL/data/UNION_DATA_LAPLACE/FIO12_regrided_LAPLACE/FIO1_regrided_LAPLACE/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_2014_12.nc')
sst_after_laplace = ds_after['thetao'].isel(time=0, lev=0).values

# 2. 准备三种场景的数据
print("正在准备三种场景的数据...")
# 场景A: 标准做法 - 将陆地(NaN)用0填充
sst_before_zerofilled = np.nan_to_num(sst_before_raw, nan=0.0)

# 场景B: 更强的基线 - 将陆地(NaN)用海洋均值填充
ocean_mean_sst = np.nanmean(sst_before_raw)
sst_before_meanfilled = np.nan_to_num(sst_before_raw, nan=ocean_mean_sst)
print(f"计算出的海洋平均温度为: {ocean_mean_sst:.4f} °C")

# 场景C: 您的方法 - 使用平滑填充后的数据
# sst_after_laplace 已加载

# 3. 计算所有指标
print("\n正在计算所有指标...")
# 总变差
tv_zero = total_variation(sst_before_zerofilled)
tv_mean = total_variation(sst_before_meanfilled)
tv_laplace = total_variation(sst_after_laplace)

# 分形维数
fd_zero = fractal_dimension_box_counting(sst_before_zerofilled)
fd_mean = fractal_dimension_box_counting(sst_before_meanfilled)
fd_laplace = fractal_dimension_box_counting(sst_after_laplace)

# 可压缩性
size_zero = calculate_compressibility(sst_before_zerofilled, 'zero_filled')
size_mean = calculate_compressibility(sst_before_meanfilled, 'mean_filled')
size_laplace = calculate_compressibility(sst_after_laplace, 'laplace_filled')

# 4. 打印最终对比结果
print("\n--- 指标对比最终结果 ---")
print(f"{'指标':<20} | {'场景A (填0)':<20} | {'场景B (填均值)':<20} | {'场景C (您的方法)':<20}")
print("-" * 85)
print(f"{'总变差 (TV)':<20} | {tv_zero:<20.4e} | {tv_mean:<20.4e} | {tv_laplace:<20.4e}")
print(f"{'分形维数 (FD)':<20} | {fd_zero:<20.4f} | {fd_mean:<20.4f} | {fd_laplace:<20.4f}")
print(f"{'压缩大小 (KB)':<20} | {size_zero/1024:<20.2f} | {size_mean/1024:<20.2f} | {size_laplace/1024:<20.2f}")
print("-" * 85)