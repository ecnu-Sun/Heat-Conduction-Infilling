import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from datetime import datetime
import glob
import re

# ==============================================================================
# 1. 请在这里修改您的配置
# ==============================================================================

# 包含年度 .nc 文件的输入文件夹路径
# 例如: /hdd/SunZL/data/SuperResolutionData/OISST/
INPUT_NC_DIR = "/hdd/SunZL/data/SuperResolutionData/OISST"

# [输出] 验证集 .npy 文件的保存路径 (1982-1984年)
VAL_GT_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA/val_GT"

# [输出] 训练集 .npy 文件的保存路径 (1985年及以后)
TRAIN_GT_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA/train_GT"

# # JP1 区域的经纬度范围,实际上： 30.125°N 到 45.875°N 130.125°E 到 145.875°E
# JP1_COORDS = {
#     "lat_min": 30.0,
#     "lat_max": 46.0,
#     "lon_min": 130.0,
#     "lon_max": 146.0,
# }
# 东南亚区域的经纬度范围 (64° x 64°)
# 在0.25度分辨率的OISST数据上，将截取出一个 256x256 的高分辨率区域
# 实际截取范围 Latitude: -21.875°S to 41.875°N, Longitude: 95.125°E to 158.875°E
COORDS = {
    "lat_min": -22.0,
    "lat_max": 42.0,
    "lon_min": 95.0,
    "lon_max": 159.0,
}

# ==============================================================================
# 2. 脚本主逻辑 (通常无需修改)
# ==============================================================================

def process_yearly_file(yearly_file_path, output_dir):
    """读取年度NC文件，裁剪JP1区域，并按天分割保存为NPY文件"""
    
    filename = os.path.basename(yearly_file_path)
    print(f"\n--- 正在向 {os.path.basename(output_dir)} 输出数据 ---")
    print(f"--- 处理源文件: {filename} ---")
    
    try:
        with xr.open_dataset(yearly_file_path) as ds:
            jp1_ds = ds.sel(
                lat=slice(COORDS['lat_min'], COORDS['lat_max']),
                lon=slice(COORDS['lon_min'], COORDS['lon_max'])
            )
            time_points = jp1_ds.time.values

            for timestamp in tqdm(time_points, desc=f"处理 {filename}"):
                dt_object = datetime.utcfromtimestamp(timestamp.astype('O') / 1e9)
                npy_filename = dt_object.strftime('%Y%m%d') + ".npy"
                npy_save_path = os.path.join(output_dir, npy_filename)

                if os.path.exists(npy_save_path):
                    continue

                daily_data = jp1_ds['sst'].sel(time=timestamp)
                npy_data = daily_data.values

                if npy_data.shape != (256, 256):
                    print(f"警告：{npy_filename} 的尺寸为 {npy_data.shape}，而不是预期！")

                np.save(npy_save_path, npy_data.astype(np.float32))

    except Exception as e:
        print(f"处理文件 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(TRAIN_GT_DIR, exist_ok=True)
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    
    nc_files = glob.glob(os.path.join(INPUT_NC_DIR, '*.nc'))
    
    if not nc_files:
        print(f"错误：在目录 {INPUT_NC_DIR} 中没有找到任何 .nc 文件。")
    else:
        print(f"找到 {len(nc_files)} 个年度 .nc 文件，开始分类处理...")

    # 循环处理所有找到的年度文件
    for file_path in sorted(nc_files):
        # 从文件名中提取年份 (例如从 'sst.day.mean.1985.nc' 中提取 '1985')
        match = re.search(r'\.(\d{4})\.nc$', os.path.basename(file_path))
        if not match:
            print(f"跳过无法识别年份的文件: {file_path}")
            continue
        
        year = int(match.group(1))
        
        # 根据年份判断输出目录
        if year >= 1982 and year <= 1984:
            target_dir = VAL_GT_DIR
        else:
            # 1985年及以后都放入训练集
            target_dir = TRAIN_GT_DIR
            
        # 调用处理函数
        process_yearly_file(file_path, target_dir)

    print("\n所有数据已按要求分割到训练集和验证集文件夹！")
    print(f"训练集 (GT) 数据位于: {TRAIN_GT_DIR}")
    print(f"验证集 (GT) 数据位于: {VAL_GT_DIR}")