import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm
from eofs.xarray import Eof

def convert_temperature_units(data_arr):
    """自动检测并转换温度单位"""
    valid_data = data_arr.values[~np.isnan(data_arr.values)]
    if len(valid_data) == 0:
        return data_arr
    
    data_mean = np.mean(valid_data)
    if data_mean > 250:  # 开尔文
        data_arr.values = data_arr.values - 273.15
    return data_arr

def calculate_mse_metrics(y_true, y_pred, guide_mask):
    """计算 MSE-g, MSE-r, 和 Total MSE"""
    y_true = convert_temperature_units(y_true.copy())
    y_pred = convert_temperature_units(y_pred.copy())
    
    y_true_np = y_true.values
    y_pred_np = y_pred.values
    
    valid_mask = ~np.isnan(y_true_np) & ~np.isnan(y_pred_np)
    
    guide_mask_3d = np.broadcast_to(guide_mask, y_true_np.shape) if guide_mask.ndim == 2 else guide_mask
    
    # MSE-g
    g_mask = guide_mask_3d & valid_mask
    mse_g = np.mean((y_pred_np[g_mask] - y_true_np[g_mask])**2) if g_mask.sum() > 0 else np.nan
    
    # MSE-r
    r_mask = (~guide_mask_3d) & valid_mask
    mse_r = np.mean((y_pred_np[r_mask] - y_true_np[r_mask])**2) if r_mask.sum() > 0 else np.nan
    
    # Total MSE
    total_mse = np.mean((y_pred_np[valid_mask] - y_true_np[valid_mask])**2) if valid_mask.sum() > 0 else np.nan
    
    print(f"\n--- 性能评估结果 ---")
    print(f"  MSE-g (引导点误差): {mse_g:.4f}")
    print(f"  MSE-r (重建点误差): {mse_r:.4f}")
    print(f"  Total MSE (总误差): {total_mse:.4f}")

def find_match_streaming(test_data_arr, train_dir, method='random', percentage=0.003):
    """流式处理训练数据找到最佳匹配"""
    test_data = convert_temperature_units(test_data_arr.copy())
    
    # 准备引导点掩码
    if method == 'random':
        valid_indices = np.argwhere(~np.isnan(test_data.values))
        num_guide_points = max(1, int(len(valid_indices) * percentage))
        random_indices = np.random.choice(len(valid_indices), num_guide_points, replace=False)
        guide_coords = valid_indices[random_indices]
        
        guide_mask = np.zeros_like(test_data.values, dtype=bool)
        guide_mask[tuple(guide_coords.T)] = True
        test_values_at_guides = test_data.values[guide_mask]
    else:  # EOF方法
        # 需要先快速扫描一遍数据来计算EOF
        print("正在计算EOF...")
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.nc")))
        
        # 采样部分文件来计算EOF（避免内存问题）
        sample_size = min(50, len(train_files))
        sample_files = np.random.choice(train_files, sample_size, replace=False)
        
        samples = []
        for f in tqdm(sample_files, desc="采样数据计算EOF"):
            with xr.open_dataset(f) as ds:
                data = ds['thetao']
                data = data.isel(lev=0) if 'lev' in data.dims else data
                samples.append(data)
        
        sample_data = xr.concat(samples, dim='time')
        sample_data = convert_temperature_units(sample_data)
        
        coslat = np.cos(np.deg2rad(sample_data.coords['lat'].values)).clip(0., 1.)
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(sample_data.fillna(0), weights=wgts)
        
        eof1 = solver.eofsAsCovariance(neofs=1).squeeze()
        abs_eof1 = np.abs(eof1.values)
        threshold = np.percentile(abs_eof1[~np.isnan(abs_eof1)], 100 - (100 * percentage))
        guide_mask = (abs_eof1 >= threshold) & (~np.isnan(eof1.values))
        
        test_sst = test_data.isel(lev=0) if 'lev' in test_data.dims else test_data
        test_values_at_guides = test_sst.values[guide_mask]
    
    # 流式处理找最佳匹配
    min_mse = float('inf')
    best_match_data = None
    best_match_time = None
    
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.nc")))
    print(f"\n开始流式处理 {len(train_files)} 个训练文件...")
    min_distance=999999
    
    for train_file in tqdm(train_files, desc="搜索最佳匹配"):
        with xr.open_dataset(train_file) as ds:
            train_var_name = 'temperature' if 'temperature' in ds else 'thetao'
            train_data = ds[train_var_name] 
            train_data = convert_temperature_units(train_data)
            
            for i in range(len(train_data.time)):
                train_slice = train_data.isel(time=i)
                
                if method == 'eof':
                    train_slice = train_slice.isel(lev=0) if 'lev' in train_slice.dims else train_slice
                    train_values = train_slice.values[guide_mask]
                else:
                    train_values = train_slice.values[guide_mask]
                
                mask = ~np.isnan(test_values_at_guides) & ~np.isnan(train_values)
                if mask.sum() == 0:
                    continue
                    
                distance = np.mean((test_values_at_guides[mask] - train_values[mask])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    print(min_distance)
                    best_match_time = train_slice.time.values
                    # 保存完整的3D数据用于最终评估
                    best_match_data = ds[train_var_name].sel(time=train_slice.time).squeeze()
    
    print(f"\n找到的最佳匹配时间: {best_match_time}")
    print(f"引导点上的最小MSE: {min_distance:.4f}")
    
    if best_match_data is not None:
        calculate_mse_metrics(test_data_arr, best_match_data, guide_mask)

def main():
    TRAIN_DATA_DIR = "/data/coding/data/ReconMOST-testdata-celsius_except1"
    TEST_DATA_DIR = "/data/coding/data/ReconMOST-testdata-celsius"
    TEST_FILE_INDEX = 6
    PERCENTAGE_OF_POINTS = 0.1
    
    np.random.seed(42)
    
    # 加载测试数据
    test_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.nc")))
    test_file_path = test_files[TEST_FILE_INDEX]
    print(f"使用测试文件: {os.path.basename(test_file_path)}")
    
    test_ds = xr.open_dataset(test_file_path)
    test_var_name = 'temperature' if 'temperature' in test_ds else 'thetao'
    test_data_arr = test_ds[test_var_name].squeeze(dim='time', drop=True)
    
    # 方法一：随机点匹配
    print("\n" + "="*20 + " 方法一：随机点匹配 " + "="*20)
    find_match_streaming(test_data_arr, TRAIN_DATA_DIR, method='random', percentage=PERCENTAGE_OF_POINTS)
    
    # 方法二：EOF关键点匹配
    # print("\n" + "="*20 + " 方法二：EOF关键点匹配 " + "="*20)
    # find_match_streaming(test_data_arr, TRAIN_DATA_DIR, method='eof', percentage=PERCENTAGE_OF_POINTS * 1000)

if __name__ == '__main__':
    main()