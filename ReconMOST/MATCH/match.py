import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm
from eofs.xarray import Eof
import pandas as pd

# ----------------- Helper for Debugging -----------------
def print_debug_info(arr, name="array"):
    """打印 xarray.DataArray 或 numpy.ndarray 的调试信息"""
    print(f"\n--- DEBUG: 详细信息 for '{name}' ---")
    if isinstance(arr, xr.DataArray):
        print("类型: xarray.DataArray")
        print(f"形状 (Shape): {arr.shape}")
        print(f"维度 (Dims): {arr.dims}")
        print("坐标 (Coords):")
        for coord in arr.coords:
            print(f"  - {coord}: {arr.coords[coord].shape}")
        # 检查NaN值
        nan_count = np.isnan(arr.values).sum()
        total_count = arr.size
        print(f"NaN 值统计: {nan_count} / {total_count} ({nan_count/total_count:.2%})")
        # 添加数据范围信息
        valid_data = arr.values[~np.isnan(arr.values)]
        if len(valid_data) > 0:
            print(f"数据范围: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")
            print(f"数据均值: {np.mean(valid_data):.2f}")
    elif isinstance(arr, np.ndarray):
        print("类型: numpy.ndarray")
        print(f"形状 (Shape): {arr.shape}")
        nan_count = np.isnan(arr).sum()
        total_count = arr.size
        print(f"NaN 值统计: {nan_count} / {total_count} ({nan_count/total_count:.2%})")
    else:
        print(f"未知类型: {type(arr)}")
    print("---------------------------------------\n")
# ---------------------------------------------------------


def load_all_training_data(train_dir):
    """
    加载所有训练数据到一个大的 xarray.Dataset 中。
    【修正版】：使用显式排序和xr.concat来合并文件。
    """
    print(f"--- 正在加载所有训练数据从: {train_dir} ---")
    
    all_files = sorted(glob.glob(os.path.join(train_dir, "*.nc")))
    if not all_files:
        raise ValueError(f"在目录 {train_dir} 中找不到任何 .nc 文件。")

    print(f"找到 {len(all_files)} 个训练文件，开始加载...")

    datasets = [xr.open_dataset(f) for f in tqdm(all_files, desc="正在打开文件")]

    print("正在合并所有文件...")
    combined_ds = xr.concat(datasets, dim='time')
    
    print(f"所有训练数据加载完毕，总共 {len(combined_ds.time)} 个时间步。")
    return combined_ds['thetao']


def convert_temperature_units(data_arr, from_unit='auto'):
    """
    自动检测并转换温度单位
    from_unit: 'kelvin', 'celsius', or 'auto'
    """
    valid_data = data_arr.values[~np.isnan(data_arr.values)]
    if len(valid_data) == 0:
        return data_arr
    
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    data_mean = np.mean(valid_data)
    
    if from_unit == 'auto':
        # 自动检测单位
        if data_min > 200 and data_mean > 250:  # 很可能是开尔文
            from_unit = 'kelvin'
        elif data_min > -5 and data_max < 50:  # 很可能是摄氏度
            from_unit = 'celsius'
        else:
            print(f"警告：无法自动确定温度单位 (范围: [{data_min:.2f}, {data_max:.2f}], 均值: {data_mean:.2f})")
            return data_arr
    
    if from_unit == 'kelvin':
        print(f"检测到开尔文温度，转换为摄氏度...")
        converted_data = data_arr.copy()
        converted_data.values = data_arr.values - 273.15
        return converted_data
    else:
        return data_arr


def calculate_and_print_mse_metrics(y_true, y_pred, guide_mask):
    """计算并打印 MSE-g, MSE-r, 和 Total MSE，并加入详细的调试信息。"""
    
    # --- DEBUG: 检查传入计算函数的数据 ---
    print("\n--- DEBUG: 进入 MSE 计算函数 ---")
    print_debug_info(y_true, "y_true (测试数据) - 原始")
    print_debug_info(y_pred, "y_pred (匹配的训练数据) - 原始")
    
    # 转换温度单位使其一致
    y_true_converted = convert_temperature_units(y_true.copy())
    y_pred_converted = convert_temperature_units(y_pred.copy())
    
    print("\n--- DEBUG: 单位转换后的数据 ---")
    print_debug_info(y_true_converted, "y_true (测试数据) - 转换后")
    print_debug_info(y_pred_converted, "y_pred (匹配的训练数据) - 转换后")
    
    y_true_np = y_true_converted.values
    y_pred_np = y_pred_converted.values

    # --- DEBUG: 检查维度和 NaN 重叠情况 ---
    if y_true_np.shape != y_pred_np.shape:
        print(f"!!! 严重错误: y_true 和 y_pred 的形状不匹配! {y_true_np.shape} vs {y_pred_np.shape}")
        return

    valid_mask_true = ~np.isnan(y_true_np)
    valid_mask_pred = ~np.isnan(y_pred_np)
    valid_mask_both = valid_mask_true & valid_mask_pred
    
    print("--- DEBUG: 数据有效性分析 ---")
    print(f"y_true 中的有效点数: {valid_mask_true.sum()}")
    print(f"y_pred 中的有效点数: {valid_mask_pred.sum()}")
    print(f"共同有效点数 (可用于比较的总点数): {valid_mask_both.sum()}")
    if valid_mask_both.sum() == 0:
        print("!!! 严重错误: 测试数据和匹配数据之间没有共同的有效数据点！这很可能是导致所有MSE为NaN的原因。请检查陆地/海洋掩码是否一致。")
    print("----------------------------")


    if guide_mask.ndim == 2:
        # 将2D掩码广播到3D数据的形状
        guide_mask_3d = np.broadcast_to(guide_mask, y_true_np.shape)
    else:
        guide_mask_3d = guide_mask

    # MSE-g 计算
    g_points_mask = guide_mask_3d & valid_mask_both
    num_g_points = np.sum(g_points_mask)
    print(f"\n--- DEBUG: MSE-g (引导点) 分析 ---")
    print(f"原始引导点掩码中的点数: {np.sum(guide_mask_3d)}")
    print(f"同时在引导点掩码和双方数据中都有效的点数: {num_g_points}")
    if num_g_points > 0:
        mse_g = np.mean((y_pred_np[g_points_mask] - y_true_np[g_points_mask])**2)
    else:
        print("警告: 没有有效的引导点可用于计算 MSE-g。")
        mse_g = np.nan

    # MSE-r 计算
    r_points_mask = (~guide_mask_3d) & valid_mask_both
    num_r_points = np.sum(r_points_mask)
    print(f"\n--- DEBUG: MSE-r (重建点) 分析 ---")
    print(f"原始重建点掩码中的点数: {np.sum(~guide_mask_3d)}")
    print(f"同时在重建点掩码和双方数据中都有效的点数: {num_r_points}")
    if num_r_points > 0:
        mse_r = np.mean((y_pred_np[r_points_mask] - y_true_np[r_points_mask])**2)
    else:
        print("警告: 没有有效的重建点可用于计算 MSE-r。")
        mse_r = np.nan

    # Total MSE 计算
    total_points = np.sum(valid_mask_both)
    print(f"\n--- DEBUG: Total MSE (总误差) 分析 ---")
    print(f"用于计算总MSE的点数: {total_points}")
    if total_points > 0:
        total_mse = np.mean((y_pred_np[valid_mask_both] - y_true_np[valid_mask_both])**2)
    else:
        print("警告: 没有任何点可用于计算 Total MSE。")
        total_mse = np.nan


    print("\n--- 性能评估结果 ---")
    print(f"  MSE-g (引导点误差): {mse_g:.4f}")
    print(f"  MSE-r (重建点误差): {mse_r:.4f}")
    print(f"  Total MSE (总误差): {total_mse:.4f}")
    print("----------------------")


def find_match_random_points(test_data_arr, train_data_arr, percentage=0.003):
    print("\n" + "="*20 + " 方法一：随机点匹配 " + "="*20)
    
    # 首先转换温度单位
    test_data_converted = convert_temperature_units(test_data_arr.copy())
    train_data_converted = convert_temperature_units(train_data_arr.copy())
    
    valid_indices = np.argwhere(~np.isnan(test_data_converted.values))
    num_valid_points = len(valid_indices)
    # num_guide_points = int(num_valid_points * percentage)
    num_guide_points = 1
    random_indices = np.random.choice(num_valid_points, num_guide_points, replace=False)
    guide_coords_indices = valid_indices[random_indices]
    
    guide_mask_3d = np.zeros_like(test_data_converted.values, dtype=bool)
    guide_points_idx_tuple = tuple(guide_coords_indices.T)
    guide_mask_3d[guide_points_idx_tuple] = True
    
    print(f"从测试数据中随机选择了 {num_guide_points} 个引导点。")
    
    test_values_at_guides = test_data_converted.values[guide_mask_3d]
    
    min_distance = float('inf')
    best_match_time = None

    print("正在遍历训练集寻找最佳匹配...")
    for i in tqdm(range(len(train_data_converted.time))):
        train_slice = train_data_converted.isel(time=i)
        train_values_at_guides = train_slice.values[guide_mask_3d]
        
        mask = ~np.isnan(test_values_at_guides) & ~np.isnan(train_values_at_guides)
        if np.sum(mask) == 0: continue
        distance = np.mean(np.abs(test_values_at_guides[mask] - train_values_at_guides[mask]))
        
        if distance < min_distance:
            min_distance = distance
            best_match_time = train_slice.time.values

    print("\n--- 随机点匹配结果 ---")
    if best_match_time is not None:
        print(f"找到的最佳匹配样本时间为: {best_match_time}")
        print(f"最小平均绝对差异: {min_distance:.4f}°C")
        best_match_slice = train_data_arr.sel(time=best_match_time).squeeze()
        calculate_and_print_mse_metrics(test_data_arr, best_match_slice, guide_mask_3d)
    else:
        print("未能找到任何有效的匹配。")


def find_match_pca_points(test_data_arr, train_data_arr, percentage=0.03):
    print("\n" + "="*20 + " 方法二：主成分分析(EOF)关键点匹配 " + "="*20)
    
    # 首先转换温度单位
    test_data_converted = convert_temperature_units(test_data_arr.copy())
    train_data_converted = convert_temperature_units(train_data_arr.copy())
    
    print("正在对训练集进行EOF分析以找到关键区域...")
    # --- 修正：确保用于EOF分析的数据是2D的（表面数据） ---
    if 'lev' in train_data_converted.dims:
        train_sst = train_data_converted.isel(lev=0, drop=True)
        print("--- DEBUG: 已从训练数据中选取 lev=0 的表面数据进行EOF分析。---")
    else:
        # 如果没有深度层，假设它已经是2D+time数据
        train_sst = train_data_converted
        print("--- DEBUG: 训练数据没有 'lev' 维度，将直接用于EOF分析。---")

    coslat = np.cos(np.deg2rad(train_sst.coords['lat'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(train_sst.fillna(0), weights=wgts) # 使用 fillna(0) 增加稳定性
    
    eof1 = solver.eofsAsCovariance(neofs=1).squeeze()
    
    abs_eof1 = np.abs(eof1.values)
    threshold = np.percentile(abs_eof1[~np.isnan(abs_eof1)], 100 - (100 * percentage))
    key_points_mask_2d = (abs_eof1 >= threshold) & (~np.isnan(eof1.values))
    num_key_points = np.sum(key_points_mask_2d)
    print(f"EOF分析完成，已确定 {num_key_points} 个最重要的关键点。")
    
    # 修正：同时检查 'lev' 和 'depth' 维度
    if 'lev' in test_data_converted.dims:
        test_sst = test_data_converted.isel(lev=0, drop=True)
        print("--- DEBUG: 已从测试数据中选取 lev=0 的表面数据用于匹配。---")
    elif 'depth' in test_data_converted.dims:
        test_sst = test_data_converted.isel(depth=0, drop=True)
        print("--- DEBUG: 已从测试数据中选取 depth=0 的表面数据用于匹配。---")
    else:
        test_sst = test_data_converted
        print("--- DEBUG: 测试数据没有深度维度，将直接用于匹配。---")
    
    test_values_at_keys = test_sst.values[key_points_mask_2d]
    
    min_distance = float('inf')
    best_match_time = None

    print("正在使用EOF关键点遍历训练集寻找最佳匹配...")
    for i in tqdm(range(len(train_data_converted.time))):
        train_slice_3d = train_data_converted.isel(time=i)
        # 同样，从训练切片中提取表面数据
        if 'lev' in train_slice_3d.dims:
             train_slice_sst = train_slice_3d.isel(lev=0, drop=True)
        else:
             train_slice_sst = train_slice_3d

        train_values_at_keys = train_slice_sst.values[key_points_mask_2d]
        
        mask = ~np.isnan(test_values_at_keys) & ~np.isnan(train_values_at_keys)
        if np.sum(mask) == 0: continue
        distance = np.mean(np.abs(test_values_at_keys[mask] - train_values_at_keys[mask]))
        
        if distance < min_distance:
            min_distance = distance
            best_match_time = train_slice_3d.time.values
            
    print("\n--- EOF关键点匹配结果 ---")
    if best_match_time is not None:
        print(f"找到的最佳匹配样本时间为: {best_match_time}")
        print(f"最小平均绝对差异: {min_distance:.4f}°C")
        best_match_slice_3d = train_data_arr.sel(time=best_match_time).squeeze()
        
        # --- DEBUG: 传递给MSE计算的掩码是2D的，而数据是3D的，这是预期的行为 ---
        print("--- DEBUG: 将使用2D EOF掩码和完整的3D数据场进行最终的性能评估。---")
        calculate_and_print_mse_metrics(test_data_arr, best_match_slice_3d, key_points_mask_2d)
    else:
        print("未能找到任何有效的匹配。")


if __name__ == '__main__':
    TRAIN_DATA_DIR = "/data/coding/data/ReconMOST-testdata-celsius-align"
    TEST_DATA_DIR = "/data/coding/data/ReconMOST-testdata-celsius-align"
    TEST_FILE_INDEX = 0 
    PERCENTAGE_OF_POINTS = 0.00003
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    try:
        train_data_arr = load_all_training_data(TRAIN_DATA_DIR)
        # --- DEBUG: 检查加载后的训练数据 ---
        print_debug_info(train_data_arr, "train_data_arr (loaded)")

    except ValueError as e:
        print(e); exit()

    test_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.nc")))
    if not test_files:
        print(f"错误：在测试目录 {TEST_DATA_DIR} 中找不到任何文件。"); exit()
    
    test_file_path = test_files[TEST_FILE_INDEX]
    print(f"\n将使用测试文件 '{os.path.basename(test_file_path)}' 进行匹配。")
    test_ds = xr.open_dataset(test_file_path)
    

    if 'temperature' in test_ds:
        test_var_name = 'temperature'
    elif 'thetao' in test_ds:
        test_var_name = 'thetao'
    else:
        print(f"错误：测试文件 {test_file_path} 中找不到 'temperature' 或 'thetao' 变量。")
        print(f"可用的变量: {list(test_ds.data_vars)}")
        exit()
        
    test_data_arr = test_ds[test_var_name].squeeze(dim='time', drop=True)
    
    # --- DEBUG: 检查加载后的测试数据 ---
    print_debug_info(test_data_arr, "test_data_arr (loaded)")

    
    find_match_random_points(test_data_arr, train_data_arr, percentage=PERCENTAGE_OF_POINTS)
    # find_match_pca_points(test_data_arr, train_data_arr, percentage=PERCENTAGE_OF_POINTS)