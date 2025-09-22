import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
import glob
warnings.filterwarnings('ignore')


def gauss_seidel_sor_vectorized(T, land_mask, omega=1.8, max_iter=5000, tol=1e-4):
    """
    使用向量化操作的SOR迭代法求解二维拉普拉斯方程
    采用红黑排序(Red-Black ordering)来实现并行更新
    
    参数:
        T: 2D numpy数组，初始温度场
        land_mask: 2D布尔数组，True表示陆地（需要填充的区域）
        omega: SOR松弛因子
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        T_filled: 填充后的2D数组
        converged: 是否收敛
        iterations: 实际迭代次数
    """
    T_filled = T.copy()
    ny, nx = T.shape
    
    # 创建红黑棋盘掩码
    i_indices, j_indices = np.ogrid[:ny, :nx]
    red_mask = ((i_indices + j_indices) % 2 == 0) & land_mask
    black_mask = ((i_indices + j_indices) % 2 == 1) & land_mask
    
    for iteration in range(max_iter):
        T_old = T_filled.copy()
        
        # 红黑排序：先更新红点，再更新黑点
        for mask in [red_mask, black_mask]:
            if np.sum(mask) == 0:
                continue
                
            # 使用np.roll获取四个方向的邻居
            T_left = np.roll(T_filled, 1, axis=1)
            T_right = np.roll(T_filled, -1, axis=1)
            T_up = np.roll(T_filled, 1, axis=0)
            T_down = np.roll(T_filled, -1, axis=0)
            
            # 处理纬度边界
            T_up[0, :] = T_filled[0, :]
            T_down[-1, :] = T_filled[-1, :]
            
            # 计算邻居数量
            neighbor_count = np.ones_like(T_filled) * 4
            neighbor_count[0, :] = 3    # 第一行
            neighbor_count[-1, :] = 3   # 最后一行
            
            # 计算邻居和
            neighbor_sum = T_left + T_right + T_up + T_down
            neighbor_sum[0, :] -= T_up[0, :]
            neighbor_sum[-1, :] -= T_down[-1, :]
            
            # 计算平均值
            average = neighbor_sum / neighbor_count
            
            # SOR更新（只更新当前颜色的点）
            T_filled[mask] = T_filled[mask] + omega * (average[mask] - T_filled[mask])
        
        # 检查收敛
        if np.sum(land_mask) > 0:
            max_change = np.max(np.abs(T_filled[land_mask] - T_old[land_mask]))
            if max_change < tol:
                return T_filled, True, iteration + 1
    
    return T_filled, False, max_iter


def gauss_seidel_sor_jacobi(T, land_mask, omega=1.8, max_iter=5000, tol=1e-4):
    """
    使用Jacobi风格的向量化SOR迭代（所有点同时更新）
    收敛较慢但完全并行
    
    参数:
        T: 2D numpy数组，初始温度场
        land_mask: 2D布尔数组，True表示陆地（需要填充的区域）
        omega: SOR松弛因子（对Jacobi方法，通常需要较小的omega）
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        T_filled: 填充后的2D数组
        converged: 是否收敛
        iterations: 实际迭代次数
    """
    T_filled = T.copy()
    omega_jacobi = omega * 0.8  # Jacobi方法需要较小的松弛因子
    
    for iteration in range(max_iter):
        T_old = T_filled.copy()
        
        # 使用np.roll获取四个方向的邻居（基于旧值）
        T_left = np.roll(T_old, 1, axis=1)
        T_right = np.roll(T_old, -1, axis=1)
        T_up = np.roll(T_old, 1, axis=0)
        T_down = np.roll(T_old, -1, axis=0)
        
        # 处理纬度边界
        T_up[0, :] = T_old[0, :]
        T_down[-1, :] = T_old[-1, :]
        
        # 计算邻居数量
        neighbor_count = np.ones_like(T_filled) * 4
        neighbor_count[0, :] = 3
        neighbor_count[-1, :] = 3
        
        # 计算邻居和
        neighbor_sum = T_left + T_right + T_up + T_down
        neighbor_sum[0, :] -= T_up[0, :]
        neighbor_sum[-1, :] -= T_down[-1, :]
        
        # 计算平均值
        average = neighbor_sum / neighbor_count
        
        # SOR更新（只更新陆地点）
        T_filled[land_mask] = T_old[land_mask] + omega_jacobi * (average[land_mask] - T_old[land_mask])
        
        # 检查收敛
        if np.sum(land_mask) > 0:
            max_change = np.max(np.abs(T_filled[land_mask] - T_old[land_mask]))
            if max_change < tol:
                return T_filled, True, iteration + 1
    
    return T_filled, False, max_iter


def process_single_layer(args):
    """
    处理单个深度层的函数（用于并行处理）
    """
    da_slice, level, idx, total_levels, omega, method = args
    
    print(f"处理第 {idx+1}/{total_levels} 层，深度: {level:.2f} m")
    
    # 获取NumPy数组
    T_2d = da_slice.values
    
    if np.all(np.isnan(T_2d)):
        print(f"  深度 {level:.2f} m：数据全为NaN，跳过处理")
        # 直接返回原始切片，表示未做任何改动
        # 返回 (切片, 迭代次数, 是否收敛) 的元组
        return da_slice, 0, True 
    # 创建掩码与初始化
    land_mask = np.isnan(T_2d)
    num_land_points = np.sum(land_mask)
    
    if num_land_points == 0:
        print(f"  深度 {level:.2f} m：无需填充")
        return da_slice, 0, True
    
    # 计算海洋区域平均值作为初始值
    ocean_mean = np.nanmean(T_2d)
    T_2d[land_mask] = ocean_mean
    
    print(f"  陆地点数: {num_land_points}, 海洋均温: {ocean_mean:.2f}")
    
    # 根据方法选择迭代函数
    if method == 'redblack':
        T_2d_filled, converged, iterations = gauss_seidel_sor_vectorized(
            T_2d, land_mask, omega=omega, max_iter=5000, tol=3e-4
        )
    else:  # jacobi
        T_2d_filled, converged, iterations = gauss_seidel_sor_jacobi(
            T_2d, land_mask, omega=omega, max_iter=5000, tol=3e-4
        )
    
    # 创建填充后的切片
    filled_slice = da_slice.copy(data=T_2d_filled)
    
    if converged:
        print(f"  ✓ 深度 {level:.2f} m 收敛于 {iterations} 次迭代")
    else:
        print(f"  ⚠ 深度 {level:.2f} m 未收敛")
    
    return filled_slice, iterations, converged


def fill_sst_land_areas(ds_input, output_file, omega=1.8, n_workers=4, method='redblack'):
    """
    填充分层海温数据中的陆地区域（并行处理版本）
    
    参数:
        ds_input: xarray.Dataset或文件路径
        output_file: 输出文件名
        omega: SOR松弛因子
        n_workers: 并行工作进程数
        method: 'redblack' 或 'jacobi'
    
    返回:
        ds_filled: 填充后的xarray.Dataset
    """
    # 数据加载
    if isinstance(ds_input, str):
        ds = xr.open_dataset(ds_input)
    else:
        ds = ds_input
    if 'temperature' in ds.variables and 'depth' in ds.variables:
        ds=ds.rename({'temperature':'thetao','depth':'lev'})
        ds['thetao']=ds['thetao']-273.15
    # 选取thetao变量
    da_thetao = ds['thetao']
    
    # 去除时间维度
    if 'time' in da_thetao.dims and da_thetao.sizes['time'] == 1:
        da_thetao = da_thetao.isel(time=0)
    
    print(f"数据形状: {da_thetao.shape}")
    print(f"SOR松弛因子: ω = {omega}")
    print(f"迭代方法: {method}")
    print(f"并行工作进程: {n_workers}")
    print("-" * 50)
    
    # 准备并行处理的参数
    lev_values = da_thetao.coords['lev'].values
    total_levels = len(lev_values)
    
    # 创建参数列表
    process_args = [
        (da_thetao.sel(lev=level), level, idx, total_levels, omega, method)
        for idx, level in enumerate(lev_values)
    ]
    
    # 并行处理所有层
    processed_layers = [None] * total_levels
    total_iterations = 0
    converged_count = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(process_single_layer, process_args)
        
        for idx, (filled_slice, iterations, converged) in enumerate(results):
            processed_layers[idx] = filled_slice
            total_iterations += iterations
            if converged:
                converged_count += 1
    
    # 合并结果
    print("\n合并处理结果...")
    final_da = xr.concat(processed_layers, dim='lev')
    
    # 创建新数据集
    ds_filled = xr.Dataset()
    ds_filled['thetao'] = final_da
    ds_filled.attrs = ds.attrs.copy()
    
    # 恢复时间维度
    if 'time' in ds['thetao'].dims:
        ds_filled['thetao'] = ds_filled['thetao'].expand_dims('time')
        ds_filled = ds_filled.assign_coords(time=ds.coords['time'])
    
    # 保存结果
    print(f"保存到: {output_file}")
    ds_filled.to_netcdf(output_file)
    
    # 统计信息
    print("\n处理完成！")
    print(f"收敛层数: {converged_count}/{total_levels}")
    if total_levels > 0:
        print(f"平均迭代: {total_iterations/total_levels:.1f} 次")
    
    return ds_filled


# 使用示例
if __name__ == "__main__":
    # 1. 定义输入和输出目录
    input_dir = "/hdd/SunZL/data/MRI12_regrided/MRI2_regrided"
    output_dir = "/hdd/SunZL/data/MRI12_regrided_LAPLACE/MRI2_regrided_LAPLACE"
    
    # 2. 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 查找输入目录中所有.nc文件
    # 使用 sorted() 来确保处理顺序一致
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.nc')))
    
    if not input_files:
        print(f"在目录 {input_dir} 中没有找到任何 .nc 文件。")
    else:
        print(f"找到 {len(input_files)} 个文件进行处理。")

    # 4. 遍历每个文件并进行处理
    for i, input_file in enumerate(input_files):
        # 从输入文件名构建输出文件名（保持文件名不变，只改变路径）
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)
        if os.path.exists(output_file):
            print(f"\n--- 文件 {i+1}/{len(input_files)}: {base_name} ---")
            print(f"输出文件已存在，跳过: {output_file}")
            continue  # 如果文件存在，则跳过本次循环，处理下一个文件
        print(f"\n{'='*60}")
        print(f"开始处理文件 {i+1}/{len(input_files)}: {base_name}")
        print(f"输入: {input_file}")
        print(f"输出: {output_file}")
        print(f"{'='*60}\n")
        
        # 调用核心处理函数
        # 使用红黑排序方法（推荐）
        ds_result = fill_sst_land_areas(input_file, output_file, omega=1.88, n_workers=4, method='redblack')
        
        # 或使用Jacobi方法（完全并行但收敛较慢）
        # ds_result = fill_sst_land_areas(input_file, output_file, omega=1.5, n_workers=4, method='jacobi')

    print(f"\n{'='*60}")
    print("所有文件处理完毕！")
    print(f"{'='*60}")