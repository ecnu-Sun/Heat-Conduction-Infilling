import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def visualize_sst_layers(file_path, layers_to_plot=None, output_file='sst_layers.png'):
    """
    可视化NetCDF文件中的多个深度层海温数据
    
    参数:
        file_path: NetCDF文件路径
        layers_to_plot: 要绘制的层索引列表，None表示自动选择
        output_file: 输出图片文件名
    """
    # 加载数据
    ds = xr.open_dataset(file_path)
    ds=ds.rename({'thetao':'temperature','lev':'depth'})
    da = ds['temperature']
    # da = ds['thetao']
    
    # 如果有时间维度，选择第一个时间点
    if 'time' in da.dims:
        da = da.isel(time=0)
    
    # 获取深度信息
    # lev = da.coords['lev'].values
    lev = da.coords['depth'].values
    # 选择要绘制的层
    if layers_to_plot is None:
        # 默认选择6个层：表层和几个代表性深度
        layers_to_plot = [0, 1, 2,3,4,5,6,7,8,9,41]
        layers_to_plot = [i for i in layers_to_plot if i < len(lev)]
    
    n_plots = len(layers_to_plot)
    
    # 计算子图布局
    if n_plots <= 3:
        rows, cols = 1, n_plots
    else:
        rows = 2
        cols = (n_plots + 1) // 2
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 计算全局最小最大值
    vmin = float('inf')
    vmax = float('-inf')
    for idx in layers_to_plot:
        data = da.isel(depth=idx).values
        vmin = min(vmin, np.nanmin(data))
        vmax = max(vmax, np.nanmax(data))
    
    # 绘制每一层
    for plot_idx, layer_idx in enumerate(layers_to_plot):
        ax = axes[plot_idx]
        
        # 获取数据
        data = da.isel(depth=layer_idx).values
        current_lev = lev[layer_idx]
        
        # 绘制数据
        im = ax.imshow(data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                      aspect='auto', origin='lower',interpolation='nearest')
        
        # 添加标题
        ax.set_title(f'depth: {current_lev:.1f} m', fontsize=10)
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # 总标题
    fig.suptitle('thetao', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_file}")
    
    # 打印统计信息
    # print("\n数据统计:")
    # for idx in layers_to_plot:
    #     data = da.isel(lev=idx).values
    #     valid_data = data[~np.isnan(data)]
    #     if len(valid_data) > 0:
    #         print(f"深度 {lev[idx]:6.1f} m: "
    #               f"均值={np.mean(valid_data):6.2f}°C, "
    #               f"最小={np.min(valid_data):6.2f}°C, "
    #               f"最大={np.max(valid_data):6.2f}°C")
    
    plt.show()
    return fig


# 使用示例
if __name__ == "__main__":
    filled_file = "/hdd/SunZL/data/BCC12_regrided_LAPLACE/BCC1_regrided_LAPLACE/thetao_Omon_BCC-CSM2-MR_historical_r1i1p1f1_gn_1850_01.nc"
    
    # 可视化默认6个深度层
    fig = visualize_sst_layers(filled_file, output_file='sst_layers_BCC1_185001_LAPLACE.png')
    
    # 或指定特定层
    # fig = visualize_sst_layers(filled_file, layers_to_plot=[0, 5, 10, 20], output_file='sst_custom.png')