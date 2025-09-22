import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
# ==============================================================================
# --- 请根据您的实际情况修改以下文件路径 ---
# ==============================================================================
LOG_UNCONDITIONAL = '/hdd/SunZL/Super-resolution/Super-resolution-for-sea-surface-temperature-with-CNN-and-GAN/results/test_SEA_rcan_laplace_noise/test_test_SEA_rcan_laplace_noise_250910-131458.log'  # 第1步生成的无条件模型日志
LOG_CONDITIONAL = '/hdd/SunZL/Super-resolution/Super-resolution-for-sea-surface-temperature-with-CNN-and-GAN/results/test_SEA_rcan_laplace_nino_noise/test_test_SEA_rcan_laplace_nino_noise_250910-131700.log'    # 第1步生成的有条件模型日志
NINO4_CSV_PATH = '/hdd/SunZL/data/nino4.long.anom.csv' # 您的Nino4指数文件
OUTPUT_FIGURE_NAME = 'RMSE_Advantage_vs_Nino4_Correlation.png'
# ==============================================================================

def parse_rmse_log(filepath):
    """从日志文件中解析出日期和物理单位的RMSE"""
    if not os.path.exists(filepath):
        print(f"错误: 日志文件未找到 {filepath}")
        return None
    pattern = re.compile(r"(\d{8})\s+-\s+RMSE:\s+([\d.]+);")
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data.append({'Date': match.group(1), 'RMSE': float(match.group(2))})
    return pd.DataFrame(data)

def run_analysis():
    """主分析函数"""
    print("1. 正在解析日志文件...")
    df_uncond = parse_rmse_log(LOG_UNCONDITIONAL)
    df_cond = parse_rmse_log(LOG_CONDITIONAL)
    if df_uncond is None or df_cond is None: return

    print("2. 正在合并模型结果...")
    df_merged = pd.merge(df_uncond, df_cond, on='Date', suffixes=('_uncond', '_cond'))
    df_merged['advantage_rmse'] = df_merged['RMSE_uncond'] - df_merged['RMSE_cond']
    
    print("3. 正在加载并匹配Nino4指数...")
    df_nino = pd.read_csv(NINO4_CSV_PATH)
    df_nino.columns = [col.strip() for col in df_nino.columns]
    df_nino['YYYY-MM'] = pd.to_datetime(df_nino['Date']).dt.strftime('%Y-%m')
    
    df_merged['YYYY-MM'] = pd.to_datetime(df_merged['Date'], format='%Y%m%d').dt.strftime('%Y-%m')
    df_final = pd.merge(df_merged, df_nino[['YYYY-MM', 'NINA4']], on='YYYY-MM', how='left')
    df_final['NINA4_abs'] = df_final['NINA4'].abs()
    
    # --- 步骤2：修改计算方式 ---
    # 在计算前移除任何可能存在的缺失值行
    df_final.dropna(subset=['advantage_rmse', 'NINA4_abs'], inplace=True)
    
    # 使用 pearsonr 同时计算相关系数 r 和 P值 p
    correlation, p_value = pearsonr(df_final['NINA4_abs'], df_final['advantage_rmse'])
    num_samples = len(df_final)

    print("\n" + "="*50)
    print(f"样本数量 (n): {num_samples}")
    print(f"相关系数 (r): {correlation:.4f}")
    print(f"P值 (p-value): {p_value:.4e}") # 使用科学计数法显示极小值
    print("="*50 + "\n")

    # --- 步骤3：更新绘图代码以显示P值 ---
    print("5. 正在绘制相关性散点图...")
    plt.figure(figsize=(10, 8))
    sns.regplot(data=df_final, x='NINA4_abs', y='advantage_rmse',
                scatter_kws={'alpha':0.3}, line_kws={"color":"red", "linewidth": 2.5})
    
    plt.title('Performance Advantage of Conditional Model vs. Climate Anomaly Strength', fontsize=16)
    plt.xlabel('Climate Anomaly Strength (|Niño 4 Index|)', fontsize=12)
    plt.ylabel('RMSE Improvement (Unconditional - Conditional)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # 在图上同时标注 r 和 p
    annotation_text = f'Pearson r = {correlation:.3f}\np-value = {p_value:.2e}\nn = {num_samples}'
    plt.annotate(annotation_text, 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=14, ha='left', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="ivory", ec="gray", lw=1))
                 
    plt.savefig(OUTPUT_FIGURE_NAME, dpi=150)
    print(f"分析图表已保存为: {OUTPUT_FIGURE_NAME}")
    plt.show()

if __name__ == '__main__':
    run_analysis()