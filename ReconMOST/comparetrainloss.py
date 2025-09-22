import pandas as pd
import matplotlib.pyplot as plt

# --- 请在这里修改您的文件路径 ---
file1 = '/hdd/SunZL/data/train_logs_raw_FIO1and2_fillna/progress.csv'
file2 = '/hdd/SunZL/data/train_logs_FIO1and2_Laplace_nearest_s2d_raw/progress.csv'

# 加载 CSV 文件
# 如果您的 CSV 文件没有标题行，请取消下面两行代码的注释
# columns = ['grad_norm', 'lg_loss_scale', 'loss', 'loss_q0', 'loss_q1', 'loss_q3', 'mse', 'mse_q0', 'mse_q1', 'mse_q3', 'samples', 'step', 'loss_q2', 'mse_q2']
# df1 = pd.read_csv(file1, names=columns)
# df2 = pd.read_csv(file2, names=columns)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


# 创建图表
plt.figure(figsize=(10, 6))

# 绘制第一个文件的散点图
plt.scatter(df1.index, df1['loss'], label='FIO1and2_fillna_nearest_s2d', alpha=0.6)

# 绘制第二个文件的散点图
plt.scatter(df2.index, df2['loss'], label='FIO1and2_Laplace_nearest_s2d_raw', alpha=0.6)

# 将纵轴设置为对数坐标
plt.yscale('log')

# 添加标签和标题
plt.xlabel('Index')
plt.ylabel('Loss (log scale)')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('loss_comparison_scatter.png')

print("散点图已保存为 loss_comparison_scatter.png")