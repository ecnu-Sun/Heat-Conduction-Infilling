import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import pandas as pd
from scipy import stats
# matplotlib.rcParams['font.family'] = 'Ubuntu'
matplotlib.rcParams['axes.unicode_minus'] = False

class LossAnalyzer:
    def __init__(self, file1_path, file2_path):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.data1 = self.parse_log_file(file1_path)
        self.data2 = self.parse_log_file(file2_path)
        
    def parse_log_file(self, filepath):
        """解析日志文件，提取所有的total_loss数据"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 匹配采样名称和对应的total_loss
        pattern = r'sampling (EN\.\d+\.\d+\.\d+\.f\.analysis\.g\d+\.\d+).*?total_loss: \[([\d\.\s]+)\]'
        matches = re.findall(pattern, content, re.DOTALL)
        
        data = {}
        for sample_name, loss_str in matches:
            # 将字符串转换为浮点数数组
            losses = [float(x) for x in loss_str.split()]
            data[sample_name] = np.array(losses)
            
        return data
    
    def analyze_first_n_layers(self, n=10):
        """分析前n层的loss"""
        print(f"\n=== First {n} Layers Loss Analysis ===\n")
        
        # 提取前n层的数据
        file1_losses = np.array([loss[:n] for loss in self.data1.values()])
        file2_losses = np.array([loss[:n] for loss in self.data2.values()])
        
        # 计算统计量
        stats1 = {
            'mean': np.mean(file1_losses, axis=0),
            'std': np.std(file1_losses, axis=0),
            'min': np.min(file1_losses, axis=0),
            'max': np.max(file1_losses, axis=0)
        }
        
        stats2 = {
            'mean': np.mean(file2_losses, axis=0),
            'std': np.std(file2_losses, axis=0),
            'min': np.min(file2_losses, axis=0),
            'max': np.max(file2_losses, axis=0)
        }
        
        # 打印统计结果
        print(f"File 1 ({Path(self.file1_path).name}):")
        print(f"Mean: {stats1['mean']}")
        print(f"Std: {stats1['std']}")
        print(f"Min: {stats1['min']}")
        print(f"Max: {stats1['max']}")
        
        print(f"\nFile 2 ({Path(self.file2_path).name}):")
        print(f"Mean: {stats2['mean']}")
        print(f"Std: {stats2['std']}")
        print(f"Min: {stats2['min']}")
        print(f"Max: {stats2['max']}")
        
        # 计算差异
        mean_diff = stats2['mean'] - stats1['mean']
        relative_diff = (mean_diff / stats1['mean']) * 100
        
        print(f"\nDifference Analysis:")
        print(f"Mean difference: {mean_diff}")
        print(f"Relative difference (%): {relative_diff}")
        
        return stats1, stats2
    
    def plot_comparison(self, n=10):
        """绘制前n层的对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'First {n} Layers Loss Comparison Analysis', fontsize=16)
        
        # 1. 平均值对比
        ax1 = axes[0, 0]
        file1_means = []
        file2_means = []
        
        for loss in self.data1.values():
            file1_means.append(loss[:n])
        for loss in self.data2.values():
            file2_means.append(loss[:n])
            
        file1_mean = np.mean(file1_means, axis=0)
        file2_mean = np.mean(file2_means, axis=0)
        
        layers = np.arange(1, n+1)
        ax1.plot(layers, file1_mean, 'b-o', label='File 1', linewidth=2)
        ax1.plot(layers, file2_mean, 'r-s', label='File 2', linewidth=2)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Average Loss Comparison by Layer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱线图对比
        ax2 = axes[0, 1]
        data_for_box = []
        labels = []
        
        for i in range(n):
            data_for_box.append([loss[i] for loss in self.data1.values()])
            labels.append(f'L{i+1}-F1')
            data_for_box.append([loss[i] for loss in self.data2.values()])
            labels.append(f'L{i+1}-F2')
        
        ax2.boxplot(data_for_box[:20], labels=labels[:20])  # 只显示前10层
        ax2.set_xlabel('Layer-File')
        ax2.set_ylabel('Loss Distribution')
        ax2.set_title('First 10 Layers Loss Distribution Boxplot')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 相对差异图
        ax3 = axes[1, 0]
        relative_diff = ((file2_mean - file1_mean) / file1_mean) * 100
        ax3.bar(layers, relative_diff, color=['green' if x < 0 else 'red' for x in relative_diff])
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Relative Difference (%)')
        ax3.set_title('File 2 vs File 1 Loss Change Percentage')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. 热力图
        ax4 = axes[1, 1]
        # 创建样本×层的矩阵
        sample_names = list(self.data1.keys())
        diff_matrix = []
        
        for sample in sample_names:
            if sample in self.data2:
                diff = self.data2[sample][:n] - self.data1[sample][:n]
                diff_matrix.append(diff)
        
        im = ax4.imshow(diff_matrix, aspect='auto', cmap='RdBu_r')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Sample')
        ax4.set_title('Sample-Layer Loss Difference Heatmap')
        ax4.set_xticks(range(n))
        ax4.set_xticklabels(range(1, n+1))
        ax4.set_yticks(range(len(sample_names)))
        ax4.set_yticklabels([s.split('.')[-1] for s in sample_names])
        plt.colorbar(im, ax=ax4, label='Loss Difference')
        
        plt.tight_layout()
        plt.savefig(f'loss_comparison_n_nino{n}.png', dpi=300)
    
    def statistical_test(self, n=10):
        """对前n层进行统计检验"""
        print(f"\n=== Statistical Test Results (First {n} Layers) ===\n")
        
        for i in range(n):
            layer_data1 = [loss[i] for loss in self.data1.values()]
            layer_data2 = [loss[i] for loss in self.data2.values()]
            
            # T检验
            t_stat, p_value = stats.ttest_ind(layer_data1, layer_data2)
            
            print(f"Layer {i+1}:")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significance: {'Significant' if p_value < 0.05 else 'Not significant'} (α=0.05)")
    
    def generate_report(self, n=10):
        """生成完整的分析报告"""
        print("\n" + "="*50)
        print("Loss Comparison Analysis Report")
        print("="*50)
        
        print(f"\nFile 1: {Path(self.file1_path).name}")
        print(f"File 2: {Path(self.file2_path).name}")
        print(f"Sample count: File1={len(self.data1)}, File2={len(self.data2)}")
        
        # 基础统计分析
        stats1, stats2 = self.analyze_first_n_layers(n)
        
        # 统计检验
        self.statistical_test(n)
        
        # 找出变化最大的层
        mean_diff = stats2['mean'] - stats1['mean']
        max_increase_layer = np.argmax(mean_diff) + 1
        max_decrease_layer = np.argmin(mean_diff) + 1
        
        print(f"\n=== Key Findings ===")
        print(f"Layer with maximum loss increase: Layer {max_increase_layer} (increased by {mean_diff[max_increase_layer-1]:.4f})")
        print(f"Layer with maximum loss decrease: Layer {max_decrease_layer} (decreased by {abs(mean_diff[max_decrease_layer-1]):.4f})")
        
        # 整体趋势
        overall_mean1 = np.mean([np.mean(loss[:n]) for loss in self.data1.values()])
        overall_mean2 = np.mean([np.mean(loss[:n]) for loss in self.data2.values()])
        overall_change = ((overall_mean2 - overall_mean1) / overall_mean1) * 100
        
        print(f"\nFirst {n} layers overall average loss:")
        print(f"  File 1: {overall_mean1:.4f}")
        print(f"  File 2: {overall_mean2:.4f}")
        print(f"  Change: {overall_change:.2f}%")
        
        # 绘制对比图
        self.plot_comparison(n)

# 使用示例
if __name__ == "__main__":
    # 请替换为您的实际文件路径
    file1_path = "/data/coding/ReconMOST/sampling_results_200000_ema_sample1_withtime_FIO_batchsize4_constlr/log.txt"
    file2_path = "/data/coding/ReconMOST/sampling_results_200000_ema_sample1_FIO_batchsize4_constlr_nino/log.txt"
    
    # 创建分析器实例
    analyzer = LossAnalyzer(file1_path, file2_path)
    
    # 生成完整报告（默认分析前10层）
    analyzer.generate_report(n=10)
    
    # 如果需要分析其他层数，可以修改n的值
    # analyzer.generate_report(n=15)