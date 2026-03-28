
"""
无人机通信网络数据分析工具
功能：提供时间趋势分析、分组对比分析、分布与相关性分析三大核心功能
适用数据：需包含time_ms、latency_ms、download_time、throughput_mbps等核心字段的CSV数据
作者：数据分析工具生成
日期：2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局配置 --------------------------
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 颜色与标签映射（可根据需求修改）
COLOR_MAP = {
    'latency_ms': '#2E86AB',       # 蓝色：延迟
    'download_time': '#A23B72',    # 紫红色：下载时间
    'throughput_mbps': '#F18F01',  # 橙色：吞吐量
    'file_size_MB': '#C73E1D'      # 红色：文件大小
}

LABEL_MAP = {
    'latency_ms': '延迟 (ms)',
    'download_time': '下载时间 (ms)',
    'throughput_mbps': '吞吐量 (Mbps)',
    'file_size_MB': '文件大小 (MB)',
    'time_ms': '时间 (ms)',
    'time_s': '时间 (s)'
}

GROUP_LABEL_MAP = {
    'server_node': '无人机节点',
    'algo': '算法类型',
    'cache_status': '缓存状态',
    'http_code': 'HTTP状态码'
}

# -------------------------- 核心分析函数 --------------------------
def plot_time_trend_analysis(df, target_cols=['latency_ms', 'download_time'], 
                             filter_node=None, time_unit='ms', save_path='/mnt/time_trend.png',
                             window_ratio=1/50):
    """
    时间趋势分析图表：展示目标指标随时间的变化，支持移动平均线
    
    参数：
    df: DataFrame - 输入数据（需包含time_ms列）
    target_cols: list - 需分析的指标列（如['latency_ms', 'throughput_mbps']）
    filter_node: str - 筛选特定无人机节点（如'UAV_01'，None表示全部）
    time_unit: str - 时间单位（'ms'=毫秒，'s'=秒）
    save_path: str - 图表保存路径
    window_ratio: float - 移动平均窗口比例（数据量的比例，如1/50表示窗口为数据量的1/50）
    
    返回：
    None - 直接保存图表文件
    """
    # 1. 数据预处理
    df_plot = df.copy()
    
    # 时间单位转换
    if time_unit == 's':
        df_plot['time'] = df_plot['time_ms'] / 1000
        x_label = LABEL_MAP['time_s']
    else:
        df_plot['time'] = df_plot['time_ms']
        x_label = LABEL_MAP['time_ms']
    
    # 筛选节点数据
    if filter_node is not None:
        if 'server_node' not in df_plot.columns:
            raise ValueError("数据中缺少'server_node'列，无法筛选节点")
        df_plot = df_plot[df_plot['server_node'] == filter_node]
        title_suffix = f'（{filter_node}）'
    else:
        title_suffix = '（所有节点）'
    
    # 检查目标列是否存在
    for col in target_cols:
        if col not in df_plot.columns:
            raise ValueError(f"数据中缺少目标列：{col}")
    
    # 2. 创建图表
    fig, axes = plt.subplots(len(target_cols), 1, figsize=(14, 4*len(target_cols)), sharex=True)
    if len(target_cols) == 1:
        axes = [axes]  # 统一列表处理逻辑
    
    # 3. 绘制每个指标的趋势
    for i, col in enumerate(target_cols):
        # 按时间排序
        df_sorted = df_plot.sort_values('time').reset_index(drop=True)
        valid_data = df_sorted[col].dropna()
        
        # 绘制原始数据趋势（降低透明度避免视觉拥挤）
        axes[i].plot(df_sorted['time'], df_sorted[col], 
                    color=COLOR_MAP.get(col, '#888888'), alpha=0.5, linewidth=0.6, label='原始数据')
        
        # 计算并绘制移动平均线
        window_size = max(1, int(len(df_sorted) * window_ratio))
        df_sorted[f'{col}_ma'] = df_sorted[col].rolling(window=window_size, center=True).mean()
        axes[i].plot(df_sorted['time'], df_sorted[f'{col}_ma'], 
                    color=COLOR_MAP.get(col, '#888888'), linewidth=2, 
                    label=f'移动平均（窗口={window_size}）')
        
        # 图表样式设置
        axes[i].set_ylabel(LABEL_MAP.get(col, col), fontsize=11)
        axes[i].set_title(f'{LABEL_MAP.get(col, col)}时间趋势{title_suffix}', fontsize=12, pad=10)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].legend(fontsize=10, loc='upper right')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    # 4. 统一x轴设置
    axes[-1].set_xlabel(x_label, fontsize=11)
    axes[-1].xaxis.set_major_locator(plt.MaxNLocator(10))  # 最多显示10个刻度
    
    # 5. 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 时间趋势图表已保存：{save_path}")


def plot_group_comparison(df, group_col='server_node', target_col='latency_ms', 
                          plot_type='boxplot', save_path='/mnt/group_comparison.png',
                          show_value=True):
    """
    分组对比分析图表：支持箱线图（分布）和柱状图（均值±标准差）
    
    参数：
    df: DataFrame - 输入数据
    group_col: str - 分组列（如'server_node'、'algo'）
    target_col: str - 目标指标列（如'latency_ms'）
    plot_type: str - 图表类型（'boxplot'=箱线图，'bar'=柱状图）
    save_path: str - 图表保存路径
    show_value: bool - 柱状图是否显示数值标签（仅plot_type='bar'时生效）
    
    返回：
    None - 直接保存图表文件
    """
    # 1. 数据校验
    if group_col not in df.columns:
        raise ValueError(f"数据中缺少分组列：{group_col}")
    if target_col not in df.columns:
        raise ValueError(f"数据中缺少目标指标列：{target_col}")
    
    # 过滤无效分组（空值）
    df_plot = df.dropna(subset=[group_col, target_col])
    valid_groups = df_plot[group_col].unique()
    
    # 分组数量限制（避免图表拥挤）
    if len(valid_groups) > 12:
        print(f"⚠️  分组数量过多（{len(valid_groups)}个），建议筛选后分析（如取前10个分组）")
        valid_groups = valid_groups[:10]
        df_plot = df_plot[df_plot[group_col].isin(valid_groups)]
    
    # 2. 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 3. 绘制分组图
    group_name = GROUP_LABEL_MAP.get(group_col, group_col)
    target_name = LABEL_MAP.get(target_col, target_col)
    
    if plot_type == 'boxplot':
        # 箱线图：展示分布特征（中位数、四分位、异常值）
        box_data = [df_plot[df_plot[group_col] == g][target_col] for g in valid_groups]
        bp = ax.boxplot(box_data, labels=valid_groups, patch_artist=True,
                        boxprops=dict(facecolor='#E8F4FD', alpha=0.8, edgecolor=COLOR_MAP.get(target_col, '#888888')),
                        medianprops=dict(color=COLOR_MAP.get(target_col, '#888888'), linewidth=2),
                        whiskerprops=dict(color=COLOR_MAP.get(target_col, '#888888'), linewidth=1.2),
                        capprops=dict(color=COLOR_MAP.get(target_col, '#888888'), linewidth=1.2))
        
        ax.set_ylabel(target_name, fontsize=11)
        ax.set_title(f'{group_name} vs {target_name} 分布对比', fontsize=12, pad=10)
    
    elif plot_type == 'bar':
        # 柱状图：展示均值±标准差
        group_stats = df_plot.groupby(group_col)[target_col].agg(['mean', 'std', 'count']).reset_index()
        group_stats = group_stats[group_stats[group_col].isin(valid_groups)]
        
        x_pos = np.arange(len(group_stats))
        bars = ax.bar(x_pos, group_stats['mean'], yerr=group_stats['std'],
                      capsize=5, color=COLOR_MAP.get(target_col, '#A23B72'), 
                      alpha=0.8, edgecolor='#8B2C69', linewidth=1)
        
        # 添加数值标签（均值）
        if show_value:
            for i, (bar, mean_val) in enumerate(zip(bars, group_stats['mean'])):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (bar.get_height()*0.01),
                        f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_stats[group_col], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(f'{target_name}（均值±标准差）', fontsize=11)
        ax.set_title(f'{group_name} vs {target_name} 均值对比', fontsize=12, pad=10)
    
    else:
        raise ValueError("plot_type仅支持'boxplot'或'bar'")
    
    # 4. 通用样式设置
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 5. 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分组对比图表已保存：{save_path}")


def plot_distribution_correlation(df, dist_cols=['latency_ms', 'file_size_MB'], 
                                  corr_cols=['latency_ms', 'download_time', 'file_size_MB'], 
                                  save_path='/mnt/distribution_correlation.png',
                                  bins_ratio=1/50):
    """
    分布与相关性分析图表：左侧展示指标分布，右侧展示相关性热力图
    
    参数：
    df: DataFrame - 输入数据
    dist_cols: list - 需展示分布的指标列（建议1-3个）
    corr_cols: list - 需计算相关性的指标列（仅数值型）
    save_path: str - 图表保存路径
    bins_ratio: float - 直方图bins数量比例（数据量的比例，如1/50）
    
    返回：
    None - 直接保存图表文件
    """
    # 1. 数据预处理
    # 过滤分布分析数据
    df_dist = df[dist_cols].dropna()
    for col in dist_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少分布分析列：{col}")
    
    # 过滤相关性分析数据（仅保留数值型）
    df_corr = df[corr_cols].select_dtypes(include=[np.number]).dropna()
    for col in corr_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少相关性分析列：{col}")
    
    # 2. 创建图表布局
    n_dist = len(dist_cols)
    fig = plt.figure(figsize=(14, 4*n_dist))
    
    # 3. 绘制分布子图（直方图+核密度估计）
    for i, col in enumerate(dist_cols):
        ax_dist = plt.subplot(n_dist, 2, 2*i + 1)
        
        # 计算bins数量
        bins = max(10, min(50, int(len(df_dist[col]) * bins_ratio)))
        
        # 绘制直方图
        ax_dist.hist(df_dist[col], bins=bins, color=COLOR_MAP.get(col, '#F18F01'), 
                    alpha=0.6, edgecolor=COLOR_MAP.get(col, '#D97706'), linewidth=1)
        
        # 绘制核密度估计曲线
        df_dist[col].plot.kde(ax=ax_dist, color=COLOR_MAP.get(col, '#C2410C'), 
                             linewidth=2, label='核密度曲线')
        
        # 样式设置
        col_name = LABEL_MAP.get(col, col)
        ax_dist.set_xlabel(col_name, fontsize=10)
        ax_dist.set_ylabel('频数', fontsize=10)
        ax_dist.set_title(f'{col_name} 分布特征', fontsize=11, pad=8)
        ax_dist.grid(True, alpha=0.3, linestyle='--')
        ax_dist.legend(fontsize=9)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
    
    # 4. 绘制相关性热力图（仅当相关性列数≥2时）
    if len(corr_cols) >= 2:
        ax_corr = plt.subplot(n_dist, 2, 2)
        
        # 计算相关系数矩阵
        corr_matrix = df_corr.corr().round(2)
        
        # 绘制热力图
        im = ax_corr.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # 添加数值标签
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                ax_corr.text(j, i, corr_matrix.iloc[i, j],
                            ha="center", va="center", color=text_color, 
                            fontsize=10, fontweight='bold')
        
        # 样式设置
        corr_labels = [LABEL_MAP.get(col, col) for col in corr_cols]
        ax_corr.set_xticks(range(len(corr_cols)))
        ax_corr.set_yticks(range(len(corr_cols)))
        ax_corr.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=10)
        ax_corr.set_yticklabels(corr_labels, fontsize=10)
        ax_corr.set_title('指标相关性热力图', fontsize=11, pad=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax_corr, shrink=0.8)
        cbar.set_label('相关系数', fontsize=10)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    
    # 5. 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分布与相关性图表已保存：{save_path}")


def load_and_check_data(file_path):
    """
    数据加载与基础校验函数：确保数据格式正确，返回基础信息
    
    参数：
    file_path: str - 数据文件路径（CSV格式）
    
    返回：
    df: DataFrame - 加载后的DataFrame
    """
    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 成功加载数据：{file_path}")
        print(f"📊 数据规模：{df.shape[0]}行 × {df.shape[1]}列")
    except Exception as e:
        raise ValueError(f"数据加载失败：{str(e)}")
    
    # 基础校验
    required_cols = ['time_ms', 'latency_ms', 'download_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  数据缺少核心列：{missing_cols}，部分分析功能可能受限")
    else:
        print("✅ 数据包含所有核心分析列")
    
    # 缺失值统计
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if len(missing_stats) > 0:
        print("⚠️  缺失值统计：")
        for col, cnt in missing_stats.items():
            print(f"  - {col}: {cnt}个（{cnt/df.shape[0]*100:.1f}%）")
    else:
        print("✅ 数据无缺失值")
    
    return df


# -------------------------- 示例调用代码 --------------------------
if __name__ == "__main__":
    """
    示例：完整的数据分析流程
    使用前需修改：
    1. data_path：数据文件路径
    2. 各函数的save_path：图表保存路径
    """
    # 1. 加载数据
    data_path = "output/networks.csv"  # 请替换为你的数据路径
    df = load_and_check_data(data_path)
    
    # 2. 1 时间趋势分析：UAV_01的延迟和下载时间（秒单位）
    plot_time_trend_analysis(
        df=df,
        target_cols=['latency_ms', 'download_time'],
        filter_node='UAV_01',
        time_unit='s',
        save_path='output/uav01_time_trend.png',
        window_ratio=1/100  # 更小的窗口比例（更灵敏的趋势）
    )
    
    # 2. 2 时间趋势分析：所有节点的吞吐量（毫秒单位）
    plot_time_trend_analysis(
        df=df,
        target_cols=['throughput_mbps'],
        filter_node=None,
        time_unit='ms',
        save_path='output/all_nodes_throughput_trend.png'
    )
    
    # 3. 1 分组对比：不同无人机节点的延迟分布（箱线图）
    plot_group_comparison(
        df=df,
        group_col='server_node',
        target_col='latency_ms',
        plot_type='boxplot',
        save_path='output/node_latency_boxplot.png'
    )
    
    # 3. 2 分组对比：不同算法的下载时间均值（柱状图）
    plot_group_comparison(
        df=df,
        group_col='algo',
        target_col='download_time',
        plot_type='bar',
        save_path='output/algo_downloadtime_bar.png',
        show_value=True  # 显示均值数值
    )
    
    # 4. 分布与相关性分析：延迟、文件大小、下载时间
    plot_distribution_correlation(
        df=df,
        dist_cols=['latency_ms', 'file_size_MB'],
        corr_cols=['latency_ms', 'download_time', 'file_size_MB'],
        save_path='output/dist_corr_analysis.png'
    )
    
    print("\n🎉 所有分析图表生成完成！")
