# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Set font for English
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 1. Data Preprocessing (consistent with original analysis, shared by all charts)
# --------------------------
# Read data (replace with your actual data path)
df = pd.read_csv('output/networks_20260329_182649.csv')

# Handle infinite throughput values
df['throughput_mbps'] = df['throughput_mbps'].replace(np.inf, 20.00)

# Content type classification
def classify_content(content_id):
    if 'telemetry' in content_id:
        return 'Telemetry Data'
    elif 'low_res_img' in content_id:
        return 'Low-Resolution Images'
    elif 'status_update' in content_id:
        return 'Status Updates'
    else:
        return 'Other Content'
df['content_type'] = df['content_id'].apply(classify_content)

# File size range division
size_bins = [0, 0.1, 1, 5, 10]
size_labels = ['0-0.1', '0.1-1', '1-5', '5-10']
df['size_range'] = pd.cut(df['file_size_MB'], bins=size_bins, labels=size_labels, right=False)

# Path length calculation
def count_path_nodes(path_str):
    try:
        return len(eval(path_str))
    except:
        return 0
df['path_length'] = df['path'].apply(count_path_nodes)

# Time interval division
time_bins = np.arange(df['time_ms'].min(), df['time_ms'].max() + 20000, 20000)
df['time_bin'] = pd.cut(df['time_ms'], bins=time_bins)

# --------------------------
# 2. Global Style Settings (ensure consistent style for all charts)
# --------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#52B788']  # Unified color scheme

# --------------------------
# Chart 1: Content Type Distribution Pie Chart (separate file)
# --------------------------
plt.figure(figsize=(10, 8))  # Individual chart size (larger and clearer)
content_counts = df['content_type'].value_counts()

# Draw pie chart
wedges, texts, autotexts = plt.pie(
    content_counts.values,
    labels=content_counts.index,
    autopct='%1.1f%%',
    colors=colors[:len(content_counts)],
    startangle=90,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # White stroke for pie chart edges, more refined
)

# Optimize text style (bold percentage labels)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title('Distribution of Network Transmission Content Types', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('pic/content_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Content type distribution pie chart saved: content_distribution.png")

# --------------------------
# Chart 2: Request vs Service Node Comparison Chart (separate file)
# --------------------------
plt.figure(figsize=(12, 8))
node_data = pd.DataFrame({
    'Request Nodes': df['node_id'].value_counts(),
    'Service Nodes': df['server_node'].value_counts()
}).fillna(0)

x = np.arange(len(node_data.index))
width = 0.35

# Draw double bar chart
bars1 = plt.bar(x - width/2, node_data['Request Nodes'], width, 
                label='Request Initiating Nodes', color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
bars2 = plt.bar(x + width/2, node_data['Service Nodes'], width, 
                label='Service Providing Nodes', color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)

# Add value labels (larger font size, clearer)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 30,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart detail optimization
plt.xlabel('Node ID', fontsize=14, fontweight='bold')
plt.ylabel('Number of Requests', fontsize=14, fontweight='bold')
plt.title('Comparison of Load Distribution between Request and Service Nodes', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, node_data.index, rotation=45, ha='right')
plt.legend(loc='upper right', frameon=True, shadow=True)  # Legend with shadow, more three-dimensional
plt.grid(True, alpha=0.3, axis='y')  # Only show y-axis grid to avoid interference
plt.tight_layout()
plt.savefig('pic/node_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Request and service node comparison chart saved: node_comparison.png")

# --------------------------
# Chart 3: Scheduling Algorithm Latency Comparison Chart (separate file)
# --------------------------
plt.figure(figsize=(10, 8))
algo_latency = df.groupby('algo')['latency_ms'].agg(['mean', 'median']).round(3)

x = np.arange(len(algo_latency))
width = 0.35

# Draw double indicator bar chart
bars1 = plt.bar(x - width/2, algo_latency['mean'], width, 
                label='Average Latency', color=colors[2], alpha=0.8, edgecolor='white', linewidth=1)
bars2 = plt.bar(x + width/2, algo_latency['median'], width, 
                label='Median Latency', color=colors[3], alpha=0.8, edgecolor='white', linewidth=1)

# Add value labels (display 3 decimal places to accurately reflect latency)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                 f'{height:.3f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart detail optimization
plt.xlabel('Scheduling Algorithm', fontsize=14, fontweight='bold')
plt.ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
plt.title('Comparison of Latency Performance of Different Scheduling Algorithms', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, algo_latency.index, rotation=45, ha='right')
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pic/algo_latency.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Scheduling algorithm latency comparison chart saved: algo_latency.png")

# --------------------------
# Chart 4: File Size Distribution Histogram (separate file)
# --------------------------
plt.figure(figsize=(10, 8))
size_counts = df['size_range'].value_counts().sort_index()
valid_sizes = size_counts[size_counts > 0]

# Draw single bar chart
bars = plt.bar(range(len(valid_sizes)), valid_sizes.values, 
               color=colors[4], alpha=0.8, edgecolor='white', linewidth=1)

# Add value labels and percentages
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height)}\n({percentage:.1f}%)',  # Line break to display quantity and percentage
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart detail optimization
plt.xlabel('File Size Range (MB)', fontsize=14, fontweight='bold')
plt.ylabel('Number of Requests', fontsize=14, fontweight='bold')
plt.title('Distribution of Network Transmission File Sizes', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(valid_sizes)), valid_sizes.index)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pic/file_size_dist.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ File size distribution histogram saved: file_size_dist.png")

# --------------------------
# Chart 5: Path Length vs Latency Relationship Chart (separate file)
# --------------------------
plt.figure(figsize=(10, 8))
path_latency = df.groupby('path_length')['latency_ms'].agg(['mean', 'count']).round(3)

x = np.arange(len(path_latency))
width = 0.6  # Wider bars to highlight comparison

# Draw main bar chart (average latency)
bars = plt.bar(x, path_latency['mean'], width, 
               color=colors[5], alpha=0.8, edgecolor='white', linewidth=1)

# Add latency value labels and request count annotations
for i, bar in enumerate(bars):
    height = bar.get_height()
    count = path_latency['count'].iloc[i]
    # Latency label
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{height:.3f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Request count annotation (below the bar)
    plt.text(bar.get_x() + bar.get_width()/2., 0.5,  # Fixed at 0.5ms height to avoid overlap
             f'Requests: {int(count)}', ha='center', va='bottom', fontsize=10, 
             color='gray', fontweight='bold')

# Chart detail optimization
plt.xlabel('Number of Nodes in Transmission Path', fontsize=14, fontweight='bold')
plt.ylabel('Average Latency (milliseconds)', fontsize=14, fontweight='bold')
plt.title('Relationship between Transmission Path Length and Average Latency', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, [f'{int(idx)} Nodes' for idx in path_latency.index])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pic/path_latency.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Path length vs latency relationship chart saved: path_latency.png")

# --------------------------
# Chart 6: Request Volume Time Trend Chart (separate file)
# --------------------------
plt.figure(figsize=(14, 8))  # Wider chart to accommodate time interval labels
time_counts = df['time_bin'].value_counts().sort_index()
x_labels = [f'{int(interval.left/1000)}-{int(interval.right/1000)}s' 
            for interval in time_counts.index]

# Draw line chart (bold line + large dots to highlight trend)
plt.plot(range(len(time_counts)), time_counts.values, 
         marker='o', linewidth=3, markersize=8, color=colors[0],
         markerfacecolor='white', markeredgecolor=colors[0], markeredgewidth=2)  # Hollow dots

# Add value labels for key nodes (only peak and valley values to avoid crowding)
max_idx = time_counts.values.argmax()
min_idx = time_counts.values.argmin()
for idx in [max_idx, min_idx]:
    plt.text(idx, time_counts.values[idx] + 10,
             f'{int(time_counts.values[idx])}', ha='center', va='bottom',
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor=colors[0], alpha=0.7, edgecolor='none'), color='white')

# Chart detail optimization
plt.xlabel('Time Interval', fontsize=14, fontweight='bold')
plt.ylabel('Number of Requests', fontsize=14, fontweight='bold')
plt.title('Time Trend of Network Transmission Request Volume (10 Minutes)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(0, len(x_labels), 2), [x_labels[i] for i in range(0, len(x_labels), 2)], 
           rotation=45, ha='right')  # Display labels every 2 intervals to avoid overlap
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pic/request_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Request volume time trend chart saved: request_trend.png")

print("\n🎉 All 6 independent charts have been generated successfully!")
print("File List:")
print("1. content_distribution.png - Content Type Distribution Pie Chart")
print("2. node_comparison.png      - Request vs Service Node Comparison Chart")
print("3. algo_latency.png         - Scheduling Algorithm Latency Comparison Chart")
print("4. file_size_dist.png       - File Size Distribution Histogram")
print("5. path_latency.png         - Path Length vs Latency Relationship Chart")
print("6. request_trend.png        - Request Volume Time Trend Chart")
