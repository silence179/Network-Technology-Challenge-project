#!/usr/bin/env python3
"""
Script to create bar plots for comparative metrics results.
Creates grouped bar charts with confidence intervals for each environment size.
"""

import argparse
from sarenv.utils.plot import create_individual_metric_plots

def main():
    parser = argparse.ArgumentParser(description='Create grouped bar plots for comparative metrics results')
    parser.add_argument('--input1', '-i1', type=str,
                       default='results/comparative_metrics_results_n5_budget300000.csv',
                       help='First input CSV file path (high budget)')
    parser.add_argument('--input2', '-i2', type=str,
                       default='results/comparative_metrics_results_n5_budget100000.csv',
                       help='Second input CSV file path (low budget)')
    parser.add_argument('--output', '-o', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--size', '-s', type=str, default='medium',
                       help='Environment size to plot (default: medium)')

    args = parser.parse_args()

    # Create plots
    print(f"\nCreating grouped plots for environment size: {args.size}")
    
    # Use the new flexible function with file paths and budget labels
    file_paths = [args.input1, args.input2]
    budget_labels = ['$60$km', '$20$km']  # 300/5 = 60, 100/5 = 20
    
    create_individual_metric_plots(file_paths, args.size, args.output, budget_labels)

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# Script to create bar plots for comparative metrics results.
# Creates grouped bar charts with confidence intervals for each environment size.
# """

# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for scientific plotting
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from pathlib import Path
# import argparse
# from scipy import stats

# # Set scientific plotting style
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("Set2")  # Scientific colorblind-friendly palette

# # Enable mathematical notation (try LaTeX, fallback to mathtext)
# try:
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.family'] = 'serif'
# except Exception:
#     # Fallback to mathtext if LaTeX is not available
#     plt.rcParams['text.usetex'] = False
#     plt.rcParams['mathtext.default'] = 'regular'

# def calculate_ci(data, confidence=0.95):
#     """Calculate confidence interval for the given data."""
#     n = len(data)
#     if n < 2:
#         return 0

#     mean = np.mean(data)
#     sem = stats.sem(data)  # Standard error of the mean
#     t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
#     ci = t_val * sem
#     return ci

# def create_individual_metric_plots(df_or_files, environment_size, output_dir="plots", budget_labels=None):
#     """
#     Create separate bar plots for each metric, saved as individual PDF files.
#     Now creates grouped bar plots with data from both budget conditions.
    
#     Args:
#         df_or_files: Either a pandas DataFrame or a list of file paths to CSV files
#         environment_size: Environment size to filter by
#         output_dir: Directory to save plots
#         budget_labels: Optional list of budget labels (if using multiple files)
#     """
#     # Handle both single DataFrame and multiple files
#     if isinstance(df_or_files, pd.DataFrame):
#         # Single DataFrame case
#         combined_df = df_or_files
#     else:
#         # Multiple files case
#         if isinstance(df_or_files, (list, tuple)):
#             file_paths = df_or_files
#         else:
#             # Single file path
#             file_paths = [df_or_files]
        
#         # Load and combine multiple files
#         dataframes = []
#         for i, file_path in enumerate(file_paths):
#             try:
#                 df = pd.read_csv(file_path)
                
#                 # Add budget condition labels
#                 if budget_labels and i < len(budget_labels):
#                     df['Budget Condition'] = budget_labels[i]
#                 elif len(file_paths) > 1:
#                     # Auto-generate labels if not provided
#                     df['Budget Condition'] = f'Condition {i+1}'
#                 else:
#                     # Single file case - no budget condition needed
#                     df['Budget Condition'] = 'Default'
                
#                 dataframes.append(df)
#             except FileNotFoundError:
#                 print(f"Error: Could not find file: {file_path}")
#                 continue
#             except Exception as e:
#                 print(f"Error loading file {file_path}: {e}")
#                 continue
        
#         if not dataframes:
#             print("Error: No valid data files found")
#             return
        
#         combined_df = pd.concat(dataframes, ignore_index=True)
    
#     # Filter data for the specific environment size
#     size_data = combined_df[combined_df['Environment Size'] == environment_size].copy()

#     if size_data.empty:
#         print(f"No data found for environment size: {environment_size}")
#         return

#     # Rename algorithms for cleaner display
#     size_data['Algorithm'] = size_data['Algorithm'].replace('RandomWalk', 'Random')

#     # Metrics to plot with their mathematical notation
#     metrics = {
#         'Likelihood Score': {'column': 'Likelihood Score', 'unit': r'$\mathcal{L}(\pi)$'},
#         'Time-Discounted Score': {'column': 'Time-Discounted Score', 'unit': r'$\mathcal{I}(\pi)$'},
#         'Victims Found (%)': {'column': 'Victims Found (%)', 'unit': r'$\mathcal{D}(\pi)$'}
#     }

#     algorithms = sorted([alg for alg in size_data['Algorithm'].unique() if alg != 'Spiral'])
#     budgets = sorted(size_data['Budget Condition'].unique())

#     # Create separate plot for each metric
#     for metric_name, metric_info in metrics.items():
#         fig, ax = plt.subplots(figsize=(8,8))

#         # Prepare data for this metric
#         n_algorithms = len(algorithms)
#         n_budgets = len(budgets)

#         # Set up grouped bar positions
#         bar_width = 0.35
#         x_positions = np.arange(n_algorithms)

#         # Colors for different budgets - use seaborn Set2 palette
#         colors = sns.color_palette("Set2", n_budgets)

#         for i, budget in enumerate(budgets):
#             means = []
#             cis = []

#             for algorithm in algorithms:
#                 alg_budget_data = size_data[
#                     (size_data['Algorithm'] == algorithm) &
#                     (size_data['Budget Condition'] == budget)
#                 ]
#                 values = alg_budget_data[metric_info['column']].values

#                 if len(values) > 0:
#                     mean_val = np.mean(values)
#                     ci_val = calculate_ci(values)
#                 else:
#                     mean_val = 0
#                     ci_val = 0

#                 means.append(mean_val)
#                 cis.append(ci_val)

#             # Position bars for this budget condition
#             bar_positions = x_positions + i * bar_width

#             bars = ax.bar(bar_positions, means, bar_width,
#                          label=f'{budget}', color=colors[i], alpha=0.8,
#                          yerr=cis, capsize=5, error_kw={'linewidth': 1.5})

#             # Add value labels
#             for bar, mean, ci in zip(bars, means, cis):
#                 if mean > 0:  # Only add label if there's data
#                     height = bar.get_height()
#                     text_y = height + ci
#                     ax.text(bar.get_x() + bar.get_width()/2., text_y,
#                            f'{mean:.3f}', ha='center', va='bottom', fontsize=17)

#         # Customize subplot
#         ax.set_ylabel(metric_info['unit'], fontsize=36, fontweight='bold')
#         ax.set_xticks(x_positions + bar_width / 2)
#         ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=26)
#         ax.tick_params(axis='y', labelsize=26)
#         ax.grid(True, alpha=0.3, axis='y')

#         # Change legend title font size
#         legend = ax.legend(fontsize=20, title="Budget", frameon=True, fancybox=True, edgecolor='black')
#         if legend.get_title() is not None:
#             legend.get_title().set_fontsize(20)

#         # Adjust y-axis limits to ensure labels are visible
#         current_ylim = ax.get_ylim()
#         all_means = []
#         all_cis = []
#         for budget in budgets:
#             for algorithm in algorithms:
#                 if algorithm == 'Spiral':
#                     continue
#                 alg_budget_data = size_data[
#                     (size_data['Algorithm'] == algorithm) &
#                     (size_data['Budget Condition'] == budget)
#                 ]
                
#                 values = alg_budget_data[metric_info['column']].values
#                 if len(values) > 0:
#                     all_means.append(np.mean(values))
#                     all_cis.append(calculate_ci(values))

#         if all_means:
#             max_value = max(all_means)
#             max_ci = max(all_cis)
#             new_ylim_top = (max_value + max_ci) * 1.15
#             ax.set_ylim(current_ylim[0], new_ylim_top)

#         plt.tight_layout()

#         # Save individual plot as PDF
#         output_path = Path(output_dir)
#         output_path.mkdir(exist_ok=True)

#         # Clean metric name for filename
#         clean_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')
#         pdf_filename = f"{clean_metric_name}_{environment_size}_grouped.pdf"
#         pdf_filepath = output_path / pdf_filename

#         plt.savefig(pdf_filepath, bbox_inches='tight', dpi=300)
#         print(f"Individual metric plot saved to: {pdf_filepath}")

#         plt.close()  # Close the figure to free memory

# def main():
#     parser = argparse.ArgumentParser(description='Create grouped bar plots for comparative metrics results')
#     parser.add_argument('--input1', '-i1', type=str,
#                        default='results/comparative_metrics_results_n5_budget300000.csv',
#                        help='First input CSV file path (high budget)')
#     parser.add_argument('--input2', '-i2', type=str,
#                        default='results/comparative_metrics_results_n5_budget100000.csv',
#                        help='Second input CSV file path (low budget)')
#     parser.add_argument('--output', '-o', type=str, default='plots',
#                        help='Output directory for plots')
#     parser.add_argument('--size', '-s', type=str, default='medium',
#                        help='Environment size to plot (default: medium)')

#     args = parser.parse_args()

#     # Create plots
#     print(f"\nCreating grouped plots for environment size: {args.size}")
    
#     # Use the new flexible function with file paths and budget labels
#     file_paths = [args.input1, args.input2]
#     budget_labels = ['$60$km', '$20$km']  # 300/5 = 60, 100/5 = 20
    
#     create_individual_metric_plots(file_paths, args.size, args.output, budget_labels)

# if __name__ == "__main__":
#     main()
