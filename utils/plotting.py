"""
Plotting utilities for seed node identification experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plotting Configuration ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.autolayout': True
})


def save_plot(fig, filename, folder="/home/bvthach/erdos renyi/plots"):
    """Save a figure to disk as EPS."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{filename}.eps")
    fig.savefig(filepath, format='eps', dpi=300, bbox_inches='tight')
    print(f"📸 Đã lưu biểu đồ tại: {filepath}")


def plot_bar_chart(metrics_dict, title, ylabel, filename):
    """
    Plot a grouped bar chart (Overall Results).

    Args:
        metrics_dict: dict of the form
            {'NetFill': {'Precision': 0.8, 'Recall': 0.7, 'F1': 0.75}, ...}
        title: chart title string
        ylabel: y-axis label string
        filename: output filename (without extension)
    """
    algorithms = list(metrics_dict.keys())
    metrics_names = list(metrics_dict[algorithms[0]].keys())

    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, algo in enumerate(algorithms):
        values = [metrics_dict[algo][m] for m in metrics_names]
        offset = (i - len(algorithms) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=algo, edgecolor='black')

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    save_plot(fig, filename)


def plot_horizontal_runtime_chart(runtime_dict, xlabel, filename, use_log_scale=True):
    """
    Plot a horizontal bar chart showing runtime comparison.

    Args:
        runtime_dict: dict of the form {'NetFill': 0.5, 'BP': 1.2, 'ILP': 120.0}
        xlabel: x-axis label string
        filename: output filename (without extension)
        use_log_scale: whether to use log scale on the x-axis
    """
    algorithms = list(runtime_dict.keys())
    times = list(runtime_dict.values())

    algorithms.reverse()
    times.reverse()

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {'NetFill': '#e74c3c', 'BP': '#2ecc71', 'ILP': '#3498db'}
    bar_colors = [colors.get(algo, 'gray') for algo in algorithms]

    bars = ax.barh(algorithms, times, color=bar_colors, edgecolor='black', height=0.5, zorder=3)

    for bar in bars:
        xval = bar.get_width()
        offset = xval * 0.15 if use_log_scale else max(times) * 0.02
        ax.text(xval + offset, bar.get_y() + bar.get_height() / 2,
                f'{xval:.2f}s', va='center', ha='left', fontweight='bold', fontsize=12)

    if use_log_scale:
        ax.set_xscale('log')
        xlabel += ' (Log Scale)'

    ax.set_xlabel(xlabel, fontweight='bold')

    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_color('#e0e0e0')
    ax.spines['right'].set_color('#e0e0e0')

    ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.2)

    save_plot(fig, filename)
