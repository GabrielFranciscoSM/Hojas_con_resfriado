#!/usr/bin/env python3
"""
Script to compare classification models based on their metrics JSON files.
Generates comparison tables and visualizations.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define paths to metrics files
BASE_DIR = Path(__file__).parent
METRICS_FILES = [
    BASE_DIR / "efficientnet" / "efficientnet_metrics.json",
    BASE_DIR / "maxvit" / "maxvit_metrics.json",
    BASE_DIR / "mobilenet" / "mobilenet_metrics.json",
]


def load_metrics(file_path: Path) -> dict:
    """Load metrics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_comparison_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Create a DataFrame comparing overall metrics across models."""
    data = []
    for metrics in metrics_list:
        data.append({
            'Model': metrics['model'],
            'Test Accuracy (%)': round(metrics['test_accuracy'], 2),
            'Test Loss': round(metrics['test_loss'], 4),
            'Precision (Macro)': round(metrics['precision_macro'], 4),
            'Recall (Macro)': round(metrics['recall_macro'], 4),
            'F1-Score (Macro)': round(metrics['f1_macro'], 4),
        })
    return pd.DataFrame(data)


def create_per_class_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Create a DataFrame with per-class metrics for all models."""
    data = []
    for metrics in metrics_list:
        model_name = metrics['model']
        for class_name, class_metrics in metrics['per_class'].items():
            data.append({
                'Model': model_name,
                'Class': class_name,
                'Precision': round(class_metrics['precision'], 4),
                'Recall': round(class_metrics['recall'], 4),
                'F1-Score': round(class_metrics['f1'], 4),
                'Support': class_metrics['support'],
            })
    return pd.DataFrame(data)


def plot_overall_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar plots comparing overall metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.6
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Accuracy comparison
    axes[0].bar(x, df['Test Accuracy (%)'], width, color=colors)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].set_ylim([90, 101])
    for i, v in enumerate(df['Test Accuracy (%)']):
        axes[0].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
    
    # F1-Score comparison
    axes[1].bar(x, df['F1-Score (Macro)'], width, color=colors)
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score (Macro) Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylim([0.90, 1.01])
    for i, v in enumerate(df['F1-Score (Macro)']):
        axes[1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10)
    
    # Test Loss comparison
    axes[2].bar(x, df['Test Loss'], width, color=colors)
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Test Loss Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15, ha='right')
    for i, v in enumerate(df['Test Loss']):
        axes[2].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_dir / 'overall_comparison.png'}")


def plot_per_class_comparison(df: pd.DataFrame, output_dir: Path):
    """Create grouped bar plots for per-class metrics comparison."""
    classes = df['Class'].unique()
    models = df['Model'].unique()
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x = np.arange(len(classes))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            values = [model_data[model_data['Class'] == c][metric].values[0] for c in classes]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i])
        
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim([0.90, 1.02])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_dir / 'per_class_comparison.png'}")


def plot_radar_chart(df: pd.DataFrame, output_dir: Path):
    """Create a radar chart comparing models across all macro metrics."""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Low Loss']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Normalize metrics for radar chart (0-1 scale)
    models = df['Model'].tolist()
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Get max loss for normalization
    max_loss = df['Test Loss'].max()
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [
            row['Test Accuracy (%)'] / 100,
            row['Precision (Macro)'],
            row['Recall (Macro)'],
            row['F1-Score (Macro)'],
            1 - (row['Test Loss'] / max_loss) if max_loss > 0 else 1,  # Invert loss (lower is better)
        ]
        values += values[:1]  # Close the polygon
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
    ax.set_xticklabels(categories)
    ax.set_ylim([0.9, 1.0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Comparison Radar Chart', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_dir / 'radar_comparison.png'}")


def print_summary(df_overall: pd.DataFrame, df_per_class: pd.DataFrame):
    """Print a formatted summary of the comparison."""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\n📈 OVERALL METRICS:")
    print("-" * 80)
    print(df_overall.to_string(index=False))
    
    print("\n\n📊 PER-CLASS METRICS:")
    print("-" * 80)
    for model in df_per_class['Model'].unique():
        print(f"\n{model}:")
        model_data = df_per_class[df_per_class['Model'] == model][['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
        print(model_data.to_string(index=False))
    
    # Determine the best model
    print("\n\n🏆 SUMMARY:")
    print("-" * 80)
    best_accuracy_idx = df_overall['Test Accuracy (%)'].idxmax()
    best_model = df_overall.loc[best_accuracy_idx, 'Model']
    best_accuracy = df_overall.loc[best_accuracy_idx, 'Test Accuracy (%)']
    
    print(f"✅ Best Accuracy: {best_model} ({best_accuracy:.2f}%)")
    
    lowest_loss_idx = df_overall['Test Loss'].idxmin()
    lowest_loss_model = df_overall.loc[lowest_loss_idx, 'Model']
    lowest_loss = df_overall.loc[lowest_loss_idx, 'Test Loss']
    print(f"✅ Lowest Loss: {lowest_loss_model} ({lowest_loss:.4f})")
    
    best_f1_idx = df_overall['F1-Score (Macro)'].idxmax()
    best_f1_model = df_overall.loc[best_f1_idx, 'Model']
    best_f1 = df_overall.loc[best_f1_idx, 'F1-Score (Macro)']
    print(f"✅ Best F1-Score: {best_f1_model} ({best_f1:.4f})")
    
    print("\n" + "=" * 80)


def save_comparison_csv(df_overall: pd.DataFrame, df_per_class: pd.DataFrame, output_dir: Path):
    """Save comparison results to CSV files."""
    df_overall.to_csv(output_dir / 'overall_comparison.csv', index=False)
    df_per_class.to_csv(output_dir / 'per_class_comparison.csv', index=False)
    print(f"✅ Saved: {output_dir / 'overall_comparison.csv'}")
    print(f"✅ Saved: {output_dir / 'per_class_comparison.csv'}")


def main():
    """Main function to run the model comparison."""
    print("🔄 Loading metrics from JSON files...")
    
    # Load all metrics
    metrics_list = []
    for file_path in METRICS_FILES:
        if file_path.exists():
            metrics = load_metrics(file_path)
            metrics_list.append(metrics)
            print(f"  ✅ Loaded: {file_path.name}")
        else:
            print(f"  ⚠️  Not found: {file_path}")
    
    if len(metrics_list) < 2:
        print("❌ Error: Need at least 2 model metrics files to compare.")
        return
    
    # Create DataFrames
    df_overall = create_comparison_dataframe(metrics_list)
    df_per_class = create_per_class_dataframe(metrics_list)
    
    # Print summary
    print_summary(df_overall, df_per_class)
    
    # Create output directory
    output_dir = BASE_DIR / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV files
    save_comparison_csv(df_overall, df_per_class, output_dir)
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_overall_comparison(df_overall, output_dir)
    plot_per_class_comparison(df_per_class, output_dir)
    plot_radar_chart(df_overall, output_dir)
    
    print("\n✅ Comparison complete! Check the 'comparison_results' folder for outputs.")


if __name__ == "__main__":
    main()
