#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Visualization System
자동 시각화 시스템 - 데이터 타입에 따라 최적 플롯 생성
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for both interactive and non-interactive use
try:
    matplotlib.use('Qt5Agg')
    INTERACTIVE_AVAILABLE = True
except:
    try:
        matplotlib.use('TkAgg')
        INTERACTIVE_AVAILABLE = True
    except:
        matplotlib.use('Agg')
        INTERACTIVE_AVAILABLE = False

class AutoVisualizer:
    """Automatic visualization system that creates optimal plots based on data types"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set styling
        plt.style.use('default')
        sns.set_palette("husl")

        self.created_plots = []

    def analyze_and_visualize(self, df: pd.DataFrame, show_plots: bool = True) -> Dict[str, Any]:
        """
        Automatically analyze data and create appropriate visualizations
        """
        print("🎨 Starting automatic visualization...")

        # Analyze data structure
        analysis = self._analyze_data_structure(df)

        # Create visualizations based on data types
        visualization_results = []

        # 1. Numeric distributions (histograms)
        if analysis['numeric_columns']:
            hist_result = self._create_histograms(df, analysis['numeric_columns'], show_plots)
            visualization_results.append(hist_result)

        # 2. Correlation heatmap (if multiple numeric columns)
        if len(analysis['numeric_columns']) >= 2:
            corr_result = self._create_correlation_heatmap(df, analysis['numeric_columns'], show_plots)
            visualization_results.append(corr_result)

        # 3. Scatter plots (for numeric pairs)
        if len(analysis['numeric_columns']) >= 2:
            scatter_result = self._create_scatter_plots(df, analysis['numeric_columns'], show_plots)
            visualization_results.append(scatter_result)

        # 4. Categorical distributions (bar charts)
        if analysis['categorical_columns']:
            cat_result = self._create_categorical_plots(df, analysis['categorical_columns'], show_plots)
            visualization_results.append(cat_result)

        # 5. Box plots (numeric vs categorical)
        if analysis['numeric_columns'] and analysis['categorical_columns']:
            box_result = self._create_box_plots(df, analysis['numeric_columns'],
                                              analysis['categorical_columns'], show_plots)
            visualization_results.append(box_result)

        # 6. Summary dashboard
        dashboard_result = self._create_summary_dashboard(df, show_plots)
        visualization_results.append(dashboard_result)

        return {
            "success": True,
            "data_analysis": analysis,
            "visualizations_created": len(visualization_results),
            "visualization_results": visualization_results,
            "interactive_available": INTERACTIVE_AVAILABLE,
            "plots_saved": self.created_plots
        }

    def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure to determine visualization strategy"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'boolean_columns': boolean_cols,
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }

    def _create_histograms(self, df: pd.DataFrame, numeric_cols: List[str], show_plots: bool) -> Dict[str, Any]:
        """Create histograms for numeric columns"""

        n_cols = len(numeric_cols)
        if n_cols == 0:
            return {"error": "No numeric columns found"}

        # Calculate subplot grid
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        n_subplot_cols = min(3, n_cols)

        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold')

        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i] if n_cols > 1 else axes[0]

            # Create histogram with density curve
            ax.hist(df[col].dropna(), bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')

            # Add density curve
            try:
                df[col].dropna().plot.density(ax=ax, color='red', linewidth=2)
            except:
                pass

            ax.set_title(f'{col}', fontweight='bold')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
            ax.legend()

        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "histograms_numeric_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("📊 Histograms displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "histograms",
            "columns_plotted": numeric_cols,
            "plot_saved": str(plot_path)
        }

    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str], show_plots: bool) -> Dict[str, Any]:
        """Create correlation heatmap for numeric columns"""

        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("🔥 Correlation heatmap displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "correlation_heatmap",
            "correlation_matrix": corr_matrix.to_dict(),
            "plot_saved": str(plot_path)
        }

    def _create_scatter_plots(self, df: pd.DataFrame, numeric_cols: List[str], show_plots: bool) -> Dict[str, Any]:
        """Create scatter plots for key numeric relationships"""

        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for scatter plots"}

        # Find most correlated pairs
        corr_matrix = df[numeric_cols].corr()

        # Get upper triangle correlations
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                pairs.append({
                    'x': numeric_cols[i],
                    'y': numeric_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Take top 4 pairs or all if less than 4
        top_pairs = pairs[:min(4, len(pairs))]

        # Create subplots
        n_plots = len(top_pairs)
        n_rows = (n_plots + 1) // 2
        n_cols = min(2, n_plots)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        fig.suptitle('Key Scatter Plot Relationships', fontsize=16, fontweight='bold')

        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, pair in enumerate(top_pairs):
            ax = axes[i] if n_plots > 1 else axes[0]

            x_data = df[pair['x']].dropna()
            y_data = df[pair['y']].dropna()

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Add trend line
            if abs(pair['correlation']) > 0.1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), "r--", alpha=0.8)

            ax.set_xlabel(pair['x'])
            ax.set_ylabel(pair['y'])
            ax.set_title(f"{pair['y']} vs {pair['x']} (r={pair['correlation']:.3f})")
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "scatter_plots_relationships.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("📍 Scatter plots displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "scatter_plots",
            "relationships_plotted": top_pairs,
            "plot_saved": str(plot_path)
        }

    def _create_categorical_plots(self, df: pd.DataFrame, categorical_cols: List[str], show_plots: bool) -> Dict[str, Any]:
        """Create bar charts for categorical columns"""

        if not categorical_cols:
            return {"error": "No categorical columns found"}

        # Filter columns with reasonable number of categories
        suitable_cols = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 1 < unique_count <= 20:  # Reasonable number of categories
                suitable_cols.append(col)

        if not suitable_cols:
            return {"error": "No categorical columns with suitable number of categories"}

        n_cols = len(suitable_cols)
        n_rows = (n_cols + 1) // 2
        n_subplot_cols = min(2, n_cols)

        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(12, 5 * n_rows))
        fig.suptitle('Categorical Variable Distributions', fontsize=16, fontweight='bold')

        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(suitable_cols):
            ax = axes[i] if n_cols > 1 else axes[0]

            value_counts = df[col].value_counts()

            # Create bar plot
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                         color='lightcoral', edgecolor='black', alpha=0.7)

            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {col}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(value_counts),
                       str(value), ha='center', va='bottom')

            ax.grid(True, alpha=0.3, axis='y')

        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "categorical_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("📊 Categorical distributions displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "categorical_distributions",
            "columns_plotted": suitable_cols,
            "plot_saved": str(plot_path)
        }

    def _create_box_plots(self, df: pd.DataFrame, numeric_cols: List[str],
                         categorical_cols: List[str], show_plots: bool) -> Dict[str, Any]:
        """Create box plots showing numeric distributions by categorical variables"""

        # Find suitable categorical columns (not too many categories)
        suitable_cat_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10]

        if not suitable_cat_cols or not numeric_cols:
            return {"error": "No suitable columns for box plots"}

        # Create box plots for each numeric vs categorical combination
        combinations = []
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            for cat_col in suitable_cat_cols[:2]:  # Limit to first 2 categorical columns
                combinations.append((num_col, cat_col))

        if not combinations:
            return {"error": "No valid combinations found"}

        n_plots = len(combinations)
        n_rows = (n_plots + 1) // 2
        n_subplot_cols = min(2, n_plots)

        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(14, 6 * n_rows))
        fig.suptitle('Numeric Distributions by Categories', fontsize=16, fontweight='bold')

        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, (num_col, cat_col) in enumerate(combinations):
            ax = axes[i] if n_plots > 1 else axes[0]

            # Create box plot
            df.boxplot(column=num_col, by=cat_col, ax=ax)
            ax.set_title(f'{num_col} by {cat_col}')
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)

            # Remove automatic title
            ax.set_title(f'{num_col} by {cat_col}')

        # Remove the automatic suptitle from pandas
        fig.suptitle('Numeric Distributions by Categories', fontsize=16, fontweight='bold')

        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "box_plots_numeric_by_categorical.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("📦 Box plots displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "box_plots",
            "combinations_plotted": combinations,
            "plot_saved": str(plot_path)
        }

    def _create_summary_dashboard(self, df: pd.DataFrame, show_plots: bool) -> Dict[str, Any]:
        """Create a summary dashboard with key insights"""

        fig = plt.figure(figsize=(16, 10))

        # Data overview text
        ax1 = plt.subplot(2, 3, 1)
        ax1.axis('off')

        # Data summary
        summary_text = f"""
DATA OVERVIEW
{'='*30}
📊 Rows: {len(df):,}
📊 Columns: {len(df.columns)}
📊 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

COLUMN TYPES
{'='*30}
🔢 Numeric: {len(df.select_dtypes(include=[np.number]).columns)}
📝 Categorical: {len(df.select_dtypes(include=['object']).columns)}
📅 DateTime: {len(df.select_dtypes(include=['datetime']).columns)}
✅ Boolean: {len(df.select_dtypes(include=['bool']).columns)}

MISSING DATA
{'='*30}
🕳️ Total Missing: {df.isnull().sum().sum():,}
📊 Percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%
"""

        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        # Missing data heatmap (if any missing data)
        if df.isnull().sum().sum() > 0:
            ax2 = plt.subplot(2, 3, 2)
            missing_data = df.isnull()
            if len(df.columns) <= 20:  # Only if reasonable number of columns
                sns.heatmap(missing_data.transpose(), cbar=True, ax=ax2, cmap='Blues')
                ax2.set_title('Missing Data Pattern')
                ax2.set_xlabel('Rows')
                ax2.set_ylabel('Columns')
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, 'Too many columns\nfor missing data heatmap',
                        ha='center', va='center', transform=ax2.transAxes)

        # Data types pie chart
        ax3 = plt.subplot(2, 3, 3)
        type_counts = {
            'Numeric': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical': len(df.select_dtypes(include=['object']).columns),
            'DateTime': len(df.select_dtypes(include=['datetime']).columns),
            'Boolean': len(df.select_dtypes(include=['bool']).columns)
        }

        # Remove zero counts
        type_counts = {k: v for k, v in type_counts.items() if v > 0}

        if type_counts:
            ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax3.set_title('Column Types Distribution')

        # Numeric summary statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ax4 = plt.subplot(2, 3, 4)
            numeric_summary = df[numeric_cols].describe().transpose()

            # Create a simplified stats table
            im = ax4.imshow(numeric_summary.values, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(len(numeric_summary.columns)))
            ax4.set_xticklabels(numeric_summary.columns, rotation=45)
            ax4.set_yticks(range(len(numeric_summary.index)))
            ax4.set_yticklabels(numeric_summary.index)
            ax4.set_title('Numeric Variables Summary')

            # Add text annotations
            for i in range(len(numeric_summary.index)):
                for j in range(len(numeric_summary.columns)):
                    text = ax4.text(j, i, f'{numeric_summary.iloc[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=8)

        # Sample data preview
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')

        sample_text = "SAMPLE DATA (First 5 rows)\n" + "="*40 + "\n"
        sample_data = df.head().to_string(max_cols=5)
        if len(sample_data) > 500:  # Truncate if too long
            sample_data = sample_data[:500] + "..."

        ax5.text(0.05, 0.95, sample_text + sample_data, transform=ax5.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Data Analysis Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "summary_dashboard.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.created_plots.append(str(plot_path))

        if show_plots and INTERACTIVE_AVAILABLE:
            plt.show(block=False)
            print("📋 Summary dashboard displayed!")

        if not show_plots:
            plt.close()

        return {
            "plot_type": "summary_dashboard",
            "plot_saved": str(plot_path)
        }

def main():
    """Main function for CLI usage"""
    try:
        # Read JSON data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Initialize auto visualizer
        visualizer = AutoVisualizer()

        # Create all visualizations
        result = visualizer.analyze_and_visualize(df, show_plots=True)

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

        # Keep plots open if interactive
        if INTERACTIVE_AVAILABLE:
            input("Press Enter to close all plots and continue...")
            plt.close('all')

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "auto_visualization"
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()