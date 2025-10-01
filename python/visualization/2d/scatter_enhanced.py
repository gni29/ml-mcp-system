#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Scatter Plot Visualization Module
산점도 시각화 모듈 (개선된 버전)
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for both interactive and non-interactive modes
try:
    # Try to use Qt backend for interactive plots
    matplotlib.use('Qt5Agg')
    INTERACTIVE_AVAILABLE = True
except:
    try:
        matplotlib.use('TkAgg')
        INTERACTIVE_AVAILABLE = True
    except:
        # Fallback to non-interactive backend
        matplotlib.use('Agg')
        INTERACTIVE_AVAILABLE = False

class ScatterPlotVisualizer:
    """Enhanced Scatter Plot Visualization with multiple features"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_scatter_plot(self, df: pd.DataFrame, x_col: str = None, y_col: str = None,
                           color_col: str = None, size_col: str = None,
                           show_plot: bool = True, save_plot: bool = True) -> Dict[str, Any]:
        """
        Create enhanced scatter plot with optional color and size encoding

        Args:
            df: DataFrame with data
            x_col: X-axis column (auto-selected if None)
            y_col: Y-axis column (auto-selected if None)
            color_col: Column to encode as color (optional)
            size_col: Column to encode as size (optional)
            show_plot: Whether to display plot interactively
            save_plot: Whether to save plot to file
        """

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return {
                "error": "최소 2개 이상의 숫자형 열이 필요합니다",
                "available_numeric_columns": numeric_cols
            }

        # Auto-select columns if not provided
        x_col = x_col or numeric_cols[0]
        y_col = y_col or numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

        # Validate columns exist
        if x_col not in df.columns or y_col not in df.columns:
            return {
                "error": f"Columns {x_col} or {y_col} not found",
                "available_columns": list(df.columns)
            }

        # Prepare data
        plot_data = df[[x_col, y_col]].copy()

        # Add color and size columns if specified
        if color_col and color_col in df.columns:
            plot_data[color_col] = df[color_col]
        if size_col and size_col in df.columns:
            plot_data[size_col] = df[size_col]

        # Remove rows with NaN in essential columns
        plot_data = plot_data.dropna(subset=[x_col, y_col])

        if len(plot_data) == 0:
            return {"error": "유효한 데이터 포인트가 없습니다"}

        # Calculate correlation
        correlation = plot_data[x_col].corr(plot_data[y_col])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))

        # Prepare scatter plot parameters
        scatter_kwargs = {
            'alpha': 0.7,
            'edgecolors': 'black',
            'linewidth': 0.5
        }

        # Handle color encoding
        if color_col and color_col in plot_data.columns:
            if pd.api.types.is_numeric_dtype(plot_data[color_col]):
                # Numeric color mapping
                scatter = ax.scatter(plot_data[x_col], plot_data[y_col],
                                   c=plot_data[color_col], cmap='viridis', **scatter_kwargs)
                plt.colorbar(scatter, label=color_col)
            else:
                # Categorical color mapping
                unique_categories = plot_data[color_col].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))

                for i, category in enumerate(unique_categories):
                    mask = plot_data[color_col] == category
                    ax.scatter(plot_data[mask][x_col], plot_data[mask][y_col],
                             c=[colors[i]], label=str(category), **scatter_kwargs)
                ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple scatter plot
            ax.scatter(plot_data[x_col], plot_data[y_col], **scatter_kwargs)

        # Handle size encoding
        if size_col and size_col in plot_data.columns and pd.api.types.is_numeric_dtype(plot_data[size_col]):
            # Normalize sizes
            sizes = plot_data[size_col]
            normalized_sizes = 20 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 200
            scatter_kwargs['s'] = normalized_sizes

        # Styling
        ax.set_title(f'{y_col} vs {x_col}', fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=14)
        ax.set_ylabel(y_col, fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add correlation annotation
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                verticalalignment='top')

        # Add trend line if correlation is significant
        if abs(correlation) > 0.3:
            z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(plot_data[x_col], p(plot_data[x_col]), "r--", alpha=0.7,
                   label=f'Trend line (r={correlation:.3f})')
            if not (color_col and color_col in plot_data.columns and not pd.api.types.is_numeric_dtype(plot_data[color_col])):
                ax.legend()

        # Adjust layout
        plt.tight_layout()

        # Save plot
        plot_filename = None
        if save_plot:
            plot_filename = f"scatter_{x_col}_vs_{y_col}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        # Show plot interactively
        if show_plot and INTERACTIVE_AVAILABLE:
            try:
                plt.show(block=False)  # Non-blocking show
                print(f"📊 Interactive plot displayed! Close the window to continue.")
            except Exception as e:
                print(f"⚠️ Could not display interactive plot: {e}")

        # Don't close immediately to allow viewing
        if not show_plot:
            plt.close()

        # Prepare result
        result = {
            "success": True,
            "plot_saved": str(plot_path) if save_plot else None,
            "correlation": float(correlation),
            "data_points": len(plot_data),
            "x_column": x_col,
            "y_column": y_col,
            "color_column": color_col,
            "size_column": size_col,
            "interactive_displayed": show_plot and INTERACTIVE_AVAILABLE,
            "statistics": {
                "x_mean": float(plot_data[x_col].mean()),
                "y_mean": float(plot_data[y_col].mean()),
                "x_std": float(plot_data[x_col].std()),
                "y_std": float(plot_data[y_col].std())
            }
        }

        return result

    def create_multiple_scatter_plots(self, df: pd.DataFrame, show_plot: bool = True) -> Dict[str, Any]:
        """Create scatter plots for all numeric column pairs"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return {
                "error": "최소 2개 이상의 숫자형 열이 필요합니다",
                "available_columns": numeric_cols
            }

        results = []

        # Create pairwise scatter plots
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                x_col = numeric_cols[i]
                y_col = numeric_cols[j]

                result = self.create_scatter_plot(df, x_col, y_col,
                                                show_plot=False, save_plot=True)
                if result.get("success"):
                    results.append(result)

        # Create pairplot using seaborn for overview
        if len(numeric_cols) <= 6:  # Only for reasonable number of columns
            plt.figure(figsize=(12, 10))
            pair_plot = sns.pairplot(df[numeric_cols], diag_kind='hist', plot_kws={'alpha': 0.7})

            pairplot_path = self.output_dir / "pairplot_overview.png"
            pair_plot.savefig(pairplot_path, dpi=300, bbox_inches='tight')

            if show_plot and INTERACTIVE_AVAILABLE:
                try:
                    plt.show(block=False)
                    print(f"📊 Pairplot overview displayed!")
                except Exception as e:
                    print(f"⚠️ Could not display pairplot: {e}")

            plt.close()

        return {
            "success": True,
            "individual_plots": results,
            "pairplot_saved": str(pairplot_path) if len(numeric_cols) <= 6 else None,
            "total_plots_created": len(results)
        }

def main():
    """Main function for CLI usage"""
    try:
        # Read JSON data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Initialize visualizer
        visualizer = ScatterPlotVisualizer()

        # Check command line arguments for specific plot type
        if len(sys.argv) > 1 and sys.argv[1] == "multiple":
            # Create multiple scatter plots
            result = visualizer.create_multiple_scatter_plots(df, show_plot=True)
        else:
            # Create single scatter plot
            result = visualizer.create_scatter_plot(df, show_plot=True, save_plot=True)

        # Final result
        final_result = {
            "success": True,
            "analysis_type": "enhanced_scatter_plot_visualization",
            "interactive_backend": INTERACTIVE_AVAILABLE,
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()
            },
            "visualization_result": result
        }

        print(json.dumps(final_result, ensure_ascii=False, indent=2, default=str))

        # Keep plot open for a moment if interactive
        if INTERACTIVE_AVAILABLE and result.get("interactive_displayed"):
            input("Press Enter to close plots and continue...")
            plt.close('all')

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "enhanced_scatter_plot_visualization"
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()