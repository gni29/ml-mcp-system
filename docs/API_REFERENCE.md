# ML MCP System - API Reference

## 📖 Overview

The ML MCP System provides 21 powerful tools across three modules for comprehensive data analysis, machine learning, and visualization. This document provides complete API specifications for all available tools.

## 🏗️ System Architecture

```
ML MCP System
├── ml-mcp-analysis/     # Statistical Analysis (5 tools)
├── ml-mcp-ml/          # Machine Learning (8 tools)
├── ml-mcp-visualization/ # Data Visualization (8 tools)
└── ml-mcp-shared/      # Common utilities
```

## 📊 Module: Analysis Tools

### 1. Basic Statistics (`basic_stats`)

**Description**: Comprehensive descriptive statistics analysis

**Parameters**:
```json
{
  "data_file": "string (required) - Path to data file (CSV/Excel)",
  "columns": "array[string] (optional) - Specific columns to analyze",
  "output_dir": "string (optional, default: 'results') - Output directory"
}
```

**Input Formats**: CSV, Excel (.xlsx, .xls)

**Output**:
```json
{
  "success": true,
  "statistics": {
    "column_name": {
      "count": "number - Non-null count",
      "mean": "number - Arithmetic mean",
      "std": "number - Standard deviation",
      "min": "number - Minimum value",
      "25%": "number - 25th percentile",
      "50%": "number - Median",
      "75%": "number - 75th percentile",
      "max": "number - Maximum value",
      "skewness": "number - Skewness measure",
      "kurtosis": "number - Kurtosis measure"
    }
  },
  "data_info": {
    "shape": "[rows, columns]",
    "memory_usage": "string - Memory usage info",
    "dtypes": "object - Data types per column"
  }
}
```

**Error Codes**:
- `FILE_NOT_FOUND`: Data file does not exist
- `INVALID_FORMAT`: Unsupported file format
- `NO_NUMERIC_COLUMNS`: No numeric columns found for analysis

### 2. Correlation Analysis (`correlation`)

**Description**: Pearson, Spearman, and Kendall correlation analysis

**Parameters**:
```json
{
  "data_file": "string (required)",
  "columns": "array[string] (optional) - Columns to correlate",
  "method": "string (optional, default: 'pearson') - correlation|spearman|kendall",
  "min_correlation": "number (optional, default: 0.0) - Minimum correlation threshold",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "correlation_matrix": "object - Full correlation matrix",
  "significant_correlations": [
    {
      "variable1": "string",
      "variable2": "string",
      "correlation": "number",
      "p_value": "number",
      "interpretation": "string - weak|moderate|strong"
    }
  ],
  "method_used": "string",
  "generated_files": ["correlation_heatmap.png"]
}
```

### 3. Distribution Analysis (`distribution`)

**Description**: Statistical distribution analysis and visualization

**Parameters**:
```json
{
  "data_file": "string (required)",
  "columns": "array[string] (optional)",
  "plot_types": "array[string] (optional) - histogram|kde|qq|box",
  "bins": "number (optional, default: 30) - Histogram bins",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "distributions": {
    "column_name": {
      "normality_test": {
        "shapiro_wilk": {"statistic": "number", "p_value": "number"},
        "jarque_bera": {"statistic": "number", "p_value": "number"}
      },
      "distribution_params": {
        "mean": "number",
        "std": "number",
        "skewness": "number",
        "kurtosis": "number"
      }
    }
  },
  "generated_files": ["distribution_plots.png"]
}
```

### 4. Missing Data Analysis (`missing_data`)

**Description**: Comprehensive missing data pattern analysis

**Parameters**:
```json
{
  "data_file": "string (required)",
  "strategy": "string (optional, default: 'analyze') - analyze|impute",
  "impute_method": "string (optional) - mean|median|mode|forward_fill|backward_fill",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "missing_summary": {
    "total_missing": "number",
    "missing_percentage": "number",
    "columns_with_missing": "array[string]"
  },
  "missing_patterns": [
    {
      "pattern": "string - Binary pattern of missingness",
      "count": "number - Rows with this pattern",
      "percentage": "number"
    }
  ],
  "recommendations": {
    "action": "string - Strategy recommendation",
    "reasoning": "string - Why this strategy is recommended"
  }
}
```

### 5. Outlier Detection (`outlier_detection`)

**Description**: Multi-method outlier detection and analysis

**Parameters**:
```json
{
  "data_file": "string (required)",
  "columns": "array[string] (optional)",
  "methods": "array[string] (optional) - iqr|zscore|isolation_forest|lof",
  "contamination": "number (optional, default: 0.1) - Expected outlier proportion",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "outliers_detected": {
    "method_name": {
      "outlier_indices": "array[number] - Row indices of outliers",
      "outlier_count": "number",
      "outlier_percentage": "number",
      "threshold_values": "object - Method-specific thresholds"
    }
  },
  "consensus_outliers": "array[number] - Outliers detected by multiple methods",
  "generated_files": ["outlier_analysis.png"]
}
```

## 🤖 Module: Machine Learning Tools

### 1. Feature Engineering (`feature_engineering`)

**Description**: Automated feature engineering and transformation

**Parameters**:
```json
{
  "data_file": "string (required)",
  "target_column": "string (optional) - Target variable for supervised learning",
  "feature_types": "object (optional) - Specify column types",
  "transformations": "array[string] (optional) - log|sqrt|polynomial|interaction",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "engineered_features": {
    "new_features": "array[string] - Names of created features",
    "transformation_log": "array[object] - Applied transformations",
    "feature_importance": "object - Feature importance scores"
  },
  "data_shape": {
    "original": "[rows, cols]",
    "engineered": "[rows, cols]"
  },
  "saved_artifacts": {
    "engineered_data": "string - File path",
    "transformer_pipeline": "string - Sklearn pipeline file"
  }
}
```

### 2. Classification Training (`classification`)

**Description**: Multi-algorithm classification model training and evaluation

**Parameters**:
```json
{
  "data_file": "string (required)",
  "target_column": "string (required) - Classification target",
  "feature_columns": "array[string] (optional) - Features to use",
  "algorithms": "array[string] (optional) - logistic|random_forest|svm|gradient_boosting",
  "test_size": "number (optional, default: 0.2) - Train/test split ratio",
  "cross_validation": "number (optional, default: 5) - CV folds",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "models_trained": {
    "algorithm_name": {
      "accuracy": "number",
      "precision": "number",
      "recall": "number",
      "f1_score": "number",
      "roc_auc": "number",
      "confusion_matrix": "array[array[number]]",
      "feature_importance": "object",
      "model_file": "string - Saved model path"
    }
  },
  "best_model": {
    "algorithm": "string",
    "score": "number",
    "hyperparameters": "object"
  }
}
```

### 3. Regression Training (`regression`)

**Description**: Multi-algorithm regression model training and evaluation

**Parameters**:
```json
{
  "data_file": "string (required)",
  "target_column": "string (required) - Regression target",
  "feature_columns": "array[string] (optional)",
  "algorithms": "array[string] (optional) - linear|ridge|lasso|random_forest|gradient_boosting",
  "test_size": "number (optional, default: 0.2)",
  "cross_validation": "number (optional, default: 5)",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "models_trained": {
    "algorithm_name": {
      "r2_score": "number - R-squared coefficient",
      "mean_squared_error": "number",
      "mean_absolute_error": "number",
      "root_mean_squared_error": "number",
      "feature_importance": "object",
      "model_file": "string"
    }
  },
  "residual_analysis": {
    "residual_std": "number",
    "residual_mean": "number",
    "normality_test": "object"
  }
}
```

### 4. Clustering Analysis (`clustering`)

**Description**: Unsupervised clustering with multiple algorithms

**Parameters**:
```json
{
  "data_file": "string (required)",
  "feature_columns": "array[string] (optional)",
  "algorithms": "array[string] (optional) - kmeans|hierarchical|dbscan|gaussian_mixture",
  "n_clusters": "number (optional, default: 3) - For applicable algorithms",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "clustering_results": {
    "algorithm_name": {
      "cluster_labels": "array[number] - Cluster assignments",
      "n_clusters_found": "number",
      "silhouette_score": "number",
      "calinski_harabasz_score": "number",
      "davies_bouldin_score": "number"
    }
  },
  "cluster_analysis": {
    "cluster_centers": "array[array[number]]",
    "cluster_sizes": "array[number]",
    "optimal_clusters": "number - Recommended cluster count"
  }
}
```

### 5. Time Series Forecasting (`forecasting`)

**Description**: Time series analysis and forecasting

**Parameters**:
```json
{
  "data_file": "string (required)",
  "date_column": "string (required) - Date/time column",
  "value_column": "string (required) - Values to forecast",
  "forecast_periods": "number (optional, default: 30) - Periods to forecast",
  "models": "array[string] (optional) - arima|exponential_smoothing|linear_trend",
  "seasonal": "boolean (optional, default: true) - Include seasonality",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "forecasting_results": {
    "model_name": {
      "forecast_values": "array[number]",
      "confidence_intervals": {
        "lower": "array[number]",
        "upper": "array[number]"
      },
      "model_metrics": {
        "aic": "number",
        "bic": "number",
        "mape": "number",
        "rmse": "number"
      }
    }
  },
  "time_series_analysis": {
    "trend": "string - increasing|decreasing|stable",
    "seasonality_detected": "boolean",
    "seasonal_period": "number"
  }
}
```

### 6. PCA Analysis (`pca`)

**Description**: Principal Component Analysis for dimensionality reduction

**Parameters**:
```json
{
  "data_file": "string (required)",
  "feature_columns": "array[string] (optional)",
  "n_components": "number (optional) - Number of components",
  "variance_threshold": "number (optional, default: 0.95) - Variance to retain",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "pca_results": {
    "n_components": "number - Components selected",
    "explained_variance_ratio": "array[number] - Variance per component",
    "cumulative_variance": "array[number]",
    "component_loadings": "array[array[number]]",
    "transformed_data_file": "string"
  },
  "dimensionality_reduction": {
    "original_dimensions": "number",
    "reduced_dimensions": "number",
    "variance_retained": "number"
  }
}
```

### 7. Advanced Feature Engineering (`advanced_feature_engineering`)

**Description**: Sophisticated feature engineering techniques

**Parameters**:
```json
{
  "data_file": "string (required)",
  "target_column": "string (optional)",
  "techniques": "array[string] (optional) - polynomial|interaction|selection|scaling",
  "polynomial_degree": "number (optional, default: 2)",
  "selection_method": "string (optional) - univariate|recursive|lasso",
  "output_dir": "string (optional)"
}
```

### 8. Model Evaluation (`model_evaluation`)

**Description**: Comprehensive model evaluation and comparison

**Parameters**:
```json
{
  "model_files": "array[string] (required) - Paths to saved models",
  "test_data_file": "string (required)",
  "target_column": "string (required)",
  "evaluation_metrics": "array[string] (optional) - Custom metrics",
  "output_dir": "string (optional)"
}
```

## 📊 Module: Visualization Tools

### 1. Scatter Plots (`scatter_plots`)

**Description**: Multi-dimensional scatter plot analysis

**Parameters**:
```json
{
  "data_file": "string (required)",
  "x_column": "string (required) - X-axis variable",
  "y_column": "string (required) - Y-axis variable",
  "color_column": "string (optional) - Categorical coloring",
  "size_column": "string (optional) - Point size variable",
  "plot_types": "array[string] (optional) - 2d|3d|matrix|outliers|correlations",
  "output_dir": "string (optional)"
}
```

**Output**:
```json
{
  "success": true,
  "generated_files": "array[string] - Generated plot files",
  "scatter_analysis": {
    "correlation_coefficient": "number",
    "trend_line": {
      "slope": "number",
      "intercept": "number",
      "r_squared": "number"
    },
    "outliers_detected": "array[number] - Outlier indices"
  }
}
```

### 2. Time Series Plots (`time_series_plots`)

**Description**: Comprehensive time series visualization

**Parameters**:
```json
{
  "data_file": "string (required)",
  "date_column": "string (required)",
  "value_columns": "array[string] (required)",
  "plot_types": "array[string] (optional) - line|area|seasonal_decompose|rolling_stats|autocorrelation",
  "rolling_window": "number (optional, default: 30)",
  "output_dir": "string (optional)"
}
```

### 3. Categorical Plots (`categorical_plots`)

**Description**: Categorical data visualization suite

**Parameters**:
```json
{
  "data_file": "string (required)",
  "categorical_columns": "array[string] (required)",
  "numeric_columns": "array[string] (optional)",
  "plot_types": "array[string] (optional) - bar|pie|box|violin|heatmap|strip",
  "output_dir": "string (optional)"
}
```

### 4. Statistical Plots (`statistical_plots`)

**Description**: Statistical analysis visualizations

**Parameters**:
```json
{
  "data_file": "string (required)",
  "numeric_columns": "array[string] (required)",
  "target_column": "string (optional) - For regression analysis",
  "plot_types": "array[string] (optional) - distribution|qq|residual|probability|confidence",
  "output_dir": "string (optional)"
}
```

### 5. Interactive Plots (`interactive_plots`)

**Description**: Interactive web-based visualizations

**Parameters**:
```json
{
  "data_file": "string (required)",
  "numeric_columns": "array[string] (optional)",
  "categorical_columns": "array[string] (optional)",
  "plot_types": "array[string] (optional) - plotly_scatter|plotly_timeseries|plotly_3d|plotly_heatmap|bokeh_scatter",
  "output_dir": "string (optional)"
}
```

**Output**: Interactive HTML files with embedded JavaScript for web viewing

### 6. Distribution Plots (`distribution_plots`)

**Description**: Statistical distribution visualization

**Parameters**:
```json
{
  "data_file": "string (required)",
  "columns": "array[string] (optional)",
  "plot_types": "array[string] (optional) - histogram|kde|qq|box|violin",
  "output_dir": "string (optional)"
}
```

### 7. Heatmaps (`heatmaps`)

**Description**: Correlation and data heatmaps

**Parameters**:
```json
{
  "data_file": "string (required)",
  "columns": "array[string] (optional)",
  "correlation_method": "string (optional, default: 'pearson')",
  "annot": "boolean (optional, default: true)",
  "output_dir": "string (optional)"
}
```

### 8. Advanced Plots (`advanced_plots`)

**Description**: Complex multi-panel visualizations

**Parameters**:
```json
{
  "data_file": "string (required)",
  "analysis_type": "string (required) - regression|classification|clustering|comparison",
  "columns": "array[string] (optional)",
  "output_dir": "string (optional)"
}
```

## 🔧 Common Parameters

### File Format Support
- **CSV**: `.csv` files with various delimiters
- **Excel**: `.xlsx`, `.xls` files (all sheets supported)
- **Parquet**: `.parquet` files (high performance)
- **JSON**: `.json` files (structured data)

### Output Directory Structure
```
output_dir/
├── data/           # Processed datasets
├── models/         # Trained model files
├── plots/          # Generated visualizations
├── reports/        # Analysis reports
└── metadata.json   # Execution metadata
```

### Error Handling

All tools return standardized error responses:
```json
{
  "success": false,
  "error": "string - Human-readable error message",
  "error_type": "string - Error category",
  "error_code": "string - Specific error code",
  "suggestions": "array[string] - Potential solutions"
}
```

### Performance Guidelines

| Dataset Size | Memory Requirement | Processing Time | Recommendations |
|-------------|-------------------|-----------------|-----------------|
| < 1MB | 50MB | < 1 second | All tools available |
| 1-100MB | 500MB | 1-30 seconds | Use sampling for exploration |
| 100MB-1GB | 2GB | 30 seconds-5 minutes | Consider chunking |
| > 1GB | 8GB+ | 5+ minutes | Use distributed processing |

## 🚨 Rate Limits & Resource Management

- **Concurrent Requests**: Max 5 per client
- **Memory Limit**: 4GB per operation
- **File Size Limit**: 2GB per file
- **Processing Timeout**: 10 minutes per operation

## 📝 Best Practices

### Data Preparation
1. **Clean Data**: Remove or handle missing values appropriately
2. **Consistent Types**: Ensure consistent data types across columns
3. **Reasonable Size**: Keep initial datasets under 100MB for exploration
4. **Backup Data**: Always keep original data backups

### Tool Selection
1. **Start Simple**: Begin with basic statistics and distribution analysis
2. **Iterate**: Use results to guide next analysis steps
3. **Validate**: Cross-check results with multiple tools
4. **Document**: Save analysis parameters for reproducibility

### Output Management
1. **Organize**: Use descriptive output directory names
2. **Version**: Include timestamps in analysis runs
3. **Archive**: Archive completed analyses for future reference
4. **Share**: Use generated reports for collaboration

---

*For additional support, examples, and tutorials, see the [Usage Examples](USAGE_EXAMPLES.md) and [Developer Guide](DEVELOPER_GUIDE.md).*