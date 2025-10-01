# ML MCP System - Usage Examples & Workflows

## 📖 Overview

This document provides comprehensive examples and real-world workflows for the ML MCP System. Each example includes sample data, step-by-step instructions, and expected outputs.

## 🏁 Quick Start Guide

### Prerequisites
```bash
# 1. Install dependencies
npm install
pip install -r python/requirements.txt

# 2. Start the MCP servers
npm run mcp:analysis
npm run mcp:ml
npm run mcp:visualization
```

### Your First Analysis (5 minutes)
```bash
# 1. Use sample data
cp data/sample_data.csv my_data.csv

# 2. Basic statistics
echo '{"data_file": "my_data.csv"}' | python python/analyzers/basic/descriptive_stats.py

# 3. Create visualizations
echo '{"data_file": "my_data.csv", "columns": ["age", "income"]}' | python python/visualization/2d/scatter.py
```

## 📊 Real-World Workflows

### 1. Customer Data Analysis Workflow

**Scenario**: Analyzing customer demographics and purchasing behavior

**Sample Data Structure**:
```csv
customer_id,age,income,education,spending_score,region
1001,25,35000,Bachelor,85,North
1002,45,75000,Master,45,South
1003,35,50000,Bachelor,75,East
...
```

#### Step 1: Initial Data Exploration
```bash
# Basic statistics
echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_analysis/step1_basics"
}' | python python/analyzers/basic/descriptive_stats.py
```

**Expected Output**:
```json
{
  "success": true,
  "statistics": {
    "age": {"mean": 38.5, "std": 12.3, "min": 18, "max": 70},
    "income": {"mean": 52500, "std": 18200, "min": 20000, "max": 150000},
    "spending_score": {"mean": 60.2, "std": 25.8, "min": 1, "max": 99}
  }
}
```

#### Step 2: Missing Data Analysis
```bash
echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_analysis/step2_missing"
}' | python python/analyzers/basic/missing_data.py
```

#### Step 3: Correlation Analysis
```bash
echo '{
  "data_file": "customer_data.csv",
  "columns": ["age", "income", "spending_score"],
  "output_dir": "customer_analysis/step3_correlation"
}' | python python/analyzers/basic/correlation.py
```

#### Step 4: Customer Segmentation (Clustering)
```bash
echo '{
  "data_file": "customer_data.csv",
  "feature_columns": ["age", "income", "spending_score"],
  "algorithms": ["kmeans", "hierarchical"],
  "n_clusters": 4,
  "output_dir": "customer_analysis/step4_segmentation"
}' | python python/ml/unsupervised/clustering.py
```

#### Step 5: Visualization Dashboard
```bash
# Scatter plot matrix
echo '{
  "data_file": "customer_data.csv",
  "x_column": "age",
  "y_column": "income",
  "color_column": "region",
  "size_column": "spending_score",
  "plot_types": ["2d", "matrix", "correlations"],
  "output_dir": "customer_analysis/step5_visuals"
}' | python python/visualizations/scatter_plots.py

# Interactive dashboard
echo '{
  "data_file": "customer_data.csv",
  "numeric_columns": ["age", "income", "spending_score"],
  "categorical_columns": ["education", "region"],
  "plot_types": ["plotly_dashboard", "plotly_3d"],
  "output_dir": "customer_analysis/step5_visuals"
}' | python python/visualizations/interactive_plots.py
```

### 2. Sales Forecasting Workflow

**Scenario**: Predicting monthly sales based on historical data

**Sample Data Structure**:
```csv
date,sales,marketing_spend,season,promotions,competitor_price
2023-01-01,125000,15000,Winter,2,99.99
2023-02-01,135000,18000,Winter,1,95.99
2023-03-01,142000,16000,Spring,3,98.99
...
```

#### Complete Workflow Script:
```bash
#!/bin/bash
# Sales Forecasting Complete Workflow

ANALYSIS_DIR="sales_forecast_$(date +%Y%m%d_%H%M%S)"
DATA_FILE="sales_data.csv"

echo "🚀 Starting Sales Forecasting Analysis..."

# Step 1: Data Quality Check
echo "📊 Step 1: Data Quality Analysis..."
echo '{
  "data_file": "'$DATA_FILE'",
  "output_dir": "'$ANALYSIS_DIR'/01_data_quality"
}' | python python/analyzers/basic/missing_data.py

# Step 2: Time Series Analysis
echo "📈 Step 2: Time Series Visualization..."
echo '{
  "data_file": "'$DATA_FILE'",
  "date_column": "date",
  "value_columns": ["sales", "marketing_spend"],
  "plot_types": ["line", "seasonal_decompose", "rolling_stats"],
  "output_dir": "'$ANALYSIS_DIR'/02_timeseries"
}' | python python/visualizations/time_series_plots.py

# Step 3: Feature Engineering
echo "🔧 Step 3: Feature Engineering..."
echo '{
  "data_file": "'$DATA_FILE'",
  "target_column": "sales",
  "transformations": ["log", "polynomial"],
  "output_dir": "'$ANALYSIS_DIR'/03_features"
}' | python python/ml/preprocessing/feature_engineering.py

# Step 4: Forecasting Models
echo "🔮 Step 4: Building Forecast Models..."
echo '{
  "data_file": "'$ANALYSIS_DIR'/03_features/engineered_data.csv",
  "date_column": "date",
  "value_column": "sales",
  "forecast_periods": 12,
  "models": ["arima", "exponential_smoothing", "linear_trend"],
  "output_dir": "'$ANALYSIS_DIR'/04_forecasting"
}' | python python/ml/time_series/forecasting.py

# Step 5: Model Evaluation
echo "📊 Step 5: Model Evaluation..."
echo '{
  "model_files": ["'$ANALYSIS_DIR'/04_forecasting/arima_model.joblib",
                  "'$ANALYSIS_DIR'/04_forecasting/exp_smoothing_model.joblib"],
  "test_data_file": "'$DATA_FILE'",
  "target_column": "sales",
  "output_dir": "'$ANALYSIS_DIR'/05_evaluation"
}' | python python/ml/evaluation/model_evaluation.py

echo "✅ Analysis complete! Results in: $ANALYSIS_DIR"
```

### 3. Medical Diagnosis Classification

**Scenario**: Building a classification model for medical diagnosis

**Sample Data Structure**:
```csv
patient_id,age,gender,symptom1,symptom2,test_result1,test_result2,diagnosis
P001,45,M,1,0,15.2,Normal,Disease_A
P002,32,F,0,1,12.8,Abnormal,Healthy
P003,67,M,1,1,18.5,Abnormal,Disease_B
...
```

#### Advanced Classification Workflow:
```bash
# 1. Comprehensive Data Analysis
echo '{
  "data_file": "medical_data.csv",
  "output_dir": "medical_analysis/exploratory"
}' | python python/analyzers/basic/descriptive_stats.py

# 2. Distribution Analysis by Diagnosis
echo '{
  "data_file": "medical_data.csv",
  "categorical_columns": ["diagnosis"],
  "numeric_columns": ["age", "test_result1"],
  "plot_types": ["box", "violin", "strip"],
  "output_dir": "medical_analysis/distributions"
}' | python python/visualizations/categorical_plots.py

# 3. Feature Selection and Engineering
echo '{
  "data_file": "medical_data.csv",
  "target_column": "diagnosis",
  "techniques": ["selection", "scaling", "interaction"],
  "selection_method": "recursive",
  "output_dir": "medical_analysis/features"
}' | python python/ml/preprocessing/advanced_feature_engineering.py

# 4. Multi-Algorithm Classification
echo '{
  "data_file": "medical_analysis/features/engineered_data.csv",
  "target_column": "diagnosis",
  "algorithms": ["logistic", "random_forest", "svm", "gradient_boosting"],
  "cross_validation": 10,
  "test_size": 0.2,
  "output_dir": "medical_analysis/models"
}' | python python/ml/supervised/classification/classification_trainer.py

# 5. Statistical Validation
echo '{
  "data_file": "medical_analysis/features/engineered_data.csv",
  "numeric_columns": ["age", "test_result1"],
  "target_column": "diagnosis",
  "plot_types": ["distribution", "qq", "confidence"],
  "output_dir": "medical_analysis/statistics"
}' | python python/visualizations/statistical_plots.py
```

### 4. Financial Risk Assessment

**Scenario**: Assessing loan default risk using customer financial data

**Sample Data Structure**:
```csv
loan_id,age,income,credit_score,loan_amount,employment_years,debt_ratio,default
L001,28,45000,650,25000,3,0.35,0
L002,45,85000,750,50000,12,0.25,0
L003,35,35000,580,30000,2,0.65,1
...
```

#### Risk Assessment Pipeline:
```python
# risk_assessment_pipeline.py
import subprocess
import json
from datetime import datetime

def run_analysis(params, script_path):
    """Helper function to run analysis tools"""
    process = subprocess.run(
        ['python', script_path],
        input=json.dumps(params),
        text=True,
        capture_output=True
    )
    return json.loads(process.stdout)

def financial_risk_pipeline(data_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"risk_assessment_{timestamp}"

    # Step 1: Outlier Detection (fraud indicators)
    print("🔍 Detecting outliers and anomalies...")
    outlier_params = {
        "data_file": data_file,
        "methods": ["iqr", "isolation_forest", "lof"],
        "contamination": 0.05,
        "output_dir": f"{output_base}/01_outliers"
    }
    outliers = run_analysis(outlier_params, "python/analyzers/advanced/outlier_detection.py")

    # Step 2: Correlation Analysis
    print("📊 Analyzing feature correlations...")
    corr_params = {
        "data_file": data_file,
        "method": "spearman",  # Better for financial data
        "min_correlation": 0.3,
        "output_dir": f"{output_base}/02_correlations"
    }
    correlations = run_analysis(corr_params, "python/analyzers/basic/correlation.py")

    # Step 3: Risk Factor Engineering
    print("🔧 Engineering risk factors...")
    feature_params = {
        "data_file": data_file,
        "target_column": "default",
        "transformations": ["log", "interaction"],
        "output_dir": f"{output_base}/03_features"
    }
    features = run_analysis(feature_params, "python/ml/preprocessing/feature_engineering.py")

    # Step 4: Model Training with Imbalanced Data Handling
    print("🤖 Training risk models...")
    model_params = {
        "data_file": f"{output_base}/03_features/engineered_data.csv",
        "target_column": "default",
        "algorithms": ["logistic", "random_forest", "gradient_boosting"],
        "cross_validation": 5,
        "class_weight": "balanced",  # Handle imbalanced classes
        "output_dir": f"{output_base}/04_models"
    }
    models = run_analysis(model_params, "python/ml/supervised/classification/classification_trainer.py")

    # Step 5: Risk Visualization
    print("📈 Creating risk visualizations...")
    viz_params = {
        "data_file": data_file,
        "numeric_columns": ["age", "income", "credit_score", "debt_ratio"],
        "categorical_columns": ["default"],
        "plot_types": ["plotly_scatter", "plotly_heatmap", "plotly_dashboard"],
        "output_dir": f"{output_base}/05_visualizations"
    }
    visualizations = run_analysis(viz_params, "python/visualizations/interactive_plots.py")

    print(f"✅ Risk assessment complete! Results in: {output_base}")
    return {
        "analysis_id": output_base,
        "outliers_detected": len(outliers.get("consensus_outliers", [])),
        "best_model": models.get("best_model", {}),
        "key_risk_factors": correlations.get("significant_correlations", [])[:5]
    }

# Usage
if __name__ == "__main__":
    result = financial_risk_pipeline("loan_data.csv")
    print(json.dumps(result, indent=2))
```

## 🎯 Specialized Use Cases

### A. Real Estate Price Prediction

```bash
# Complete real estate analysis
echo '{
  "data_file": "real_estate.csv",
  "target_column": "price",
  "feature_columns": ["sqft", "bedrooms", "bathrooms", "age", "location_score"],
  "algorithms": ["linear", "ridge", "random_forest"],
  "output_dir": "real_estate_analysis"
}' | python python/ml/supervised/regression/regression_trainer.py
```

### B. Marketing Campaign Optimization

```bash
# A/B test analysis
echo '{
  "data_file": "campaign_data.csv",
  "categorical_columns": ["campaign_type", "customer_segment"],
  "numeric_columns": ["conversion_rate", "cost_per_click", "revenue"],
  "plot_types": ["bar", "box", "heatmap"],
  "output_dir": "campaign_analysis"
}' | python python/visualizations/categorical_plots.py
```

### C. Quality Control Analysis

```bash
# Manufacturing quality analysis
echo '{
  "data_file": "production_data.csv",
  "columns": ["temperature", "pressure", "speed", "quality_score"],
  "methods": ["iqr", "zscore"],
  "output_dir": "quality_control"
}' | python python/analyzers/advanced/outlier_detection.py
```

## 📋 Best Practices & Tips

### 1. Data Preparation Checklist
```bash
# Before analysis, always check:
echo '{
  "data_file": "your_data.csv"
}' | python python/analyzers/basic/missing_data.py

# Clean and validate
echo '{
  "data_file": "your_data.csv",
  "strategy": "analyze"
}' | python python/analyzers/basic/missing_data.py
```

### 2. Progressive Analysis Strategy

**Phase 1: Exploration** (5-10 minutes)
- Basic statistics
- Missing data analysis
- Distribution plots

**Phase 2: Investigation** (15-30 minutes)
- Correlation analysis
- Outlier detection
- Categorical analysis

**Phase 3: Modeling** (30-60 minutes)
- Feature engineering
- Model training
- Model evaluation

**Phase 4: Validation** (15-30 minutes)
- Statistical tests
- Residual analysis
- Cross-validation

### 3. Output Organization

```
project_analysis_YYYYMMDD_HHMMSS/
├── 01_exploration/
│   ├── basic_stats.json
│   ├── distributions.png
│   └── missing_data.json
├── 02_investigation/
│   ├── correlations.png
│   ├── outliers.json
│   └── categorical_analysis.png
├── 03_modeling/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── 04_validation/
│   ├── statistical_tests.json
│   └── residual_plots.png
└── final_report.html
```

### 4. Common Workflows by Industry

#### Healthcare Analytics
```bash
# Patient outcome prediction
1. missing_data.py → Clean patient records
2. correlation.py → Find symptom relationships
3. classification.py → Predict outcomes
4. statistical_plots.py → Validate results
```

#### Financial Services
```bash
# Credit risk assessment
1. outlier_detection.py → Fraud detection
2. feature_engineering.py → Risk factors
3. classification.py → Default prediction
4. interactive_plots.py → Risk dashboard
```

#### Retail Analytics
```bash
# Customer segmentation
1. descriptive_stats.py → Customer demographics
2. clustering.py → Segment customers
3. scatter_plots.py → Segment visualization
4. forecasting.py → Demand prediction
```

#### Manufacturing
```bash
# Quality control
1. distribution.py → Process stability
2. outlier_detection.py → Defect detection
3. time_series_plots.py → Trend analysis
4. regression.py → Quality prediction
```

## 🔧 Integration Examples

### With Jupyter Notebooks
```python
import subprocess
import json
import pandas as pd

def run_mcp_tool(tool_script, params):
    """Run MCP tool from Jupyter"""
    result = subprocess.run(
        ['python', tool_script],
        input=json.dumps(params),
        text=True,
        capture_output=True
    )
    return json.loads(result.stdout)

# Example usage
params = {
    "data_file": "data.csv",
    "columns": ["feature1", "feature2"]
}
result = run_mcp_tool("python/analyzers/basic/correlation.py", params)
print(result)
```

### With R Integration
```r
# R wrapper for MCP tools
library(jsonlite)

run_mcp_analysis <- function(script_path, params) {
  params_json <- toJSON(params, auto_unbox = TRUE)
  result <- system2("python",
                   args = script_path,
                   input = params_json,
                   stdout = TRUE)
  fromJSON(result)
}

# Usage
params <- list(data_file = "data.csv", columns = c("x", "y"))
result <- run_mcp_analysis("python/analyzers/basic/correlation.py", params)
```

### With Web Applications
```javascript
// Node.js web service integration
const { spawn } = require('child_process');

async function runMCPAnalysis(scriptPath, params) {
  return new Promise((resolve, reject) => {
    const python = spawn('python', [scriptPath]);

    python.stdin.write(JSON.stringify(params));
    python.stdin.end();

    let result = '';
    python.stdout.on('data', (data) => {
      result += data;
    });

    python.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error('Analysis failed'));
      }
    });
  });
}

// Usage in Express.js
app.post('/api/analyze', async (req, res) => {
  try {
    const result = await runMCPAnalysis(
      'python/analyzers/basic/descriptive_stats.py',
      req.body
    );
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

**1. "Module not found" errors**
```bash
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
pip install -r python/requirements.txt
```

**2. "File not found" errors**
```bash
# Solution: Use absolute paths
echo '{
  "data_file": "/full/path/to/data.csv"
}' | python python/analyzers/basic/descriptive_stats.py
```

**3. Memory errors with large datasets**
```bash
# Solution: Use sampling
echo '{
  "data_file": "large_data.csv",
  "sample_size": 10000
}' | python python/analyzers/basic/descriptive_stats.py
```

**4. Permission errors**
```bash
# Solution: Check directory permissions
chmod 755 output_directory
mkdir -p output_directory
```

## 📈 Performance Optimization

### For Large Datasets
1. **Use chunking**: Process data in smaller batches
2. **Sample first**: Use representative samples for exploration
3. **Optimize memory**: Close files and clear variables
4. **Parallel processing**: Use multiple cores when available

### For Production Use
1. **Cache results**: Store intermediate results
2. **Validate inputs**: Check data quality before processing
3. **Monitor resources**: Track memory and CPU usage
4. **Log operations**: Maintain detailed logs for debugging

---

*For more examples and advanced usage patterns, see the [Developer Guide](DEVELOPER_GUIDE.md) and [API Reference](API_REFERENCE.md).*