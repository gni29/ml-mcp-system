# ML MCP System - Tutorials & Walkthroughs

## 📚 Learning Path Overview

This document provides step-by-step tutorials for different user levels and use cases. Follow the appropriate path based on your experience and goals.

### 🎯 Learning Paths

| Path | Duration | Prerequisites | Focus |
|------|----------|---------------|-------|
| **Beginner** | 2-3 hours | Basic data concepts | Data exploration & visualization |
| **Intermediate** | 4-6 hours | Python basics, statistics | Machine learning workflows |
| **Advanced** | 8-12 hours | ML experience, programming | Custom implementations & optimization |
| **Domain-Specific** | 3-5 hours | Domain knowledge | Industry-specific applications |

---

## 🌱 Tutorial 1: Beginner's First Analysis
*Duration: 30 minutes | Level: Beginner*

### What You'll Learn
- Basic data exploration
- Creating your first visualizations
- Understanding analysis outputs

### Prerequisites
- Basic computer skills
- Curiosity about data

### Step-by-Step Walkthrough

#### Setup (5 minutes)
1. **Download Sample Data**
   ```bash
   # Create a simple CSV file
   cat > my_first_data.csv << EOF
   name,age,salary,department,years_experience
   Alice,28,55000,Engineering,3
   Bob,35,65000,Sales,8
   Carol,42,75000,Engineering,12
   David,29,48000,Marketing,2
   Eve,31,58000,Sales,5
   Frank,38,82000,Engineering,15
   Grace,26,45000,Marketing,1
   Henry,44,95000,Engineering,18
   EOF
   ```

2. **Verify Your Setup**
   ```bash
   # Check if files exist
   ls -la my_first_data.csv

   # Test Python environment
   python --version
   ```

#### Your First Analysis (10 minutes)

**Step 1: Basic Statistics**
```bash
echo '{
  "data_file": "my_first_data.csv",
  "output_dir": "my_first_analysis"
}' | python python/analyzers/basic/descriptive_stats.py
```

**What Happened?**
- The tool read your CSV file
- Calculated statistics for numeric columns (age, salary, years_experience)
- Saved results to `my_first_analysis/` folder

**Understanding Your Results:**
```json
{
  "success": true,
  "statistics": {
    "age": {
      "mean": 34.125,     // Average age
      "std": 6.4,         // Age variation
      "min": 26,          // Youngest person
      "max": 44           // Oldest person
    },
    "salary": {
      "mean": 65250,      // Average salary
      "std": 16800,       // Salary variation
      "min": 45000,       // Lowest salary
      "max": 95000        // Highest salary
    }
  }
}
```

**Step 2: Create Your First Visualization**
```bash
echo '{
  "data_file": "my_first_data.csv",
  "x_column": "age",
  "y_column": "salary",
  "color_column": "department",
  "plot_types": ["2d"],
  "output_dir": "my_first_analysis"
}' | python python/visualizations/scatter_plots.py
```

**What You'll See:**
- A scatter plot showing age vs salary
- Different colors for each department
- Clear patterns (older employees tend to earn more)

#### Exploring Relationships (10 minutes)

**Step 3: Department Analysis**
```bash
echo '{
  "data_file": "my_first_data.csv",
  "categorical_columns": ["department"],
  "numeric_columns": ["salary", "age"],
  "plot_types": ["bar", "box"],
  "output_dir": "my_first_analysis"
}' | python python/visualizations/categorical_plots.py
```

**Step 4: Check for Patterns**
```bash
echo '{
  "data_file": "my_first_data.csv",
  "columns": ["age", "salary", "years_experience"],
  "output_dir": "my_first_analysis"
}' | python python/analyzers/basic/correlation.py
```

#### Wrap-up (5 minutes)

**What You've Learned:**
1. ✅ How to run basic analysis tools
2. ✅ How to interpret statistical summaries
3. ✅ How to create visualizations
4. ✅ How to explore relationships in data

**Next Steps:**
- Try Tutorial 2 for more advanced analysis
- Experiment with your own data files
- Explore different plot types

---

## 📊 Tutorial 2: Sales Data Deep Dive
*Duration: 60 minutes | Level: Intermediate*

### What You'll Learn
- Time series analysis
- Trend identification
- Forecasting basics
- Advanced visualizations

### Sample Scenario
You're analyzing 2 years of monthly sales data for an e-commerce company.

#### Prepare Sample Data (10 minutes)

```python
# generate_sales_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 24 months of sales data
dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
np.random.seed(42)

# Create realistic sales patterns
base_sales = 100000
trend = np.linspace(0, 50000, 24)  # Growing trend
seasonal = 20000 * np.sin(2 * np.pi * np.arange(24) / 12)  # Yearly seasonality
noise = np.random.normal(0, 10000, 24)

sales = base_sales + trend + seasonal + noise
marketing_spend = sales * 0.1 + np.random.normal(0, 2000, 24)

# Create additional features
data = pd.DataFrame({
    'date': dates,
    'sales': sales.astype(int),
    'marketing_spend': marketing_spend.astype(int),
    'season': [['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer',
               'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Winter'][i%12]
               for i in range(24)],
    'promotions': np.random.poisson(2, 24),
    'competitor_price': 99.99 + np.random.normal(0, 5, 24)
})

data.to_csv('sales_data.csv', index=False)
print("Sales data generated successfully!")
```

```bash
python generate_sales_data.py
```

#### Phase 1: Initial Exploration (15 minutes)

**Step 1: Data Overview**
```bash
echo '{
  "data_file": "sales_data.csv",
  "output_dir": "sales_analysis/01_overview"
}' | python python/analyzers/basic/descriptive_stats.py
```

**Step 2: Check Data Quality**
```bash
echo '{
  "data_file": "sales_data.csv",
  "output_dir": "sales_analysis/01_overview"
}' | python python/analyzers/basic/missing_data.py
```

**Step 3: Time Series Visualization**
```bash
echo '{
  "data_file": "sales_data.csv",
  "date_column": "date",
  "value_columns": ["sales", "marketing_spend"],
  "plot_types": ["line", "seasonal_decompose", "rolling_stats"],
  "rolling_window": 3,
  "output_dir": "sales_analysis/02_timeseries"
}' | python python/visualizations/time_series_plots.py
```

**Key Questions to Ask:**
- Is there a clear upward trend?
- Do you see seasonal patterns?
- Are there any unusual spikes or dips?

#### Phase 2: Relationship Analysis (20 minutes)

**Step 4: Marketing Impact Analysis**
```bash
echo '{
  "data_file": "sales_data.csv",
  "x_column": "marketing_spend",
  "y_column": "sales",
  "color_column": "season",
  "plot_types": ["2d", "correlations"],
  "output_dir": "sales_analysis/03_relationships"
}' | python python/visualizations/scatter_plots.py
```

**Step 5: Seasonal Analysis**
```bash
echo '{
  "data_file": "sales_data.csv",
  "categorical_columns": ["season"],
  "numeric_columns": ["sales", "marketing_spend", "promotions"],
  "plot_types": ["box", "violin", "bar"],
  "output_dir": "sales_analysis/03_relationships"
}' | python python/visualizations/categorical_plots.py
```

**Step 6: Correlation Matrix**
```bash
echo '{
  "data_file": "sales_data.csv",
  "columns": ["sales", "marketing_spend", "promotions", "competitor_price"],
  "method": "pearson",
  "output_dir": "sales_analysis/03_relationships"
}' | python python/analyzers/basic/correlation.py
```

#### Phase 3: Predictive Modeling (15 minutes)

**Step 7: Feature Engineering**
```bash
echo '{
  "data_file": "sales_data.csv",
  "target_column": "sales",
  "transformations": ["polynomial", "interaction"],
  "output_dir": "sales_analysis/04_modeling"
}' | python python/ml/preprocessing/feature_engineering.py
```

**Step 8: Sales Forecasting**
```bash
echo '{
  "data_file": "sales_data.csv",
  "date_column": "date",
  "value_column": "sales",
  "forecast_periods": 6,
  "models": ["arima", "exponential_smoothing", "linear_trend"],
  "seasonal": true,
  "output_dir": "sales_analysis/04_modeling"
}' | python python/ml/time_series/forecasting.py
```

**Step 9: Interactive Dashboard**
```bash
echo '{
  "data_file": "sales_data.csv",
  "numeric_columns": ["sales", "marketing_spend", "promotions"],
  "categorical_columns": ["season"],
  "plot_types": ["plotly_dashboard", "plotly_timeseries"],
  "output_dir": "sales_analysis/05_dashboard"
}' | python python/visualizations/interactive_plots.py
```

#### Analysis Summary Exercise

**Your Mission:** Write a brief analysis summary answering:

1. **Trend Analysis:**
   - What's the overall sales trend?
   - Which seasons perform best?

2. **Marketing Effectiveness:**
   - Is marketing spend correlated with sales?
   - What's the ROI pattern?

3. **Forecasting Insights:**
   - What do the next 6 months look like?
   - Which model performed best?

**Template:**
```markdown
# Sales Analysis Summary

## Key Findings
- Overall trend: [Increasing/Decreasing/Stable]
- Best performing season: [Season]
- Marketing correlation: [Strong/Moderate/Weak]

## Forecast
- Next 6 months predicted sales: [Amount]
- Confidence level: [High/Medium/Low]

## Recommendations
1. [Action item 1]
2. [Action item 2]
3. [Action item 3]
```

---

## 🤖 Tutorial 3: Customer Segmentation ML Project
*Duration: 90 minutes | Level: Advanced*

### What You'll Learn
- End-to-end ML workflow
- Customer segmentation techniques
- Model evaluation and validation
- Business interpretation

### Project Scenario
You're a data scientist at an e-commerce company tasked with segmenting customers for targeted marketing campaigns.

#### Project Setup (10 minutes)

**Create Realistic Customer Data:**
```python
# generate_customer_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)

# Generate customer segments using make_blobs
n_customers = 1000
n_segments = 4

# Generate core features
X, y = make_blobs(n_samples=n_customers, centers=n_segments,
                  n_features=2, cluster_std=15, random_state=42)

# Create realistic customer features
data = pd.DataFrame({
    'customer_id': [f'CUST_{i:04d}' for i in range(n_customers)],
    'age': np.random.normal(40, 15, n_customers).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.5, n_customers).astype(int),
    'spending_score': (X[:, 0] * 2 + 50).clip(1, 100).astype(int),
    'annual_purchases': (X[:, 1] * 0.5 + 20).clip(1, 50).astype(int),
    'years_customer': np.random.exponential(3, n_customers).clip(0.1, 15).round(1),
    'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_customers),
    'channel_preference': np.random.choice(['Online', 'Store', 'Mobile'], n_customers, p=[0.5, 0.3, 0.2]),
    'true_segment': y  # Hidden ground truth for validation
})

# Add some business logic
data.loc[data['age'] < 30, 'preferred_category'] = np.random.choice(['Electronics', 'Clothing'],
                                                                   sum(data['age'] < 30), p=[0.7, 0.3])
data.loc[data['income'] > 80000, 'spending_score'] += 20
data['spending_score'] = data['spending_score'].clip(1, 100)

# Save without the true_segment column (that's cheating!)
customer_data = data.drop(['true_segment'], axis=1)
customer_data.to_csv('customer_data.csv', index=False)

# Save ground truth for validation
data[['customer_id', 'true_segment']].to_csv('customer_segments_truth.csv', index=False)

print(f"Generated {n_customers} customers with {n_segments} hidden segments")
print("Files created: customer_data.csv, customer_segments_truth.csv")
```

```bash
python generate_customer_data.py
```

#### Phase 1: Exploratory Data Analysis (20 minutes)

**Step 1: Data Quality Assessment**
```bash
echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_segmentation/01_eda"
}' | python python/analyzers/basic/descriptive_stats.py

echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_segmentation/01_eda"
}' | python python/analyzers/basic/missing_data.py
```

**Step 2: Feature Distribution Analysis**
```bash
echo '{
  "data_file": "customer_data.csv",
  "numeric_columns": ["age", "income", "spending_score", "annual_purchases"],
  "plot_types": ["distribution", "qq"],
  "output_dir": "customer_segmentation/01_eda"
}' | python python/visualizations/statistical_plots.py
```

**Step 3: Correlation Exploration**
```bash
echo '{
  "data_file": "customer_data.csv",
  "columns": ["age", "income", "spending_score", "annual_purchases", "years_customer"],
  "method": "pearson",
  "output_dir": "customer_segmentation/01_eda"
}' | python python/analyzers/basic/correlation.py
```

**Step 4: Categorical Analysis**
```bash
echo '{
  "data_file": "customer_data.csv",
  "categorical_columns": ["preferred_category", "channel_preference"],
  "numeric_columns": ["spending_score", "annual_purchases"],
  "plot_types": ["bar", "box", "heatmap"],
  "output_dir": "customer_segmentation/01_eda"
}' | python python/visualizations/categorical_plots.py
```

#### Phase 2: Feature Engineering (15 minutes)

**Step 5: Advanced Feature Creation**
```bash
echo '{
  "data_file": "customer_data.csv",
  "techniques": ["scaling", "interaction", "polynomial"],
  "polynomial_degree": 2,
  "output_dir": "customer_segmentation/02_features"
}' | python python/ml/preprocessing/advanced_feature_engineering.py
```

**Step 6: Outlier Detection**
```bash
echo '{
  "data_file": "customer_data.csv",
  "columns": ["age", "income", "spending_score", "annual_purchases"],
  "methods": ["iqr", "isolation_forest"],
  "contamination": 0.05,
  "output_dir": "customer_segmentation/02_features"
}' | python python/analyzers/advanced/outlier_detection.py
```

#### Phase 3: Clustering Analysis (25 minutes)

**Step 7: Multiple Clustering Algorithms**
```bash
echo '{
  "data_file": "customer_segmentation/02_features/engineered_data.csv",
  "feature_columns": ["age", "income", "spending_score", "annual_purchases", "years_customer"],
  "algorithms": ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"],
  "n_clusters": 4,
  "output_dir": "customer_segmentation/03_clustering"
}' | python python/ml/unsupervised/clustering.py
```

**Step 8: Cluster Validation and Optimization**

Create a cluster evaluation script:
```python
# cluster_evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def evaluate_clusters(data_file, feature_cols, max_k=10):
    """Evaluate optimal number of clusters"""
    data = pd.read_csv(data_file)
    X = data[feature_cols]

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate different numbers of clusters
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Plot evaluation metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('customer_segmentation/03_clustering/cluster_evaluation.png', dpi=300)
    plt.close()

    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]

    return {
        'optimal_k': optimal_k,
        'best_silhouette': max(silhouette_scores),
        'evaluation_saved': 'customer_segmentation/03_clustering/cluster_evaluation.png'
    }

# Run evaluation
result = evaluate_clusters('customer_data.csv',
                          ['age', 'income', 'spending_score', 'annual_purchases', 'years_customer'])
print(f"Optimal number of clusters: {result['optimal_k']}")
```

```bash
python cluster_evaluation.py
```

#### Phase 4: Cluster Interpretation (15 minutes)

**Step 9: Segment Profiling**
```python
# segment_profiler.py
import pandas as pd
import numpy as np

def profile_segments(data_file, cluster_file):
    """Create detailed segment profiles"""

    # Load data and cluster results
    data = pd.read_csv(data_file)

    # Assuming KMeans results are in JSON format
    import json
    with open(cluster_file, 'r') as f:
        cluster_results = json.load(f)

    # Add cluster labels to data
    data['cluster'] = cluster_results['clustering_results']['kmeans']['cluster_labels']

    # Calculate segment profiles
    segment_profiles = {}

    for cluster_id in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_id]

        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100,
            'demographics': {
                'avg_age': cluster_data['age'].mean(),
                'age_range': f"{cluster_data['age'].min()}-{cluster_data['age'].max()}",
                'avg_income': cluster_data['income'].mean(),
                'income_std': cluster_data['income'].std()
            },
            'behavior': {
                'avg_spending_score': cluster_data['spending_score'].mean(),
                'avg_annual_purchases': cluster_data['annual_purchases'].mean(),
                'avg_years_customer': cluster_data['years_customer'].mean()
            },
            'preferences': {
                'top_category': cluster_data['preferred_category'].mode().iloc[0],
                'top_channel': cluster_data['channel_preference'].mode().iloc[0]
            }
        }

        segment_profiles[f'Segment_{cluster_id}'] = profile

    # Create business-friendly segment names
    segment_names = {
        'Segment_0': 'Budget Conscious',
        'Segment_1': 'Premium Customers',
        'Segment_2': 'Occasional Shoppers',
        'Segment_3': 'Loyal Enthusiasts'
    }

    # Generate segment summary
    summary = {
        'total_customers': len(data),
        'segments_identified': len(segment_profiles),
        'segment_profiles': segment_profiles,
        'business_segments': segment_names,
        'recommendations': generate_recommendations(segment_profiles)
    }

    return summary

def generate_recommendations(profiles):
    """Generate business recommendations for each segment"""
    recommendations = {}

    for segment, profile in profiles.items():
        recs = []

        # High spenders
        if profile['behavior']['avg_spending_score'] > 70:
            recs.append("Target with premium products and exclusive offers")
            recs.append("Implement VIP loyalty program")

        # Low engagement
        elif profile['behavior']['avg_spending_score'] < 30:
            recs.append("Focus on value propositions and discounts")
            recs.append("Implement re-engagement campaigns")

        # Long-term customers
        if profile['behavior']['avg_years_customer'] > 5:
            recs.append("Reward loyalty with special recognition")
            recs.append("Use for referral programs")

        recommendations[segment] = recs

    return recommendations

# Run profiling
summary = profile_segments('customer_data.csv',
                          'customer_segmentation/03_clustering/clustering_results.json')

# Save detailed report
import json
with open('customer_segmentation/04_interpretation/segment_profiles.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("Segment profiling complete!")
print(f"Identified {summary['segments_identified']} customer segments")
```

**Step 10: Validation Against Ground Truth**
```python
# validate_segmentation.py
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def validate_clustering():
    """Compare our clustering with ground truth"""

    # Load our clustering results and ground truth
    data = pd.read_csv('customer_data.csv')
    truth = pd.read_csv('customer_segments_truth.csv')

    # Load our clustering results (you'll need to adapt this to your output format)
    import json
    with open('customer_segmentation/03_clustering/clustering_results.json', 'r') as f:
        results = json.load(f)

    predicted_labels = results['clustering_results']['kmeans']['cluster_labels']
    true_labels = truth['true_segment'].values

    # Calculate validation metrics
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

    validation_report = {
        'adjusted_rand_index': ari_score,
        'normalized_mutual_information': nmi_score,
        'interpretation': {
            'ari': 'Perfect clustering' if ari_score > 0.9 else 'Good clustering' if ari_score > 0.7 else 'Fair clustering' if ari_score > 0.5 else 'Poor clustering',
            'nmi': 'High similarity' if nmi_score > 0.7 else 'Moderate similarity' if nmi_score > 0.5 else 'Low similarity'
        }
    }

    print(f"Validation Results:")
    print(f"Adjusted Rand Index: {ari_score:.3f}")
    print(f"Normalized Mutual Information: {nmi_score:.3f}")

    return validation_report

# Run validation
validation = validate_clustering()
```

#### Phase 5: Business Application (5 minutes)

**Step 11: Create Marketing Campaign Recommendations**

Create a final business report:
```markdown
# Customer Segmentation Analysis Report

## Executive Summary
- Analyzed 1,000 customers using machine learning clustering
- Identified 4 distinct customer segments
- Achieved clustering quality score of [X]

## Segment Profiles

### Segment 1: Premium Customers (25%)
- **Demographics**: Age 35-50, Income $75K+
- **Behavior**: High spending score (80+), frequent purchases
- **Strategy**: VIP programs, premium product launches

### Segment 2: Budget Conscious (30%)
- **Demographics**: Age 25-40, Income $30-50K
- **Behavior**: Low spending score (20-40), price-sensitive
- **Strategy**: Value deals, discount campaigns

### Segment 3: Loyal Enthusiasts (20%)
- **Demographics**: Age 40-60, varied income
- **Behavior**: Long tenure (5+ years), moderate spending
- **Strategy**: Loyalty rewards, referral programs

### Segment 4: Occasional Shoppers (25%)
- **Demographics**: Age 20-35, varied income
- **Behavior**: Infrequent purchases, low engagement
- **Strategy**: Re-engagement campaigns, targeted offers

## Next Steps
1. Implement targeted marketing campaigns
2. Develop segment-specific product recommendations
3. Monitor segment migration over time
4. A/B test campaign effectiveness
```

#### Tutorial Wrap-up

**What You've Accomplished:**
1. ✅ Complete ML project workflow
2. ✅ Advanced clustering techniques
3. ✅ Model validation and interpretation
4. ✅ Business application of results

**Key Takeaways:**
- Always validate your clustering results
- Business interpretation is as important as technical accuracy
- Multiple algorithms can provide different insights
- Segment profiling drives actionable recommendations

---

## 🏥 Tutorial 4: Healthcare Data Analysis
*Duration: 45 minutes | Level: Domain-Specific*

### Medical Research Scenario
Analyzing patient data to identify risk factors for a specific condition.

### Key Considerations for Healthcare Data
- **Privacy**: Always use anonymized data
- **Compliance**: Follow HIPAA guidelines
- **Validation**: Medical decisions require high confidence
- **Interpretability**: Models must be explainable

#### Sample Healthcare Workflow

**Phase 1: Data Preparation**
```bash
# Check for missing critical values
echo '{
  "data_file": "patient_data.csv",
  "strategy": "analyze",
  "output_dir": "healthcare_analysis/data_quality"
}' | python python/analyzers/basic/missing_data.py
```

**Phase 2: Risk Factor Analysis**
```bash
# Statistical significance testing
echo '{
  "data_file": "patient_data.csv",
  "numeric_columns": ["age", "bmi", "blood_pressure", "cholesterol"],
  "target_column": "condition_present",
  "plot_types": ["distribution", "confidence"],
  "output_dir": "healthcare_analysis/risk_factors"
}' | python python/visualizations/statistical_plots.py
```

**Phase 3: Predictive Modeling with Interpretability**
```bash
# Use interpretable algorithms
echo '{
  "data_file": "patient_data.csv",
  "target_column": "condition_present",
  "algorithms": ["logistic", "random_forest"],  # Interpretable models
  "cross_validation": 10,  # Rigorous validation
  "output_dir": "healthcare_analysis/models"
}' | python python/ml/supervised/classification/classification_trainer.py
```

---

## 💰 Tutorial 5: Financial Risk Assessment
*Duration: 45 minutes | Level: Domain-Specific*

### Banking Scenario
Assessing loan default risk using customer financial profiles.

#### Financial Analysis Workflow

**Phase 1: Fraud Detection**
```bash
echo '{
  "data_file": "loan_applications.csv",
  "methods": ["isolation_forest", "lof"],
  "contamination": 0.02,
  "output_dir": "risk_analysis/fraud_detection"
}' | python python/analyzers/advanced/outlier_detection.py
```

**Phase 2: Risk Profiling**
```bash
echo '{
  "data_file": "loan_applications.csv",
  "categorical_columns": ["employment_type", "loan_purpose"],
  "numeric_columns": ["income", "debt_ratio", "credit_score"],
  "plot_types": ["box", "heatmap"],
  "output_dir": "risk_analysis/profiling"
}' | python python/visualizations/categorical_plots.py
```

**Phase 3: Default Prediction**
```bash
echo '{
  "data_file": "loan_applications.csv",
  "target_column": "default",
  "algorithms": ["logistic", "gradient_boosting"],
  "class_weight": "balanced",
  "output_dir": "risk_analysis/models"
}' | python python/ml/supervised/classification/classification_trainer.py
```

---

## 🛒 Tutorial 6: E-commerce Analytics
*Duration: 45 minutes | Level: Domain-Specific*

### Retail Scenario
Optimizing product recommendations and inventory management.

#### E-commerce Workflow

**Phase 1: Product Performance Analysis**
```bash
echo '{
  "data_file": "sales_transactions.csv",
  "date_column": "transaction_date",
  "value_columns": ["revenue", "units_sold", "profit_margin"],
  "plot_types": ["line", "seasonal_decompose"],
  "output_dir": "ecommerce_analysis/performance"
}' | python python/visualizations/time_series_plots.py
```

**Phase 2: Customer Behavior Clustering**
```bash
echo '{
  "data_file": "customer_behavior.csv",
  "feature_columns": ["recency", "frequency", "monetary"],
  "algorithms": ["kmeans"],
  "n_clusters": 5,
  "output_dir": "ecommerce_analysis/segments"
}' | python python/ml/unsupervised/clustering.py
```

**Phase 3: Demand Forecasting**
```bash
echo '{
  "data_file": "product_sales.csv",
  "date_column": "date",
  "value_column": "daily_sales",
  "forecast_periods": 30,
  "models": ["arima", "exponential_smoothing"],
  "output_dir": "ecommerce_analysis/forecasting"
}' | python python/ml/time_series/forecasting.py
```

---

## 🎯 Best Practices Across All Tutorials

### 1. Always Start with Data Quality
```bash
# Your first two commands should always be:
python python/analyzers/basic/descriptive_stats.py
python python/analyzers/basic/missing_data.py
```

### 2. Visualize Before Modeling
- Understand your data distribution
- Check for obvious patterns
- Identify potential issues

### 3. Validate Everything
- Use cross-validation
- Check for overfitting
- Validate business assumptions

### 4. Document Your Process
- Save all parameters used
- Document decisions made
- Create reproducible workflows

### 5. Think Business First
- Start with business questions
- Ensure technical solutions solve real problems
- Communicate results clearly

---

## 🚀 Next Steps

### Continue Learning
1. **Advanced Tutorials**: Explore domain-specific guides
2. **Custom Development**: Create your own analyzers
3. **Production Deployment**: Scale your solutions
4. **Community**: Share your workflows and learn from others

### Get Help
- 📖 [API Reference](API_REFERENCE.md)
- 💼 [Usage Examples](USAGE_EXAMPLES.md)
- 👨‍💻 [Developer Guide](DEVELOPER_GUIDE.md)
- 🐛 [Troubleshooting](TROUBLESHOOTING.md)

---

*Ready to tackle your own data challenges? Start with Tutorial 1 and work your way up!*