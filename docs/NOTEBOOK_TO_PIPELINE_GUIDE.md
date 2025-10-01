# Jupyter Notebook to ML Pipeline Transformation Guide

Transform exploratory Jupyter notebooks into production-ready ML pipelines automatically.

---

## 🎯 Overview

The **Notebook to Pipeline** transformer analyzes your Jupyter notebooks and extracts:
- Data loading code
- Preprocessing steps
- Feature engineering
- Model training
- Evaluation metrics
- Prediction logic

It then generates:
- ✅ Structured Python pipeline script
- ✅ Configuration file (JSON)
- ✅ Test file (optional)
- ✅ CLI interface
- ✅ MLPipeline class for easy use

---

## 🚀 Quick Start

### Using Python Directly

```python
from python.ml.pipeline.notebook_to_pipeline import NotebookToPipeline

# Initialize transformer
transformer = NotebookToPipeline('analysis.ipynb', framework='sklearn')

# Parse notebook
parse_result = transformer.parse_notebook()
print(f"Found {parse_result['total_cells']} code cells")
print(f"Detected framework: {parse_result['framework']}")

# Generate pipeline
files = transformer.generate_pipeline(
    output_path='ml_pipeline.py',
    include_tests=True,
    include_config=True
)

print(f"Generated: {files}")

# Print summary
print(transformer.generate_summary())
```

### Using CLI

```bash
# Basic usage
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook analysis.ipynb \
  --output ml_pipeline.py

# With tests and summary
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook analysis.ipynb \
  --output ml_pipeline.py \
  --include-tests \
  --summary

# Specify framework
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook pytorch_model.ipynb \
  --output pytorch_pipeline.py \
  --framework pytorch
```

### Using MCP Tool

```javascript
await mcp.call('notebook_to_pipeline', {
  notebook_path: 'experiments/analysis.ipynb',
  output_path: 'pipelines/ml_pipeline.py',
  framework: 'sklearn',
  include_tests: true,
  include_config: true,
  show_summary: true
});
```

---

## 📋 What Gets Generated

### 1. Main Pipeline File (`ml_pipeline.py`)

```python
"""
ML Pipeline
Generated from Jupyter notebook: analysis.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configuration
CONFIG = {
    'data': {'input_path': 'data/input', 'test_size': 0.2},
    'preprocessing': {'handle_missing': True, 'scale_features': True},
    'model': {'save_path': 'models'}
}

def load_data(config):
    """Load and prepare data"""
    # Your notebook's data loading code here
    pass

def preprocess_data(X, y, config):
    """Preprocess features"""
    # Your notebook's preprocessing code here
    pass

def train_model(X_train, y_train, config):
    """Train ML model"""
    # Your notebook's training code here
    pass

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Your notebook's evaluation code here
    pass

class MLPipeline:
    """Complete ML Pipeline"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None

    def fit(self, X, y):
        """Fit the complete pipeline"""
        X_processed, y_processed = preprocess_data(X, y, self.config)
        self.model = train_model(X_processed, y_processed, self.config)
        return self

    def predict(self, X):
        """Make predictions"""
        X_processed, _ = preprocess_data(X, None, self.config)
        return self.model.predict(X_processed)

    def save(self, path):
        """Save pipeline"""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load pipeline"""
        import joblib
        return joblib.load(path)

# CLI Interface
if __name__ == '__main__':
    import argparse
    # Complete CLI implementation...
```

### 2. Configuration File (`ml_pipeline_config.json`)

```json
{
  "notebook": "analysis.ipynb",
  "generated": "2025-10-01T10:30:00",
  "framework": "sklearn",
  "dependencies": ["pandas", "numpy", "scikit-learn"],
  "components": {
    "data_loading": true,
    "preprocessing": true,
    "feature_engineering": true,
    "model_training": true,
    "model_evaluation": true
  },
  "pipeline_config": {
    "data": {
      "input_path": "data/input.csv",
      "test_size": 0.2,
      "random_state": 42
    },
    "preprocessing": {
      "handle_missing": true,
      "scale_features": true
    },
    "model": {
      "save_path": "models/model.pkl"
    }
  }
}
```

### 3. Test File (`test_ml_pipeline.py`)

```python
"""Tests for ML Pipeline"""

import unittest
import pandas as pd
import numpy as np
from ml_pipeline import MLPipeline

class TestMLPipeline(unittest.TestCase):
    """Test ML Pipeline"""

    def setUp(self):
        self.pipeline = MLPipeline()
        # Create sample data
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

    def test_pipeline_fit(self):
        """Test pipeline fitting"""
        self.pipeline.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.pipeline.model)

    def test_pipeline_predict(self):
        """Test pipeline prediction"""
        self.pipeline.fit(self.X_train, self.y_train)
        predictions = self.pipeline.predict(self.X_train)
        self.assertEqual(len(predictions), len(self.X_train))

if __name__ == '__main__':
    unittest.main()
```

---

## 💡 Usage Examples

### Example 1: Train and Save Model

```bash
# Train model
python ml_pipeline.py --train data/train.csv --model-path model.pkl

# With custom config
python ml_pipeline.py --train data/train.csv --config my_config.json --model-path model.pkl
```

### Example 2: Make Predictions

```bash
# Make predictions
python ml_pipeline.py --predict data/new_data.csv --model-path model.pkl
```

### Example 3: Train and Evaluate

```bash
# Train and evaluate in one command
python ml_pipeline.py \
  --train data/train.csv \
  --test data/test.csv \
  --model-path model.pkl
```

### Example 4: Use as Library

```python
from ml_pipeline import MLPipeline
import pandas as pd

# Load data
X_train = pd.read_csv('data/train.csv')
y_train = X_train['target']
X_train = X_train.drop('target', axis=1)

# Create and train pipeline
pipeline = MLPipeline()
pipeline.fit(X_train, y_train)

# Save model
pipeline.save('models/my_model.pkl')

# Later: Load and predict
pipeline = MLPipeline.load('models/my_model.pkl')
X_new = pd.read_csv('data/new_data.csv')
predictions = pipeline.predict(X_new)
```

---

## 🔍 What Gets Extracted

The transformer automatically identifies and extracts:

### 1. Data Loading
```python
# Detects patterns like:
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
data = np.load('data.npy')
```

### 2. Preprocessing
```python
# Detects patterns like:
df.fillna(df.mean())
df.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Feature Engineering
```python
# Detects patterns like:
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

### 4. Model Training
```python
# Detects patterns like:
model = RandomForestClassifier()
model.fit(X_train, y_train)
model = XGBClassifier()
model.fit(X_train, y_train)
```

### 5. Evaluation
```python
# Detects patterns like:
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

---

## ⚙️ Configuration Options

### Framework Detection

The transformer auto-detects the ML framework used:
- **sklearn**: scikit-learn models
- **pytorch**: PyTorch models
- **tensorflow**: TensorFlow/Keras models
- **xgboost**: XGBoost models
- **lightgbm**: LightGBM models

You can also specify manually:
```python
transformer = NotebookToPipeline('notebook.ipynb', framework='sklearn')
```

### Generate Options

```python
files = transformer.generate_pipeline(
    output_path='pipeline.py',       # Output file path
    include_tests=True,               # Generate test file
    include_config=True               # Generate config file
)
```

---

## 📊 Transformation Summary

After transformation, you get a detailed summary:

```
Notebook to Pipeline Transformation Summary
==================================================

Notebook: analysis.ipynb
Framework: sklearn
Total Cells: 25

Components Extracted:
  - Imports: 5
  - Data Loading: 2
  - Preprocessing: 4
  - Feature Engineering: 2
  - Model Training: 3
  - Model Evaluation: 2
  - Predictions: 1
  - Visualizations: 4
  - Utility Functions: 2

Dependencies: numpy, pandas, scikit-learn, matplotlib

Generated Pipeline Structure:
  ✓ Data loading function
  ✓ Preprocessing function
  ✓ Feature engineering function
  ✓ Training function
  ✓ Evaluation function
  ✓ Prediction function
  ✓ Pipeline class
  ✓ CLI interface
```

---

## 🎯 Best Practices

### 1. Organize Your Notebook
For best results, organize your notebook with clear sections:
- Data Loading
- Exploratory Data Analysis (EDA)
- Preprocessing
- Feature Engineering
- Model Training
- Evaluation
- Predictions

### 2. Use Descriptive Variable Names
```python
# Good
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Avoid
a, b, c, d = train_test_split(X, y)
```

### 3. Keep Functions Modular
```python
# Good - easier to extract
def preprocess_data(df):
    df = df.fillna(df.mean())
    return df

# Avoid - harder to extract
df = df.fillna(df.mean())  # scattered throughout notebook
```

### 4. Review Generated Code
Always review and test the generated pipeline:
```bash
# Run tests
python test_ml_pipeline.py

# Test with sample data
python ml_pipeline.py --train sample_data.csv --test sample_test.csv
```

---

## 🔧 Advanced Usage

### Custom Pipeline Templates

You can extend the transformer to use custom templates:

```python
class CustomNotebookToPipeline(NotebookToPipeline):
    def _generate_pipeline_class(self):
        # Your custom pipeline class generation
        return custom_code
```

### Integration with MLOps

Combine with MLflow for experiment tracking:

```python
from python.ml.mlops.mlflow_tracker import MLflowTracker
from ml_pipeline import MLPipeline

tracker = MLflowTracker(experiment_name='my_experiment')

with tracker.start_run():
    pipeline = MLPipeline()
    pipeline.fit(X_train, y_train)

    # Log metrics
    tracker.log_metrics({'accuracy': 0.95})

    # Log model
    tracker.log_model(pipeline.model, 'model')
```

### Deployment

Deploy the generated pipeline:

```python
from python.ml.deployment.model_server import ModelServer

# Save pipeline
pipeline.save('models/pipeline.pkl')

# Serve via API
server = ModelServer()
server.register_model('my_pipeline', 'models/pipeline.pkl', 'classifier')
server.start()
```

---

## 🐛 Troubleshooting

### Issue: Missing imports
**Solution**: Add required imports to your notebook's first cells

### Issue: Complex preprocessing not extracted
**Solution**: Refactor preprocessing into functions in your notebook

### Issue: Framework not detected
**Solution**: Specify framework explicitly:
```python
transformer = NotebookToPipeline('notebook.ipynb', framework='sklearn')
```

### Issue: Generated code doesn't run
**Solution**:
1. Review the generated code
2. Check that all dependencies are listed
3. Test with sample data first
4. Modify the generated code as needed

---

## 📚 Examples

See `examples/notebook_to_pipeline/` for complete examples:
- `example_sklearn.ipynb` - scikit-learn classification
- `example_pytorch.ipynb` - PyTorch deep learning
- `example_timeseries.ipynb` - Time series forecasting
- `example_nlp.ipynb` - NLP pipeline

---

## 🚀 Next Steps

After generating your pipeline:

1. **Review** the generated code
2. **Test** with sample data
3. **Customize** configuration as needed
4. **Integrate** with MLOps tools (MLflow, model serving)
5. **Deploy** to production

---

*Created: October 1, 2025*
*Version: 1.0.0*
*Part of ML MCP System v2.0.0*
