"""
Jupyter Notebook to ML Pipeline Transformer
Converts exploratory Jupyter notebooks into production-ready ML pipelines

Features:
- Parse Jupyter notebooks (.ipynb)
- Extract data loading, preprocessing, training, evaluation code
- Generate scikit-learn Pipeline objects
- Create standalone Python scripts
- Organize code into modular functions
- Generate configuration files
- Support for multiple ML frameworks

Usage:
    from notebook_to_pipeline import NotebookToPipeline

    transformer = NotebookToPipeline('analysis.ipynb')
    transformer.parse_notebook()
    transformer.generate_pipeline('ml_pipeline.py')

CLI:
    python -m python.ml.pipeline.notebook_to_pipeline \\
        --notebook analysis.ipynb \\
        --output ml_pipeline.py \\
        --framework sklearn
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import nbformat
from collections import defaultdict


class NotebookToPipeline:
    """
    Transform Jupyter notebooks into production ML pipelines

    Analyzes notebook structure, extracts key components, and generates
    organized pipeline code
    """

    def __init__(self, notebook_path: str, framework: str = 'sklearn'):
        """
        Initialize transformer

        Args:
            notebook_path: Path to Jupyter notebook (.ipynb)
            framework: ML framework ('sklearn', 'pytorch', 'tensorflow', 'auto')
        """
        self.notebook_path = Path(notebook_path)
        self.framework = framework

        # Notebook content
        self.notebook = None
        self.cells = []

        # Extracted components
        self.imports = []
        self.data_loading = []
        self.preprocessing = []
        self.feature_engineering = []
        self.model_training = []
        self.model_evaluation = []
        self.prediction = []
        self.visualization = []
        self.utility_functions = []

        # Pipeline metadata
        self.detected_framework = None
        self.variables = {}
        self.dependencies = set()

    def parse_notebook(self) -> Dict[str, Any]:
        """
        Parse Jupyter notebook and extract components

        Returns:
            Dictionary with parsed components
        """
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {self.notebook_path}")

        # Load notebook
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)

        # Extract code cells
        self.cells = [cell for cell in self.notebook.cells if cell.cell_type == 'code']

        # Analyze cells
        for idx, cell in enumerate(self.cells):
            source = cell.source
            if not source.strip():
                continue

            # Categorize cell content
            self._categorize_cell(source, idx)

        # Detect ML framework
        self._detect_framework()

        # Extract dependencies
        self._extract_dependencies()

        return {
            'total_cells': len(self.cells),
            'imports': len(self.imports),
            'data_loading': len(self.data_loading),
            'preprocessing': len(self.preprocessing),
            'feature_engineering': len(self.feature_engineering),
            'model_training': len(self.model_training),
            'model_evaluation': len(self.model_evaluation),
            'framework': self.detected_framework,
            'dependencies': list(self.dependencies)
        }

    def _categorize_cell(self, source: str, cell_idx: int):
        """Categorize cell content into pipeline components"""

        # Imports
        if re.search(r'^(import|from)\s+', source, re.MULTILINE):
            self.imports.append({'code': source, 'cell': cell_idx})
            self._extract_imports_from_code(source)

        # Data loading patterns
        if any(pattern in source.lower() for pattern in [
            'read_csv', 'read_excel', 'read_json', 'read_sql',
            'load_data', 'pd.read', 'np.load', 'load_dataset'
        ]):
            self.data_loading.append({'code': source, 'cell': cell_idx})

        # Preprocessing patterns
        if any(pattern in source.lower() for pattern in [
            'fillna', 'dropna', 'drop_duplicates', 'standardscaler',
            'normalize', 'minmaxscaler', 'imputer', 'encode', 'labelencoder',
            'onehotencoder', 'preprocessing'
        ]):
            self.preprocessing.append({'code': source, 'cell': cell_idx})

        # Feature engineering
        if any(pattern in source.lower() for pattern in [
            'feature_selection', 'selectkbest', 'pca', 'transform',
            'featureengineering', 'create_features', 'polynomial'
        ]):
            self.feature_engineering.append({'code': source, 'cell': cell_idx})

        # Model training
        if any(pattern in source.lower() for pattern in [
            '.fit(', 'train', 'randomforest', 'xgboost', 'lightgbm',
            'logisticregression', 'svm', 'naivebayes', 'knn',
            'gradient', 'model.compile', 'model.build'
        ]):
            self.model_training.append({'code': source, 'cell': cell_idx})

        # Model evaluation
        if any(pattern in source.lower() for pattern in [
            'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix',
            'classification_report', 'roc_auc', 'mean_squared_error',
            'r2_score', 'evaluate', 'score', 'metrics'
        ]):
            self.model_evaluation.append({'code': source, 'cell': cell_idx})

        # Prediction
        if any(pattern in source.lower() for pattern in [
            '.predict(', '.predict_proba(', 'prediction'
        ]):
            self.prediction.append({'code': source, 'cell': cell_idx})

        # Visualization
        if any(pattern in source.lower() for pattern in [
            'plt.', 'plot', 'seaborn', 'sns.', 'fig', 'ax', 'scatter', 'hist'
        ]):
            self.visualization.append({'code': source, 'cell': cell_idx})

        # Function definitions
        if re.search(r'^def\s+\w+\s*\(', source, re.MULTILINE):
            self.utility_functions.append({'code': source, 'cell': cell_idx})

    def _extract_imports_from_code(self, source: str):
        """Extract import statements"""
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.dependencies.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.dependencies.add(node.module.split('.')[0])
        except:
            pass

    def _detect_framework(self):
        """Detect ML framework used in notebook"""
        framework_keywords = {
            'sklearn': ['sklearn', 'scikit-learn', 'from sklearn'],
            'pytorch': ['torch', 'pytorch', 'from torch'],
            'tensorflow': ['tensorflow', 'tf.', 'keras', 'from tensorflow'],
            'xgboost': ['xgboost', 'xgb.'],
            'lightgbm': ['lightgbm', 'lgb.']
        }

        all_code = ' '.join([cell['code'] for cell in self.imports + self.model_training])

        for framework, keywords in framework_keywords.items():
            if any(keyword in all_code.lower() for keyword in keywords):
                self.detected_framework = framework
                return

        self.detected_framework = self.framework

    def _extract_dependencies(self):
        """Extract all dependencies from imports"""
        # Already extracted in _extract_imports_from_code
        pass

    def generate_pipeline(
        self,
        output_path: str,
        include_tests: bool = False,
        include_config: bool = True
    ) -> Dict[str, str]:
        """
        Generate production pipeline code

        Args:
            output_path: Path for output Python file
            include_tests: Whether to generate test file
            include_config: Whether to generate config file

        Returns:
            Dictionary with generated file paths
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate main pipeline
        pipeline_code = self._generate_pipeline_code()

        # Write pipeline file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pipeline_code)

        result = {'pipeline': str(output_path)}

        # Generate config file
        if include_config:
            config_path = output_path.parent / f"{output_path.stem}_config.json"
            config = self._generate_config()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            result['config'] = str(config_path)

        # Generate test file
        if include_tests:
            test_path = output_path.parent / f"test_{output_path.name}"
            test_code = self._generate_test_code()
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            result['tests'] = str(test_path)

        return result

    def _generate_pipeline_code(self) -> str:
        """Generate complete pipeline code"""

        code_parts = []

        # Header
        code_parts.append(self._generate_header())

        # Imports
        code_parts.append(self._generate_imports_section())

        # Configuration
        code_parts.append(self._generate_config_section())

        # Utility functions
        if self.utility_functions:
            code_parts.append(self._generate_utility_section())

        # Data loading
        code_parts.append(self._generate_data_loading_function())

        # Preprocessing
        code_parts.append(self._generate_preprocessing_function())

        # Feature engineering
        if self.feature_engineering:
            code_parts.append(self._generate_feature_engineering_function())

        # Model training
        code_parts.append(self._generate_training_function())

        # Model evaluation
        code_parts.append(self._generate_evaluation_function())

        # Prediction
        code_parts.append(self._generate_prediction_function())

        # Main pipeline class
        code_parts.append(self._generate_pipeline_class())

        # CLI interface
        code_parts.append(self._generate_cli_section())

        return '\n\n'.join(code_parts)

    def _generate_header(self) -> str:
        """Generate file header"""
        return f'''"""
ML Pipeline
Generated from Jupyter notebook: {self.notebook_path.name}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This pipeline was automatically generated from exploratory notebook code.
Review and modify as needed for production use.
"""'''

    def _generate_imports_section(self) -> str:
        """Generate consolidated imports"""
        imports = set()

        # Standard library
        imports.add('import os')
        imports.add('import json')
        imports.add('from pathlib import Path')
        imports.add('from typing import Dict, Any, Optional, Tuple')

        # Data processing
        imports.add('import numpy as np')
        imports.add('import pandas as pd')

        # Add notebook imports
        for imp in self.imports:
            code = imp['code'].strip()
            for line in code.split('\n'):
                if line.strip() and (line.startswith('import ') or line.startswith('from ')):
                    imports.add(line.strip())

        # ML framework specific
        if self.detected_framework == 'sklearn':
            imports.add('from sklearn.pipeline import Pipeline')
            imports.add('from sklearn.preprocessing import StandardScaler')
            imports.add('import joblib')

        return '\n'.join(sorted(imports))

    def _generate_config_section(self) -> str:
        """Generate configuration section"""
        return '''
# Configuration
CONFIG = {
    'data': {
        'input_path': 'data/input',
        'output_path': 'data/output',
        'test_size': 0.2,
        'random_state': 42
    },
    'preprocessing': {
        'handle_missing': True,
        'scale_features': True,
        'encode_categorical': True
    },
    'model': {
        'save_path': 'models',
        'checkpoint_dir': 'checkpoints'
    }
}
'''

    def _generate_utility_section(self) -> str:
        """Generate utility functions section"""
        parts = ['# Utility Functions\n']

        for func in self.utility_functions:
            parts.append(func['code'])

        return '\n\n'.join(parts)

    def _generate_data_loading_function(self) -> str:
        """Generate data loading function"""
        code = '''
def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load and prepare data

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (features DataFrame, target Series if available)
    """
'''

        if self.data_loading:
            # Use first data loading cell as template
            data_code = self.data_loading[0]['code']
            # Indent and clean
            lines = data_code.split('\n')
            for line in lines:
                if line.strip():
                    code += f'    {line}\n'
        else:
            code += '''    input_path = config['data']['input_path']
    df = pd.read_csv(input_path)
    return df, None
'''

        return code

    def _generate_preprocessing_function(self) -> str:
        """Generate preprocessing function"""
        code = '''
def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess features

    Args:
        X: Features DataFrame
        y: Target Series (optional)
        config: Configuration dictionary

    Returns:
        Tuple of (processed features, target)
    """
'''

        if self.preprocessing:
            # Extract preprocessing logic
            for prep in self.preprocessing:
                lines = prep['code'].split('\n')
                for line in lines:
                    if line.strip() and not line.strip().startswith('#'):
                        code += f'    {line}\n'
        else:
            code += '''    if config and config['preprocessing']['handle_missing']:
        X = X.fillna(X.mean())

    return X, y
'''

        return code

    def _generate_feature_engineering_function(self) -> str:
        """Generate feature engineering function"""
        code = '''
def engineer_features(
    X: pd.DataFrame,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Engineer features

    Args:
        X: Features DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with engineered features
    """
'''

        if self.feature_engineering:
            for fe in self.feature_engineering:
                lines = fe['code'].split('\n')
                for line in lines:
                    if line.strip():
                        code += f'    {line}\n'
        else:
            code += '    return X\n'

        return code

    def _generate_training_function(self) -> str:
        """Generate model training function"""
        code = '''
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any] = None
) -> Any:
    """
    Train ML model

    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary

    Returns:
        Trained model
    """
'''

        if self.model_training:
            # Extract training logic
            train_code = self.model_training[0]['code']
            lines = train_code.split('\n')
            for line in lines:
                if line.strip():
                    code += f'    {line}\n'
        else:
            code += '''    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
'''

        return code

    def _generate_evaluation_function(self) -> str:
        """Generate evaluation function"""
        code = '''
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics
    """
'''

        if self.model_evaluation:
            eval_code = self.model_evaluation[0]['code']
            lines = eval_code.split('\n')
            for line in lines:
                if line.strip():
                    code += f'    {line}\n'
        else:
            code += '''    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics
'''

        return code

    def _generate_prediction_function(self) -> str:
        """Generate prediction function"""
        return '''
def make_predictions(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions on new data

    Args:
        model: Trained model
        X: Features DataFrame

    Returns:
        Array of predictions
    """
    return model.predict(X)
'''

    def _generate_pipeline_class(self) -> str:
        """Generate main pipeline class"""
        return '''
class MLPipeline:
    """
    Complete ML Pipeline

    Orchestrates data loading, preprocessing, training, and prediction
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pipeline

        Args:
            config: Configuration dictionary
        """
        self.config = config or CONFIG
        self.model = None
        self.preprocessor = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLPipeline':
        """
        Fit the complete pipeline

        Args:
            X: Training features
            y: Training target

        Returns:
            Self (fitted pipeline)
        """
        # Preprocess
        X_processed, y_processed = preprocess_data(X, y, self.config)

        # Feature engineering
        X_processed = engineer_features(X_processed, self.config)

        # Train model
        self.model = train_model(X_processed, y_processed, self.config)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features DataFrame

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Preprocess
        X_processed, _ = preprocess_data(X, None, self.config)

        # Feature engineering
        X_processed = engineer_features(X_processed, self.config)

        # Predict
        return make_predictions(self.model, X_processed)

    def save(self, path: str):
        """Save pipeline to disk"""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'MLPipeline':
        """Load pipeline from disk"""
        import joblib
        return joblib.load(path)
'''

    def _generate_cli_section(self) -> str:
        """Generate CLI interface"""
        return '''
# CLI Interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--test', type=str, help='Path to test data')
    parser.add_argument('--predict', type=str, help='Path to prediction data')
    parser.add_argument('--model-path', type=str, default='model.pkl', help='Model save/load path')
    parser.add_argument('--config', type=str, help='Config file path')

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = CONFIG

    # Initialize pipeline
    pipeline = MLPipeline(config)

    # Training mode
    if args.train:
        print(f"Loading training data from {args.train}")
        X, y = load_data({'data': {'input_path': args.train}})

        print("Training model...")
        pipeline.fit(X, y)

        print(f"Saving model to {args.model_path}")
        pipeline.save(args.model_path)

        if args.test:
            print(f"Evaluating on test data from {args.test}")
            X_test, y_test = load_data({'data': {'input_path': args.test}})
            metrics = evaluate_model(pipeline.model, X_test, y_test)
            print(f"Evaluation metrics: {metrics}")

    # Prediction mode
    elif args.predict:
        print(f"Loading model from {args.model_path}")
        pipeline = MLPipeline.load(args.model_path)

        print(f"Loading prediction data from {args.predict}")
        X, _ = load_data({'data': {'input_path': args.predict}})

        print("Making predictions...")
        predictions = pipeline.predict(X)

        print(f"Predictions: {predictions}")

        # Save predictions
        output_path = Path(args.predict).parent / 'predictions.csv'
        pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    else:
        parser.print_help()
'''

    def _generate_config(self) -> Dict[str, Any]:
        """Generate configuration file"""
        return {
            'notebook': str(self.notebook_path),
            'generated': datetime.now().isoformat(),
            'framework': self.detected_framework,
            'dependencies': list(self.dependencies),
            'components': {
                'data_loading': len(self.data_loading) > 0,
                'preprocessing': len(self.preprocessing) > 0,
                'feature_engineering': len(self.feature_engineering) > 0,
                'model_training': len(self.model_training) > 0,
                'model_evaluation': len(self.model_evaluation) > 0
            },
            'pipeline_config': {
                'data': {
                    'input_path': 'data/input.csv',
                    'output_path': 'data/output',
                    'test_size': 0.2,
                    'random_state': 42
                },
                'preprocessing': {
                    'handle_missing': True,
                    'scale_features': True,
                    'encode_categorical': True
                },
                'model': {
                    'save_path': 'models/model.pkl',
                    'checkpoint_dir': 'checkpoints'
                }
            }
        }

    def _generate_test_code(self) -> str:
        """Generate test file"""
        return f'''"""
Tests for ML Pipeline
Generated from: {self.notebook_path.name}
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from {self.notebook_path.stem}_pipeline import MLPipeline, load_data, preprocess_data


class TestMLPipeline(unittest.TestCase):
    """Test ML Pipeline"""

    def setUp(self):
        """Setup test fixtures"""
        self.pipeline = MLPipeline()

        # Create sample data
        self.X_train = pd.DataFrame({{
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        }})
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

        self.X_test = pd.DataFrame({{
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        }})

    def test_pipeline_fit(self):
        """Test pipeline fitting"""
        self.pipeline.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.pipeline.model)

    def test_pipeline_predict(self):
        """Test pipeline prediction"""
        self.pipeline.fit(self.X_train, self.y_train)
        predictions = self.pipeline.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in predictions))

    def test_preprocessing(self):
        """Test preprocessing function"""
        X_processed, y_processed = preprocess_data(self.X_train, self.y_train)

        self.assertEqual(X_processed.shape, self.X_train.shape)
        self.assertEqual(len(y_processed), len(self.y_train))

    def test_pipeline_save_load(self):
        """Test pipeline save/load"""
        import tempfile

        self.pipeline.fit(self.X_train, self.y_train)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            self.pipeline.save(f.name)
            loaded_pipeline = MLPipeline.load(f.name)

        self.assertIsNotNone(loaded_pipeline.model)

        # Compare predictions
        pred1 = self.pipeline.predict(self.X_test)
        pred2 = loaded_pipeline.predict(self.X_test)

        np.testing.assert_array_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()
'''

    def generate_summary(self) -> str:
        """Generate transformation summary"""
        summary = f"""
Notebook to Pipeline Transformation Summary
{'=' * 50}

Notebook: {self.notebook_path.name}
Framework: {self.detected_framework}
Total Cells: {len(self.cells)}

Components Extracted:
  - Imports: {len(self.imports)}
  - Data Loading: {len(self.data_loading)}
  - Preprocessing: {len(self.preprocessing)}
  - Feature Engineering: {len(self.feature_engineering)}
  - Model Training: {len(self.model_training)}
  - Model Evaluation: {len(self.model_evaluation)}
  - Predictions: {len(self.prediction)}
  - Visualizations: {len(self.visualization)}
  - Utility Functions: {len(self.utility_functions)}

Dependencies: {', '.join(sorted(self.dependencies))}

Generated Pipeline Structure:
  ✓ Data loading function
  ✓ Preprocessing function
  ✓ Feature engineering function (if applicable)
  ✓ Training function
  ✓ Evaluation function
  ✓ Prediction function
  ✓ Pipeline class
  ✓ CLI interface
"""
        return summary


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform Jupyter Notebook to ML Pipeline')
    parser.add_argument('--notebook', type=str, required=True, help='Path to Jupyter notebook')
    parser.add_argument('--output', type=str, required=True, help='Output pipeline file path')
    parser.add_argument('--framework', type=str, default='auto',
                        choices=['auto', 'sklearn', 'pytorch', 'tensorflow'],
                        help='ML framework')
    parser.add_argument('--include-tests', action='store_true', help='Generate test file')
    parser.add_argument('--include-config', action='store_true', default=True,
                        help='Generate config file')
    parser.add_argument('--summary', action='store_true', help='Print summary')

    args = parser.parse_args()

    # Initialize transformer
    transformer = NotebookToPipeline(args.notebook, framework=args.framework)

    # Parse notebook
    print(f"Parsing notebook: {args.notebook}")
    parse_result = transformer.parse_notebook()

    # Generate pipeline
    print(f"Generating pipeline: {args.output}")
    result = transformer.generate_pipeline(
        args.output,
        include_tests=args.include_tests,
        include_config=args.include_config
    )

    print("\n✓ Pipeline generation complete!")
    print(f"\nGenerated files:")
    for file_type, path in result.items():
        print(f"  - {file_type}: {path}")

    # Print summary
    if args.summary:
        print(transformer.generate_summary())

    print("\nNext steps:")
    print("  1. Review the generated pipeline code")
    print("  2. Modify configuration as needed")
    print("  3. Run tests to validate functionality")
    print(f"  4. Train model: python {args.output} --train data/train.csv")
    print(f"  5. Make predictions: python {args.output} --predict data/new.csv --model-path model.pkl")
