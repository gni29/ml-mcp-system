#!/usr/bin/env python3
"""
AutoML Trainer Module for ML MCP System
Automatic model selection and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AutoMLTrainer:
    """Automatic ML model selection and training"""

    # Classifier models and their hyperparameter grids
    CLASSIFIER_MODELS = {
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'svm': {
            'model': SVC,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'knn': {
            'model': KNeighborsClassifier,
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }

    # Regressor models and their hyperparameter grids
    REGRESSOR_MODELS = {
        'random_forest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'ridge': {
            'model': Ridge,
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100]
            }
        },
        'lasso': {
            'model': Lasso,
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100]
            }
        },
        'svr': {
            'model': SVR,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'knn': {
            'model': KNeighborsRegressor,
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }

    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize AutoML trainer

        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed
        """
        self.task_type = task_type
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        self.all_results = {}

    def auto_train(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   cv_folds: int = 5,
                   search_method: str = 'random',
                   n_iter: int = 20,
                   models_to_try: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Automatically select best model and hyperparameters

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size
            cv_folds: Number of cross-validation folds
            search_method: 'grid' or 'random' search
            n_iter: Number of iterations for random search
            models_to_try: List of model names to try (None = all)

        Returns:
            Results dictionary with best model and metrics
        """
        print(f"Starting AutoML for {self.task_type}...", file=sys.stderr)

        # Check minimum sample size
        if len(X) < cv_folds * 2:
            cv_folds = max(2, len(X) // 2)
            print(f"Reducing CV folds to {cv_folds} due to small sample size", file=sys.stderr)

        # Prepare data
        if self.task_type == 'classification' and self.label_encoder:
            y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Select models to evaluate
        if self.task_type == 'classification':
            models_dict = self.CLASSIFIER_MODELS
        else:
            models_dict = self.REGRESSOR_MODELS

        if models_to_try:
            models_dict = {k: v for k, v in models_dict.items() if k in models_to_try}

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            if self.task_type == 'classification':
                models_dict['xgboost'] = {
                    'model': xgb.XGBClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                }
            else:
                models_dict['xgboost'] = {
                    'model': xgb.XGBRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                }

        # Evaluate each model
        best_score = -np.inf if self.task_type == 'regression' else -np.inf

        for model_name, model_config in models_dict.items():
            print(f"Evaluating {model_name}...", file=sys.stderr)

            try:
                model_class = model_config['model']
                param_grid = model_config['params']

                # Create base model
                if model_name in ['svr', 'knn']:
                    # SVR and KNN don't have random_state
                    base_model = model_class()
                else:
                    base_model = model_class(random_state=self.random_state)

                # Hyperparameter search
                if search_method == 'grid':
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=cv_folds,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2',
                        n_jobs=-1
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        cv=cv_folds,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2',
                        n_iter=n_iter,
                        random_state=self.random_state,
                        n_jobs=-1
                    )

                search.fit(X_train_scaled, y_train)

                # Evaluate on test set
                y_pred = search.best_estimator_.predict(X_test_scaled)

                if self.task_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                    metric_name = 'accuracy'
                else:
                    score = r2_score(y_test, y_pred)
                    metric_name = 'r2_score'

                # Store results
                self.all_results[model_name] = {
                    'score': float(score),
                    'best_params': search.best_params_,
                    'cv_score': float(search.best_score_)
                }

                # Update best model
                if score > best_score:
                    best_score = score
                    self.best_model = search.best_estimator_
                    self.best_model_name = model_name
                    self.best_params = search.best_params_
                    self.best_score = score

                print(f"{model_name}: {metric_name}={score:.4f}", file=sys.stderr)

            except Exception as e:
                print(f"Error with {model_name}: {str(e)}", file=sys.stderr)
                continue

        # Final results
        results = {
            'task_type': self.task_type,
            'best_model': self.best_model_name,
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'all_models': self.all_results,
            'models_evaluated': len(self.all_results),
            'search_method': search_method
        }

        # Add detailed metrics
        y_pred_final = self.best_model.predict(X_test_scaled)

        if self.task_type == 'classification':
            results['test_metrics'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_final))
            }

            if self.label_encoder:
                results['classes'] = self.label_encoder.classes_.tolist()

        else:
            mse = mean_squared_error(y_test, y_pred_final)
            results['test_metrics'] = {
                'r2_score': float(r2_score(y_test, y_pred_final)),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse))
            }

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")

        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)

        if self.task_type == 'classification' and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions


class AutoFeatureSelector:
    """Automatic feature selection"""

    def __init__(self, method: str = 'variance', threshold: float = 0.01):
        """
        Initialize feature selector

        Args:
            method: 'variance', 'correlation', or 'importance'
            threshold: Threshold for selection
        """
        self.method = method
        self.threshold = threshold
        self.selected_features = None

    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[str]:
        """
        Select features automatically

        Args:
            X: Feature DataFrame
            y: Target series (required for some methods)

        Returns:
            List of selected feature names
        """
        if self.method == 'variance':
            # Remove low variance features
            variances = X.var()
            self.selected_features = variances[variances > self.threshold].index.tolist()

        elif self.method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
            self.selected_features = [col for col in X.columns if col not in to_drop]

        elif self.method == 'importance':
            # Use feature importance
            if y is None:
                raise ValueError("Target y required for importance-based selection")

            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            importances = pd.Series(model.feature_importances_, index=X.columns)
            self.selected_features = importances[importances > self.threshold].index.tolist()

        return self.selected_features


def main():
    """CLI interface"""
    if len(sys.argv) < 4:
        print("Usage: python auto_trainer.py <data_file> <target_column> <task_type>")
        print("task_type: classification or regression")
        sys.exit(1)

    data_file = sys.argv[1]
    target_column = sys.argv[2]
    task_type = sys.argv[3]

    try:
        # Load data
        df = pd.read_csv(data_file)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Select numeric columns
        X = X.select_dtypes(include=[np.number])

        # AutoML
        trainer = AutoMLTrainer(task_type=task_type)
        results = trainer.auto_train(X, y, search_method='random', n_iter=10)

        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()