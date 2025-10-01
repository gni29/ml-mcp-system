#!/usr/bin/env python3
"""
Model Ensemble Module for ML MCP System
Combine multiple models for improved predictions
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error


class EnsembleModel:
    """Create ensemble models for improved predictions"""

    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble model

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_type = None

    def create_voting_classifier(self, models: Optional[List] = None,
                                 voting: str = 'soft') -> VotingClassifier:
        """
        Create voting classifier ensemble

        Args:
            models: List of (name, model) tuples (None = use defaults)
            voting: 'hard' or 'soft' voting

        Returns:
            VotingClassifier
        """
        if models is None:
            models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('svc', SVC(probability=True, random_state=self.random_state)),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ]

        self.model = VotingClassifier(estimators=models, voting=voting)
        self.model_type = 'voting_classifier'
        return self.model

    def create_voting_regressor(self, models: Optional[List] = None) -> VotingRegressor:
        """
        Create voting regressor ensemble

        Args:
            models: List of (name, model) tuples

        Returns:
            VotingRegressor
        """
        if models is None:
            models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('svr', SVR()),
                ('knn', KNeighborsRegressor(n_neighbors=5))
            ]

        self.model = VotingRegressor(estimators=models)
        self.model_type = 'voting_regressor'
        return self.model

    def create_stacking_classifier(self, base_models: Optional[List] = None,
                                   meta_model: Optional[Any] = None) -> StackingClassifier:
        """
        Create stacking classifier ensemble

        Args:
            base_models: List of (name, model) tuples for base models
            meta_model: Meta-model for final prediction

        Returns:
            StackingClassifier
        """
        if base_models is None:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('dt', DecisionTreeClassifier(max_depth=10, random_state=self.random_state))
            ]

        if meta_model is None:
            meta_model = LogisticRegression(random_state=self.random_state)

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
        self.model_type = 'stacking_classifier'
        return self.model

    def create_stacking_regressor(self, base_models: Optional[List] = None,
                                  meta_model: Optional[Any] = None) -> StackingRegressor:
        """
        Create stacking regressor ensemble

        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta-model for final prediction

        Returns:
            StackingRegressor
        """
        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('dt', DecisionTreeRegressor(max_depth=10, random_state=self.random_state))
            ]

        if meta_model is None:
            meta_model = Ridge()

        self.model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
        self.model_type = 'stacking_regressor'
        return self.model

    def train(self, X: pd.DataFrame, y: pd.Series,
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train ensemble model

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size

        Returns:
            Training results
        """
        if self.model is None:
            raise ValueError("No model created. Call create_* method first.")

        print(f"Training {self.model_type}...", file=sys.stderr)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        if 'classifier' in self.model_type:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')

            results = {
                'model_type': self.model_type,
                'accuracy': float(accuracy),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_samples': len(y_test)
            }

            # Individual model scores (for voting/stacking)
            if hasattr(self.model, 'estimators_'):
                individual_scores = {}
                for name, estimator in self.model.estimators_:
                    pred = estimator.predict(X_test)
                    score = accuracy_score(y_test, pred)
                    individual_scores[name] = float(score)
                results['individual_model_scores'] = individual_scores

        else:
            # Regression metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')

            results = {
                'model_type': self.model_type,
                'r2_score': float(r2),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_samples': len(y_test)
            }

            # Individual model scores
            if hasattr(self.model, 'estimators_'):
                individual_scores = {}
                for name, estimator in self.model.estimators_:
                    pred = estimator.predict(X_test)
                    score = r2_score(y_test, pred)
                    individual_scores[name] = float(score)
                results['individual_model_scores'] = individual_scores

        return results


class WeightedEnsemble:
    """Weighted ensemble of models"""

    def __init__(self):
        """Initialize weighted ensemble"""
        self.models = []
        self.weights = []
        self.task_type = None

    def add_model(self, model: Any, weight: float = 1.0):
        """
        Add model to ensemble

        Args:
            model: Trained model
            weight: Weight for this model's predictions
        """
        self.models.append(model)
        self.weights.append(weight)

    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification'):
        """
        Fit all models in ensemble

        Args:
            X: Feature DataFrame
            y: Target series
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type

        for model in self.models:
            model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted predictions

        Args:
            X: Feature DataFrame

        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Normalize weights
        weights = np.array(self.weights) / np.sum(self.weights)

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average
        if self.task_type == 'regression':
            # Average predictions
            weighted_pred = np.average(predictions, axis=0, weights=weights)
        else:
            # For classification, use weighted voting
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            weighted_pred = np.round(weighted_pred).astype(int)

        return weighted_pred


def compare_ensemble_methods(X: pd.DataFrame, y: pd.Series,
                            task_type: str = 'classification',
                            test_size: float = 0.2) -> Dict[str, Any]:
    """
    Compare different ensemble methods

    Args:
        X: Feature DataFrame
        y: Target series
        task_type: 'classification' or 'regression'
        test_size: Test set size

    Returns:
        Comparison results
    """
    results = {}

    ensemble = EnsembleModel()

    if task_type == 'classification':
        # Voting
        ensemble.create_voting_classifier(voting='soft')
        results['voting_soft'] = ensemble.train(X, y, test_size)

        # Stacking
        ensemble.create_stacking_classifier()
        results['stacking'] = ensemble.train(X, y, test_size)

    else:
        # Voting
        ensemble.create_voting_regressor()
        results['voting'] = ensemble.train(X, y, test_size)

        # Stacking
        ensemble.create_stacking_regressor()
        results['stacking'] = ensemble.train(X, y, test_size)

    return results


def main():
    """CLI interface"""
    if len(sys.argv) < 4:
        print("Usage: python model_ensemble.py <data_file> <target_column> <task_type> [ensemble_type]")
        print("task_type: classification or regression")
        print("ensemble_type: voting, stacking, or compare (default: voting)")
        sys.exit(1)

    data_file = sys.argv[1]
    target_column = sys.argv[2]
    task_type = sys.argv[3]
    ensemble_type = sys.argv[4] if len(sys.argv) > 4 else 'voting'

    try:
        # Load data
        df = pd.read_csv(data_file)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Select numeric columns
        X = X.select_dtypes(include=[np.number])

        if ensemble_type == 'compare':
            # Compare all methods
            results = compare_ensemble_methods(X, y, task_type)
        else:
            # Single ensemble method
            ensemble = EnsembleModel()

            if task_type == 'classification':
                if ensemble_type == 'voting':
                    ensemble.create_voting_classifier()
                elif ensemble_type == 'stacking':
                    ensemble.create_stacking_classifier()
            else:
                if ensemble_type == 'voting':
                    ensemble.create_voting_regressor()
                elif ensemble_type == 'stacking':
                    ensemble.create_stacking_regressor()

            results = ensemble.train(X, y)

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