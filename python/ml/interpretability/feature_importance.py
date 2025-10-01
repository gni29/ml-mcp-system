#!/usr/bin/env python3
"""
Feature Importance Analyzer for ML MCP System
Analyze and visualize feature importance for ML models
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.base import is_classifier, is_regressor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportanceAnalyzer:
    """Analyze feature importance for ML models"""

    def __init__(self, model, X, y, feature_names: Optional[List[str]] = None):
        """
        Initialize feature importance analyzer

        Args:
            model: Trained ML model
            X: Feature data (DataFrame or array)
            y: Target data
            feature_names: Feature names (optional, inferred from X if DataFrame)
        """
        self.model = model
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.y = y

        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self.X.columns = self.feature_names

    def get_tree_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from tree-based models

        Returns:
            Feature importance results
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {
                'success': False,
                'error': 'Model does not have feature_importances_ attribute'
            }

        importances = self.model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1]

        results = {
            'method': 'tree_importance',
            'importances': {
                self.feature_names[i]: float(importances[i])
                for i in indices
            },
            'top_features': [
                {
                    'feature': self.feature_names[i],
                    'importance': float(importances[i])
                }
                for i in indices[:10]
            ],
            'summary': {
                'most_important': self.feature_names[indices[0]],
                'least_important': self.feature_names[indices[-1]],
                'mean_importance': float(np.mean(importances)),
                'std_importance': float(np.std(importances))
            }
        }

        return results

    def get_permutation_importance(self, n_repeats: int = 10,
                                   random_state: int = 42) -> Dict[str, Any]:
        """
        Calculate permutation importance

        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed

        Returns:
            Permutation importance results
        """
        try:
            result = permutation_importance(
                self.model, self.X, self.y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )

            importances_mean = result.importances_mean
            importances_std = result.importances_std

            # Sort by importance
            indices = np.argsort(importances_mean)[::-1]

            results = {
                'method': 'permutation_importance',
                'importances': {
                    self.feature_names[i]: {
                        'mean': float(importances_mean[i]),
                        'std': float(importances_std[i])
                    }
                    for i in indices
                },
                'top_features': [
                    {
                        'feature': self.feature_names[i],
                        'importance_mean': float(importances_mean[i]),
                        'importance_std': float(importances_std[i])
                    }
                    for i in indices[:10]
                ],
                'summary': {
                    'n_repeats': n_repeats,
                    'most_important': self.feature_names[indices[0]],
                    'least_important': self.feature_names[indices[-1]]
                }
            }

            return results

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def get_coefficient_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from linear model coefficients

        Returns:
            Coefficient importance results
        """
        if not hasattr(self.model, 'coef_'):
            return {
                'success': False,
                'error': 'Model does not have coef_ attribute'
            }

        # Handle multi-class (2D coefficients)
        if len(self.model.coef_.shape) > 1:
            # Average absolute coefficients across classes
            coef = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            coef = np.abs(self.model.coef_)

        # Sort by magnitude
        indices = np.argsort(coef)[::-1]

        results = {
            'method': 'coefficient_importance',
            'importances': {
                self.feature_names[i]: float(coef[i])
                for i in indices
            },
            'top_features': [
                {
                    'feature': self.feature_names[i],
                    'importance': float(coef[i])
                }
                for i in indices[:10]
            ],
            'summary': {
                'most_important': self.feature_names[indices[0]],
                'least_important': self.feature_names[indices[-1]],
                'mean_importance': float(np.mean(coef)),
                'std_importance': float(np.std(coef))
            }
        }

        return results

    def get_feature_importance(self, method: str = 'auto') -> Dict[str, Any]:
        """
        Get feature importance using appropriate method

        Args:
            method: 'auto', 'tree', 'permutation', or 'coefficient'

        Returns:
            Feature importance results
        """
        if method == 'auto':
            # Try tree-based first
            if hasattr(self.model, 'feature_importances_'):
                return self.get_tree_importance()
            # Try coefficient-based
            elif hasattr(self.model, 'coef_'):
                return self.get_coefficient_importance()
            # Fall back to permutation
            else:
                return self.get_permutation_importance()

        elif method == 'tree':
            return self.get_tree_importance()

        elif method == 'permutation':
            return self.get_permutation_importance()

        elif method == 'coefficient':
            return self.get_coefficient_importance()

        else:
            return {
                'success': False,
                'error': f'Unknown method: {method}'
            }

    def plot_feature_importance(self, method: str = 'auto',
                               top_n: int = 10,
                               output_path: Optional[str] = None) -> str:
        """
        Plot feature importance

        Args:
            method: Importance method
            top_n: Number of top features to show
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        importance_data = self.get_feature_importance(method)

        if 'error' in importance_data:
            raise ValueError(f"Cannot get importance: {importance_data['error']}")

        # Extract top features
        top_features = importance_data['top_features'][:top_n]
        features = [f['feature'] for f in top_features]

        if 'importance_mean' in top_features[0]:
            values = [f['importance_mean'] for f in top_features]
            stds = [f['importance_std'] for f in top_features]
            has_std = True
        else:
            values = [f['importance'] for f in top_features]
            has_std = False

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(features))

        if has_std:
            ax.barh(y_pos, values, xerr=stds, align='center', alpha=0.7)
        else:
            ax.barh(y_pos, values, align='center', alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance ({method})')
        plt.tight_layout()

        # Save plot
        if output_path is None:
            output_path = 'feature_importance.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def get_partial_dependence(self, features: List[str],
                               grid_resolution: int = 100) -> Dict[str, Any]:
        """
        Calculate partial dependence for features

        Args:
            features: List of feature names
            grid_resolution: Number of grid points

        Returns:
            Partial dependence results
        """
        try:
            # Get feature indices
            feature_indices = [self.feature_names.index(f) for f in features]

            # Calculate partial dependence
            pd_results = []

            for idx, feature_name in zip(feature_indices, features):
                pd_result = partial_dependence(
                    self.model,
                    self.X,
                    features=[idx],
                    grid_resolution=grid_resolution
                )

                pd_results.append({
                    'feature': feature_name,
                    'grid_values': pd_result['grid_values'][0].tolist(),
                    'average': pd_result['average'][0].tolist()
                })

            return {
                'method': 'partial_dependence',
                'features': features,
                'results': pd_results,
                'grid_resolution': grid_resolution
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def plot_partial_dependence(self, features: List[str],
                                output_path: Optional[str] = None) -> str:
        """
        Plot partial dependence

        Args:
            features: List of feature names
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        pd_data = self.get_partial_dependence(features)

        if 'error' in pd_data:
            raise ValueError(f"Cannot calculate PDP: {pd_data['error']}")

        # Create subplots
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, result in enumerate(pd_data['results']):
            ax = axes[idx]
            ax.plot(result['grid_values'], result['average'], linewidth=2)
            ax.set_xlabel(result['feature'])
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f"PDP: {result['feature']}")
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save plot
        if output_path is None:
            output_path = 'partial_dependence.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def detect_feature_interactions(self, top_n: int = 10,
                                    method: str = 'permutation') -> Dict[str, Any]:
        """
        Detect potential feature interactions

        Args:
            top_n: Number of top interactions to return
            method: Importance method to use

        Returns:
            Feature interaction results
        """
        importance_data = self.get_feature_importance(method)

        if 'error' in importance_data:
            return importance_data

        # Get top features
        top_features = [f['feature'] for f in importance_data['top_features'][:top_n]]

        # Calculate correlation matrix for top features
        corr_matrix = self.X[top_features].corr()

        # Find high correlations (potential interactions)
        interactions = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Threshold for interaction
                    interactions.append({
                        'feature_1': top_features[i],
                        'feature_2': top_features[j],
                        'correlation': float(corr),
                        'interaction_strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                    })

        # Sort by absolute correlation
        interactions.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return {
            'method': 'correlation_based_interaction',
            'top_features': top_features,
            'interactions': interactions[:top_n],
            'summary': {
                'total_interactions_found': len(interactions),
                'strong_interactions': sum(1 for x in interactions if abs(x['correlation']) > 0.7),
                'moderate_interactions': sum(1 for x in interactions if 0.5 < abs(x['correlation']) <= 0.7)
            }
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python feature_importance.py <action>")
        print("Actions: demo")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'demo':
            # Demo with synthetic data
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification

            print("Feature Importance Analyzer Demo")
            print("=" * 50)

            # Generate data
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                      n_redundant=2, random_state=42)
            feature_names = [f'feature_{i}' for i in range(10)]
            X_df = pd.DataFrame(X, columns=feature_names)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_df, y)

            # Analyze
            analyzer = FeatureImportanceAnalyzer(model, X_df, y)

            # Get importance
            importance = analyzer.get_feature_importance(method='tree')
            print("\nTop 5 Features:")
            for i, feat in enumerate(importance['top_features'][:5], 1):
                print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

            # Detect interactions
            print("\nFeature Interactions:")
            interactions = analyzer.detect_feature_interactions(top_n=5)
            for inter in interactions['interactions'][:3]:
                print(f"  {inter['feature_1']} <-> {inter['feature_2']}: "
                     f"{inter['correlation']:.3f} ({inter['interaction_strength']})")

            result = {
                'success': True,
                'feature_importance': importance,
                'interactions': interactions
            }

        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

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