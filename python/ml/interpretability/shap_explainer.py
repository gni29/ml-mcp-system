#!/usr/bin/env python3
"""
SHAP Explainer for ML MCP System
SHapley Additive exPlanations for model interpretability
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SHAPExplainer:
    """SHAP-based model explainer"""

    def __init__(self, model, X_train, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer

        Args:
            model: Trained ML model
            X_train: Training data for background
            feature_names: Feature names (optional)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required. Install with: pip install shap")

        self.model = model
        self.X_train = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)

        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        self.X_train.columns = self.feature_names

        # Initialize explainer
        self.explainer = self._create_explainer()
        self.shap_values = None

    def _create_explainer(self):
        """Create appropriate SHAP explainer for model type"""
        try:
            # Try tree explainer first (fastest)
            if hasattr(self.model, 'tree_'):
                return shap.TreeExplainer(self.model)

            # For ensemble tree models
            if hasattr(self.model, 'estimators_'):
                return shap.TreeExplainer(self.model)

            # For linear models
            if hasattr(self.model, 'coef_'):
                return shap.LinearExplainer(self.model, self.X_train)

            # Fall back to kernel explainer (slower but works for any model)
            # Use subset for efficiency
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            return shap.KernelExplainer(self.model.predict, background)

        except Exception as e:
            # Last resort: kernel explainer with small background
            background = shap.sample(self.X_train, min(50, len(self.X_train)))
            return shap.KernelExplainer(self.model.predict, background)

    def calculate_shap_values(self, X) -> np.ndarray:
        """
        Calculate SHAP values for data

        Args:
            X: Data to explain

        Returns:
            SHAP values array
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)

        try:
            shap_values = self.explainer.shap_values(X_df)

            # Handle multi-class output
            if isinstance(shap_values, list):
                # Average across classes
                shap_values = np.mean(np.abs(shap_values), axis=0)

            self.shap_values = shap_values
            return shap_values

        except Exception as e:
            raise RuntimeError(f"SHAP calculation failed: {str(e)}")

    def get_global_importance(self, X) -> Dict[str, Any]:
        """
        Get global feature importance from SHAP values

        Args:
            X: Data to explain

        Returns:
            Global importance dictionary
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        # Sort by importance
        indices = np.argsort(mean_abs_shap)[::-1]

        results = {
            'method': 'shap_global_importance',
            'importances': {
                self.feature_names[i]: float(mean_abs_shap[i])
                for i in indices
            },
            'top_features': [
                {
                    'feature': self.feature_names[i],
                    'mean_abs_shap': float(mean_abs_shap[i])
                }
                for i in indices[:10]
            ],
            'summary': {
                'most_important': self.feature_names[indices[0]],
                'least_important': self.feature_names[indices[-1]],
                'mean_importance': float(np.mean(mean_abs_shap)),
                'total_samples': len(X)
            }
        }

        return results

    def explain_instance(self, X_instance, feature_values: bool = True) -> Dict[str, Any]:
        """
        Explain single prediction

        Args:
            X_instance: Single instance to explain
            feature_values: Include feature values

        Returns:
            Explanation dictionary
        """
        # Ensure 2D array
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)

        shap_values = self.calculate_shap_values(X_instance)

        # Get SHAP values for first instance
        instance_shap = shap_values[0]

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(instance_shap))[::-1]

        explanation = {
            'method': 'shap_instance_explanation',
            'feature_contributions': [
                {
                    'feature': self.feature_names[i],
                    'shap_value': float(instance_shap[i]),
                    'feature_value': float(X_instance[0, i]) if feature_values else None,
                    'impact': 'positive' if instance_shap[i] > 0 else 'negative'
                }
                for i in indices[:10]
            ],
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None,
            'summary': {
                'most_influential': self.feature_names[indices[0]],
                'total_positive_impact': float(np.sum(instance_shap[instance_shap > 0])),
                'total_negative_impact': float(np.sum(instance_shap[instance_shap < 0]))
            }
        }

        return explanation

    def summary_plot(self, X, output_path: Optional[str] = None,
                    plot_type: str = 'dot', max_display: int = 20) -> str:
        """
        Create SHAP summary plot

        Args:
            X: Data to explain
            output_path: Output file path
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum features to display

        Returns:
            Path to saved plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        plt.figure(figsize=(10, 8))

        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, X,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values, X,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )

        plt.tight_layout()

        if output_path is None:
            output_path = f'shap_summary_{plot_type}.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def force_plot(self, X_instance, output_path: Optional[str] = None) -> str:
        """
        Create SHAP force plot for single prediction

        Args:
            X_instance: Single instance to explain
            output_path: Output file path

        Returns:
            Path to saved plot (or HTML)
        """
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)

        shap_values = self.calculate_shap_values(X_instance)

        # Create force plot
        shap.force_plot(
            self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            shap_values[0],
            X_instance[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        if output_path is None:
            output_path = 'shap_force_plot.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def dependence_plot(self, feature: str, X,
                       interaction_index: Optional[str] = 'auto',
                       output_path: Optional[str] = None) -> str:
        """
        Create SHAP dependence plot

        Args:
            feature: Feature name
            X: Data to explain
            interaction_index: Feature for interaction coloring
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        feature_idx = self.feature_names.index(feature)

        plt.figure(figsize=(10, 6))

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False
        )

        plt.tight_layout()

        if output_path is None:
            output_path = f'shap_dependence_{feature}.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python shap_explainer.py <action>")
        print("Actions: demo, check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            result = {
                'shap_available': SHAP_AVAILABLE,
                'install_command': 'pip install shap' if not SHAP_AVAILABLE else None
            }

        elif action == 'demo':
            if not SHAP_AVAILABLE:
                result = {
                    'success': False,
                    'error': 'SHAP not installed',
                    'install_command': 'pip install shap'
                }
            else:
                # Demo with synthetic data
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.datasets import make_classification

                print("SHAP Explainer Demo")
                print("=" * 50)

                # Generate data
                X, y = make_classification(n_samples=1000, n_features=10,
                                          n_informative=5, random_state=42)
                feature_names = [f'feature_{i}' for i in range(10)]
                X_df = pd.DataFrame(X, columns=feature_names)

                # Train model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_df[:800], y[:800])

                # SHAP analysis
                explainer = SHAPExplainer(model, X_df[:800])

                # Global importance
                global_imp = explainer.get_global_importance(X_df[800:900])
                print("\nTop 5 Features (SHAP):")
                for i, feat in enumerate(global_imp['top_features'][:5], 1):
                    print(f"  {i}. {feat['feature']}: {feat['mean_abs_shap']:.4f}")

                # Explain single instance
                instance_exp = explainer.explain_instance(X_df.iloc[800].values)
                print("\nInstance Explanation (Top 3):")
                for i, contrib in enumerate(instance_exp['feature_contributions'][:3], 1):
                    print(f"  {i}. {contrib['feature']}: "
                         f"{contrib['shap_value']:.4f} ({contrib['impact']})")

                result = {
                    'success': True,
                    'global_importance': global_imp,
                    'instance_explanation': instance_exp
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