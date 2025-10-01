#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering for ML MCP
ML MCP용 특성 공학 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML and preprocessing libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # Fallback implementations
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


class FeatureEngineer:
    """
    Comprehensive feature engineering class
    포괄적인 특성 공학 클래스
    """

    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.transformers = {}
        self.feature_names = []
        self.original_features = []

    def fit_transform(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """
        Apply feature engineering operations
        특성 공학 작업 적용
        """
        result_df = df.copy()
        self.original_features = list(df.columns)

        # Remove target column from processing if specified
        if self.target_column and self.target_column in result_df.columns:
            target_series = result_df[self.target_column]
            result_df = result_df.drop(columns=[self.target_column])
        else:
            target_series = None

        # Apply operations in order
        for operation in operations:
            if operation == 'scaling':
                result_df = self._apply_scaling(result_df)
            elif operation == 'normalization':
                result_df = self._apply_normalization(result_df)
            elif operation == 'encoding':
                result_df = self._apply_encoding(result_df)
            elif operation == 'polynomial_features':
                result_df = self._apply_polynomial_features(result_df)
            elif operation == 'feature_selection':
                if target_series is not None:
                    result_df = self._apply_feature_selection(result_df, target_series)
                else:
                    print("경고: 타겟 컬럼이 없어 특성 선택을 건너뜁니다.")
            elif operation == 'pca':
                result_df = self._apply_pca(result_df)
            elif operation == 'interaction_features':
                result_df = self._apply_interaction_features(result_df)
            else:
                print(f"경고: 알 수 없는 작업 '{operation}'을 건너뜁니다.")

        # Add target column back if it existed
        if target_series is not None:
            result_df[self.target_column] = target_series.values

        self.feature_names = list(result_df.columns)
        return result_df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return df

        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        self.transformers['scaler'] = scaler
        self.transformers['numeric_columns_scaled'] = list(numeric_cols)

        return df_scaled

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization to numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return df

        normalizer = MinMaxScaler()
        df_normalized = df.copy()
        df_normalized[numeric_cols] = normalizer.fit_transform(df[numeric_cols])

        self.transformers['normalizer'] = normalizer
        self.transformers['numeric_columns_normalized'] = list(numeric_cols)

        return df_normalized

    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to categorical features"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            return df

        df_encoded = df.copy()
        label_encoders = {}
        onehot_columns = []

        for col in categorical_cols:
            unique_count = df[col].nunique()

            # Use one-hot encoding for low cardinality, label encoding for high cardinality
            if unique_count <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                onehot_columns.extend(list(dummies.columns))
            else:
                # Label encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        self.transformers['label_encoders'] = label_encoders
        self.transformers['onehot_columns'] = onehot_columns
        self.transformers['original_categorical_columns'] = list(categorical_cols)

        return df_encoded

    def _apply_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:  # Need at least 2 numeric columns
            return df

        # Limit to first 5 columns to avoid explosion
        selected_cols = list(numeric_cols[:5])

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[selected_cols])

        # Create column names
        poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]

        # Create new dataframe with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

        # Combine with non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        result_df = pd.concat([poly_df, df[non_numeric_cols]], axis=1)

        self.transformers['polynomial'] = poly
        self.transformers['polynomial_original_columns'] = selected_cols

        return result_df

    def _apply_feature_selection(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Apply feature selection"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return df

        # Determine if classification or regression
        is_classification = target.dtype == 'object' or target.nunique() < 10

        if is_classification:
            # Encode target if necessary
            if target.dtype == 'object':
                le = LabelEncoder()
                target_encoded = le.fit_transform(target)
            else:
                target_encoded = target
            score_func = f_classif
        else:
            target_encoded = target
            score_func = f_regression

        # Select top 50% of features or max 20 features
        k = min(max(len(numeric_cols) // 2, 1), 20)

        selector = SelectKBest(score_func=score_func, k=k)
        selected_features = selector.fit_transform(df[numeric_cols], target_encoded)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_feature_names = numeric_cols[selected_mask]

        # Create result dataframe
        selected_df = pd.DataFrame(selected_features, columns=selected_feature_names, index=df.index)

        # Add non-numeric columns back
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        result_df = pd.concat([selected_df, df[non_numeric_cols]], axis=1)

        self.transformers['feature_selector'] = selector
        self.transformers['selected_features'] = list(selected_feature_names)
        self.transformers['feature_scores'] = dict(zip(numeric_cols, selector.scores_))

        return result_df

    def _apply_pca(self, df: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return df

        # Apply PCA only to numeric features
        pca = PCA(n_components=n_components, random_state=42)
        pca_features = pca.fit_transform(df[numeric_cols])

        # Create PCA feature names
        n_components_actual = pca_features.shape[1]
        pca_feature_names = [f"pca_component_{i+1}" for i in range(n_components_actual)]

        # Create result dataframe
        pca_df = pd.DataFrame(pca_features, columns=pca_feature_names, index=df.index)

        # Add non-numeric columns back
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        result_df = pd.concat([pca_df, df[non_numeric_cols]], axis=1)

        self.transformers['pca'] = pca
        self.transformers['pca_explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
        self.transformers['pca_original_columns'] = list(numeric_cols)

        return result_df

    def _apply_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return df

        df_with_interactions = df.copy()

        # Limit to first 5 columns to avoid explosion
        selected_cols = list(numeric_cols[:5])

        # Create pairwise interactions
        for i, col1 in enumerate(selected_cols):
            for col2 in selected_cols[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                df_with_interactions[interaction_name] = df[col1] * df[col2]

        self.transformers['interaction_original_columns'] = selected_cols

        return df_with_interactions

    def get_feature_report(self) -> Dict[str, Any]:
        """Generate a comprehensive feature engineering report"""
        report = {
            'original_feature_count': len(self.original_features),
            'final_feature_count': len(self.feature_names),
            'feature_increase': len(self.feature_names) - len(self.original_features),
            'transformations_applied': list(self.transformers.keys()),
            'feature_names': self.feature_names,
            'original_features': self.original_features
        }

        # Add specific transformation details
        if 'pca' in self.transformers:
            explained_variance = self.transformers['pca_explained_variance_ratio']
            report['pca_info'] = {
                'n_components': len(explained_variance),
                'total_explained_variance': sum(explained_variance),
                'explained_variance_per_component': explained_variance[:5]  # Top 5
            }

        if 'feature_selector' in self.transformers:
            report['feature_selection_info'] = {
                'selected_features': self.transformers['selected_features'],
                'top_feature_scores': dict(sorted(
                    self.transformers['feature_scores'].items(),
                    key=lambda x: x[1], reverse=True)[:10])
            }

        return report


def perform_feature_engineering(data_file: str, operations: List[str],
                               target_column: Optional[str] = None,
                               output_file: str = 'processed_data.csv') -> Dict[str, Any]:
    """
    Perform comprehensive feature engineering
    포괄적인 특성 공학 수행
    """
    # Load data
    df = load_data(data_file)
    original_info = get_data_info(df)

    # Initialize feature engineer
    engineer = FeatureEngineer(target_column=target_column)

    # Apply transformations
    processed_df = engineer.fit_transform(df, operations)

    # Save processed data
    processed_df.to_csv(output_file, index=False)

    # Get feature report
    feature_report = engineer.get_feature_report()

    # Save transformers for future use
    transformer_file = f"feature_transformers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump({
        'transformers': engineer.transformers,
        'original_features': engineer.original_features,
        'final_features': engineer.feature_names,
        'target_column': target_column,
        'operations': operations
    }, transformer_file)

    # Generate processed data info
    processed_info = get_data_info(processed_df)

    # Calculate feature engineering metrics
    metrics = {
        'dimensionality_change': {
            'original_features': len(engineer.original_features),
            'processed_features': len(engineer.feature_names),
            'change_ratio': len(engineer.feature_names) / len(engineer.original_features),
            'net_change': len(engineer.feature_names) - len(engineer.original_features)
        },
        'data_shape': {
            'original_shape': original_info['shape'],
            'processed_shape': processed_info['shape']
        },
        'feature_types': {
            'original_numeric': len(original_info['numeric_columns']),
            'original_categorical': len(original_info['categorical_columns']),
            'processed_numeric': len(processed_info['numeric_columns']),
            'processed_categorical': len(processed_info['categorical_columns'])
        }
    }

    # Prepare results
    results = {
        'operations_applied': operations,
        'target_column': target_column,
        'original_features': len(engineer.original_features),
        'processed_features': len(engineer.feature_names),
        'data_shape': processed_info['shape'],
        'output_file': output_file,
        'transformer_file': transformer_file,
        'feature_report': feature_report,
        'metrics': metrics,
        'success': True
    }

    return results


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['data_file'])

        # Extract parameters
        data_file = options['data_file']
        operations = options.get('operations', ['scaling', 'encoding'])
        target_column = options.get('target_column')
        output_file = options.get('output_file', 'processed_data.csv')

        # Perform feature engineering
        results = perform_feature_engineering(
            data_file=data_file,
            operations=operations,
            target_column=target_column,
            output_file=output_file
        )

        # Get data info for final result
        df = load_data(data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="feature_engineering",
            data_info=data_info,
            results=results,
            summary=f"특성 공학 완료 - {results['original_features']}개 → {results['processed_features']}개 특성"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "feature_engineering"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()