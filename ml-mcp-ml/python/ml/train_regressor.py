#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Training for ML MCP
ML MCP용 회귀 모델 훈련 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

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


def get_regressor_model(model_type: str, random_state: int = 42) -> Any:
    """
    Get regressor model based on type
    모델 타입에 따른 회귀기 반환
    """
    models = {
        'linear_regression': LinearRegression(n_jobs=-1),
        'ridge': Ridge(random_state=random_state),
        'lasso': Lasso(random_state=random_state),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'svr': SVR(),
        'gradient_boosting': GradientBoostingRegressor(
            random_state=random_state
        ),
        'neural_network': MLPRegressor(
            random_state=random_state,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1
        )
    }

    if model_type not in models:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    return models[model_type]


def preprocess_data_for_regression(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Preprocess data for regression
    회귀를 위한 데이터 전처리
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values in target
    if y.isnull().any():
        print(f"경고: 타겟 컬럼에 {y.isnull().sum()}개의 결측치가 있어 해당 행을 제거합니다.")
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(y):
        print("경고: 타겟 변수가 수치형이 아닙니다. 수치형으로 변환을 시도합니다.")
        y = pd.to_numeric(y, errors='coerce')

        # Remove rows where conversion failed
        valid_y = y.notna()
        if not valid_y.all():
            print(f"경고: {(~valid_y).sum()}개 행에서 타겟 변수를 수치형으로 변환할 수 없어 제거합니다.")
            X = X[valid_y]
            y = y[valid_y]

    # Encode categorical features
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        # Handle missing values
        X[col] = X[col].fillna('missing')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Handle missing values in numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        imputer = SimpleImputer(strategy='median')
        X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

    return X, y, label_encoders


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    포괄적인 회귀 지표 계산
    """
    metrics = {}

    # Basic metrics
    metrics['r2_score'] = r2_score(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])

    # Additional metrics
    metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

    # Explained variance
    metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)

    # Residual analysis
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)

    return metrics


def train_regressor_model(data_file: str, target_column: str, model_type: str = 'random_forest',
                         test_size: float = 0.2, cross_validation: bool = True,
                         save_model: bool = True, random_state: int = 42) -> Dict[str, Any]:
    """
    Train a regression model
    회귀 모델 훈련
    """
    # Load data
    df = load_data(data_file)
    data_info = get_data_info(df)

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다")

    # Preprocess data
    X, y, label_encoders = preprocess_data_for_regression(df, target_column)

    # Check if we have enough samples
    if len(X) < 10:
        raise ValueError("훈련에 충분한 데이터가 없습니다 (최소 10개 샘플 필요)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features for certain models
    scaler = None
    if model_type in ['svr', 'neural_network', 'ridge', 'lasso']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Get and train model
    model = get_regressor_model(model_type, random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred)

    # Cross validation
    cv_scores = None
    if cross_validation and len(X) >= 30:  # Enough data for CV
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, [float(x) for x in model.feature_importances_]))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(model, 'coef_'):
        # For linear models, use coefficient magnitude
        coef_importance = np.abs(model.coef_)
        feature_importance = dict(zip(X.columns, [float(x) for x in coef_importance]))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Prediction analysis
    prediction_analysis = {
        'min_prediction': float(np.min(y_pred)),
        'max_prediction': float(np.max(y_pred)),
        'mean_prediction': float(np.mean(y_pred)),
        'std_prediction': float(np.std(y_pred)),
        'min_actual': float(np.min(y_test)),
        'max_actual': float(np.max(y_test)),
        'mean_actual': float(np.mean(y_test)),
        'std_actual': float(np.std(y_test))
    }

    # Save model if requested
    model_file = None
    if save_model:
        model_file = f"regressor_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_info = {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_columns': list(X.columns),
            'model_type': model_type,
            'target_column': target_column,
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
        joblib.dump(model_info, model_file)

    # Prepare results
    results = {
        'model_type': model_type,
        'target_column': target_column,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],

        # Performance metrics
        'r2_score': round(metrics['r2_score'], 4),
        'mae': round(metrics['mae'], 4),
        'mse': round(metrics['mse'], 4),
        'rmse': round(metrics['rmse'], 4),
        'mape': round(metrics['mean_absolute_percentage_error'], 2),
        'explained_variance': round(metrics['explained_variance'], 4),

        # Residual analysis
        'residual_analysis': {
            'mean': round(metrics['residual_mean'], 4),
            'std': round(metrics['residual_std'], 4),
            'is_unbiased': abs(metrics['residual_mean']) < 0.1 * metrics['residual_std']
        },

        # Additional info
        'feature_importance': feature_importance,
        'prediction_analysis': prediction_analysis,
        'model_file': model_file,
        'preprocessing': {
            'scaled': scaler is not None,
            'categorical_encoded': len(label_encoders) > 0
        }
    }

    # Add cross validation scores
    if cv_scores is not None:
        results['cv_scores'] = [round(float(score), 4) for score in cv_scores]
        results['cv_mean'] = round(float(cv_scores.mean()), 4)
        results['cv_std'] = round(float(cv_scores.std()), 4)

    # Performance interpretation
    r2 = metrics['r2_score']
    if r2 >= 0.9:
        performance = "우수"
    elif r2 >= 0.7:
        performance = "양호"
    elif r2 >= 0.5:
        performance = "보통"
    else:
        performance = "개선 필요"

    results['performance_interpretation'] = performance

    return results


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['data_file', 'target_column'])

        # Extract parameters
        data_file = options['data_file']
        target_column = options['target_column']
        model_type = options.get('model_type', 'random_forest')
        test_size = options.get('test_size', 0.2)
        cross_validation = options.get('cross_validation', True)
        save_model = options.get('save_model', True)

        # Train regressor
        results = train_regressor_model(
            data_file=data_file,
            target_column=target_column,
            model_type=model_type,
            test_size=test_size,
            cross_validation=cross_validation,
            save_model=save_model
        )

        # Get data info for final result
        df = load_data(data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="regression_training",
            data_info=data_info,
            results=results,
            summary=f"{model_type} 회귀 모델 훈련 완료 - R² 점수: {results['r2_score']:.4f}"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "regression_training"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()