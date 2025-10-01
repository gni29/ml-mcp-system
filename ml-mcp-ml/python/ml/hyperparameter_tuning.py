#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning for ML MCP
ML MCP용 하이퍼파라미터 튜닝 스크립트
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score

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


def get_hyperparameter_grids(model_type: str, task_type: str) -> Dict[str, Any]:
    """
    Get hyperparameter grids for different models
    모델별 하이퍼파라미터 그리드 반환
    """
    grids = {}

    if model_type == 'random_forest':
        if task_type == 'classification':
            grids['random_forest'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:  # regression
            grids['random_forest'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

    elif model_type == 'svm':
        if task_type == 'classification':
            grids['svm'] = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        else:  # regression
            grids['svm'] = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly'],
                'epsilon': [0.01, 0.1, 0.2]
            }

    elif model_type == 'gradient_boosting':
        if task_type == 'classification':
            grids['gradient_boosting'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:  # regression
            grids['gradient_boosting'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

    elif model_type == 'neural_network':
        grids['neural_network'] = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000]
        }

    return grids.get(model_type, {})


def get_model_instance(model_type: str, task_type: str, random_state: int = 42) -> Any:
    """
    Get model instance for hyperparameter tuning
    하이퍼파라미터 튜닝을 위한 모델 인스턴스 반환
    """
    if task_type == 'classification':
        models = {
            'random_forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
            'svm': SVC(random_state=random_state, probability=True),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'neural_network': MLPClassifier(random_state=random_state, early_stopping=True)
        }
    else:  # regression
        models = {
            'random_forest': RandomForestRegressor(random_state=random_state, n_jobs=-1),
            'svm': SVR(),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
            'neural_network': MLPRegressor(random_state=random_state, early_stopping=True)
        }

    if model_type not in models:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    return models[model_type]


def preprocess_data_for_tuning(df: pd.DataFrame, target_column: str, task_type: str) -> tuple:
    """
    Preprocess data for hyperparameter tuning
    하이퍼파라미터 튜닝을 위한 데이터 전처리
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

    # For regression, ensure target is numeric
    if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
        print("경고: 회귀 작업을 위해 타겟 변수를 수치형으로 변환합니다.")
        y = pd.to_numeric(y, errors='coerce')
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
        X[col] = X[col].fillna('missing')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Handle missing values in numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        imputer = SimpleImputer(strategy='median')
        X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

    # Encode target for classification if needed
    target_encoder = None
    if task_type == 'classification' and (y.dtype == 'object' or pd.api.types.is_categorical_dtype(y)):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    return X, y, label_encoders, target_encoder


def perform_hyperparameter_tuning(data_file: str, target_column: str, model_type: str = 'random_forest',
                                  task_type: str = 'classification', search_method: str = 'grid_search',
                                  cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning
    하이퍼파라미터 튜닝 수행
    """
    # Load data
    df = load_data(data_file)
    data_info = get_data_info(df)

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다")

    # Preprocess data
    X, y, label_encoders, target_encoder = preprocess_data_for_tuning(df, target_column, task_type)

    # Check if we have enough samples
    if len(X) < 30:
        raise ValueError("하이퍼파라미터 튜닝에 충분한 데이터가 없습니다 (최소 30개 샘플 필요)")

    # Get model and parameter grid
    model = get_model_instance(model_type, task_type, random_state)
    param_grid = get_hyperparameter_grids(model_type, task_type)

    if not param_grid:
        raise ValueError(f"모델 타입 '{model_type}'에 대한 하이퍼파라미터 그리드가 정의되지 않았습니다")

    # Scale features for certain models
    scaler = None
    if model_type in ['svm', 'neural_network']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Set up cross-validation
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'r2'

    # Perform hyperparameter search
    if search_method == 'grid_search':
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=0, return_train_score=True
        )
    elif search_method == 'random_search':
        search = RandomizedSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_iter=20, n_jobs=-1, verbose=0, random_state=random_state,
            return_train_score=True
        )
    else:
        raise ValueError(f"지원하지 않는 탐색 방법: {search_method}")

    # Fit the search
    search.fit(X, y)

    # Get results
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    # Cross-validation results analysis
    cv_results = search.cv_results_
    mean_test_scores = cv_results['mean_test_score']
    std_test_scores = cv_results['std_test_score']
    mean_train_scores = cv_results['mean_train_score']

    # Parameter importance analysis
    param_importance = analyze_parameter_importance(cv_results, param_grid)

    # Model performance comparison
    performance_comparison = {
        'best_score': float(best_score),
        'worst_score': float(np.min(mean_test_scores)),
        'mean_score': float(np.mean(mean_test_scores)),
        'std_score': float(np.std(mean_test_scores)),
        'score_range': float(np.max(mean_test_scores) - np.min(mean_test_scores))
    }

    # Overfitting analysis
    train_test_gap = np.mean(mean_train_scores) - best_score
    overfitting_risk = "높음" if train_test_gap > 0.1 else "보통" if train_test_gap > 0.05 else "낮음"

    # Save best model if requested
    model_file = f"tuned_{model_type}_{task_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    model_info = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_columns': list(X.columns) if hasattr(X, 'columns') else None,
        'model_type': model_type,
        'task_type': task_type,
        'target_column': target_column,
        'best_params': best_params,
        'tuning_info': {
            'search_method': search_method,
            'cv_folds': cv_folds,
            'total_combinations': len(mean_test_scores)
        }
    }
    joblib.dump(model_info, model_file)

    # Prepare results
    results = {
        'model_type': model_type,
        'task_type': task_type,
        'target_column': target_column,
        'search_method': search_method,
        'cv_folds': cv_folds,
        'total_combinations_tested': len(mean_test_scores),

        # Best results
        'best_params': best_params,
        'best_score': round(float(best_score), 4),
        'best_score_std': round(float(std_test_scores[np.argmax(mean_test_scores)]), 4),

        # Performance analysis
        'performance_comparison': performance_comparison,
        'param_importance': param_importance,
        'overfitting_analysis': {
            'train_test_gap': round(float(train_test_gap), 4),
            'risk_level': overfitting_risk
        },

        # Model info
        'model_file': model_file,
        'preprocessing': {
            'scaled': scaler is not None,
            'categorical_encoded': len(label_encoders) > 0,
            'target_encoded': target_encoder is not None
        }
    }

    # Add feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_') and hasattr(X, 'columns'):
        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        results['feature_importance'] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return results


def analyze_parameter_importance(cv_results: Dict, param_grid: Dict) -> Dict[str, Any]:
    """
    Analyze parameter importance based on CV results
    CV 결과를 바탕으로 파라미터 중요도 분석
    """
    param_importance = {}

    for param_name in param_grid.keys():
        param_key = f'param_{param_name}'
        if param_key in cv_results:
            param_values = cv_results[param_key]
            test_scores = cv_results['mean_test_score']

            # Group scores by parameter value
            unique_values = set(param_values)
            value_scores = {}

            for value in unique_values:
                mask = param_values == value
                scores = test_scores[mask]
                value_scores[str(value)] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'count': int(np.sum(mask))
                }

            # Calculate importance as range of mean scores
            mean_scores = [info['mean_score'] for info in value_scores.values()]
            importance = np.max(mean_scores) - np.min(mean_scores)

            param_importance[param_name] = {
                'importance_score': round(float(importance), 4),
                'value_performance': value_scores,
                'best_value': max(value_scores.keys(), key=lambda k: value_scores[k]['mean_score'])
            }

    # Sort by importance
    param_importance = dict(sorted(param_importance.items(),
                                  key=lambda x: x[1]['importance_score'], reverse=True))

    return param_importance


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['data_file', 'target_column', 'task_type'])

        # Extract parameters
        data_file = options['data_file']
        target_column = options['target_column']
        model_type = options.get('model_type', 'random_forest')
        task_type = options['task_type']
        search_method = options.get('search_method', 'grid_search')
        cv_folds = options.get('cv_folds', 5)

        # Perform hyperparameter tuning
        results = perform_hyperparameter_tuning(
            data_file=data_file,
            target_column=target_column,
            model_type=model_type,
            task_type=task_type,
            search_method=search_method,
            cv_folds=cv_folds
        )

        # Get data info for final result
        df = load_data(data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="hyperparameter_tuning",
            data_info=data_info,
            results=results,
            summary=f"{model_type} {task_type} 모델 하이퍼파라미터 튜닝 완료 - 최고 점수: {results['best_score']:.4f}"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "hyperparameter_tuning"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()