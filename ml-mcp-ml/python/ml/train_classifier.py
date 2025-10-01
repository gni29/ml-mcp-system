#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Model Training for ML MCP
ML MCP용 분류 모델 훈련 스크립트
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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


def get_classifier_model(model_type: str, random_state: int = 42) -> Any:
    """
    Get classifier model based on type
    모델 타입에 따른 분류기 반환
    """
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'svm': SVC(
            random_state=random_state,
            probability=True  # For probability predictions
        ),
        'logistic_regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=random_state
        ),
        'neural_network': MLPClassifier(
            random_state=random_state,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1
        )
    }

    if model_type not in models:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    return models[model_type]


def preprocess_data(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Preprocess data for classification
    분류를 위한 데이터 전처리
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

    # Encode target if it's categorical
    target_encoder = None
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    return X, y, label_encoders, target_encoder


def train_classifier_model(data_file: str, target_column: str, model_type: str = 'random_forest',
                          test_size: float = 0.2, cross_validation: bool = True,
                          save_model: bool = True, random_state: int = 42) -> Dict[str, Any]:
    """
    Train a classification model
    분류 모델 훈련
    """
    # Load data
    df = load_data(data_file)
    data_info = get_data_info(df)

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다")

    # Preprocess data
    X, y, label_encoders, target_encoder = preprocess_data(df, target_column)

    # Check if we have enough samples
    if len(X) < 10:
        raise ValueError("훈련에 충분한 데이터가 없습니다 (최소 10개 샘플 필요)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features for certain models
    scaler = None
    if model_type in ['svm', 'neural_network', 'logistic_regression']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Get and train model
    model = get_classifier_model(model_type, random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Cross validation
    cv_scores = None
    if cross_validation and len(X) >= 30:  # Enough data for CV
        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Save model if requested
    model_file = None
    if save_model:
        model_file = f"classifier_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_info = {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'feature_columns': list(X.columns),
            'model_type': model_type,
            'target_column': target_column
        }
        joblib.dump(model_info, model_file)

    # Prepare results
    results = {
        'model_type': model_type,
        'target_column': target_column,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),

        # Performance metrics
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),

        # Additional info
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'model_file': model_file,
        'preprocessing': {
            'scaled': scaler is not None,
            'categorical_encoded': len(label_encoders) > 0,
            'target_encoded': target_encoder is not None
        }
    }

    # Add cross validation scores
    if cv_scores is not None:
        results['cv_scores'] = [round(float(score), 4) for score in cv_scores]
        results['cv_mean'] = round(float(cv_scores.mean()), 4)
        results['cv_std'] = round(float(cv_scores.std()), 4)

    # Add prediction probabilities info
    if y_pred_proba is not None:
        results['probability_available'] = True
        results['prediction_confidence'] = {
            'mean_max_proba': round(float(np.mean(np.max(y_pred_proba, axis=1))), 4),
            'min_max_proba': round(float(np.min(np.max(y_pred_proba, axis=1))), 4)
        }

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

        # Train classifier
        results = train_classifier_model(
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
            analysis_type="classification_training",
            data_info=data_info,
            results=results,
            summary=f"{model_type} 분류 모델 훈련 완료 - 정확도: {results['accuracy']:.4f}"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "classification_training"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()