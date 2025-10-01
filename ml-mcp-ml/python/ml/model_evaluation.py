#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation for ML MCP
ML MCP용 모델 평가 스크립트
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
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, max_error
)

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

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


def load_model_and_data(model_file: str, test_data_file: str, target_column: str) -> tuple:
    """
    Load trained model and test data
    훈련된 모델과 테스트 데이터 로드
    """
    # Load model
    model_info = joblib.load(model_file)
    model = model_info['model']
    scaler = model_info.get('scaler')
    label_encoders = model_info.get('label_encoders', {})
    target_encoder = model_info.get('target_encoder')

    # Load test data
    df = load_data(test_data_file)

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 테스트 데이터에 없습니다")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Apply preprocessing
    X_processed = preprocess_test_data(X, label_encoders, scaler)

    # Handle target encoding if necessary
    if target_encoder is not None:
        y_processed = target_encoder.transform(y)
    else:
        y_processed = y

    return model, X_processed, y_processed, y, model_info


def preprocess_test_data(X: pd.DataFrame, label_encoders: Dict, scaler: Optional[Any]) -> pd.DataFrame:
    """
    Preprocess test data using saved transformers
    저장된 변환기를 사용하여 테스트 데이터 전처리
    """
    X_processed = X.copy()

    # Apply label encoding for categorical columns
    for col, encoder in label_encoders.items():
        if col in X_processed.columns:
            # Handle unseen categories
            X_processed[col] = X_processed[col].fillna('missing')

            # Transform only known categories
            unique_values = set(encoder.classes_)
            X_processed[col] = X_processed[col].apply(
                lambda x: x if x in unique_values else 'missing'
            )

            # If 'missing' is not in classes, use the first class as default
            if 'missing' not in unique_values:
                first_class = encoder.classes_[0]
                X_processed[col] = X_processed[col].apply(
                    lambda x: x if x in unique_values else first_class
                )

            X_processed[col] = encoder.transform(X_processed[col])

    # Apply scaling if it was used during training
    if scaler is not None:
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])

    return X_processed


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray,
                                y_original: pd.Series, generate_plots: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation for classification models
    분류 모델의 포괄적 평가
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Get prediction probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Handle multi-class vs binary classification
    unique_classes = len(np.unique(y_test))
    average_method = 'binary' if unique_classes == 2 else 'weighted'

    precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC AUC for binary classification
    roc_auc = None
    if unique_classes == 2 and y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            pass

    # Class-wise performance
    class_performance = {}
    for class_label in np.unique(y_test):
        mask = y_test == class_label
        if np.sum(mask) > 0:
            class_performance[str(class_label)] = {
                'precision': float(class_report[str(class_label)]['precision']),
                'recall': float(class_report[str(class_label)]['recall']),
                'f1_score': float(class_report[str(class_label)]['f1-score']),
                'support': int(class_report[str(class_label)]['support'])
            }

    # Prediction confidence analysis
    confidence_analysis = {}
    if y_pred_proba is not None:
        max_probas = np.max(y_pred_proba, axis=1)
        confidence_analysis = {
            'mean_confidence': float(np.mean(max_probas)),
            'min_confidence': float(np.min(max_probas)),
            'max_confidence': float(np.max(max_probas)),
            'std_confidence': float(np.std(max_probas)),
            'low_confidence_count': int(np.sum(max_probas < 0.6)),  # Predictions with < 60% confidence
            'high_confidence_count': int(np.sum(max_probas > 0.8))   # Predictions with > 80% confidence
        }

    # Generate plots if requested
    plot_files = []
    if generate_plots:
        plot_files = generate_classification_plots(
            y_test, y_pred, y_pred_proba, conf_matrix, unique_classes
        )

    results = {
        'task_type': 'classification',
        'n_classes': unique_classes,
        'test_samples': len(y_test),

        # Performance metrics
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4) if roc_auc is not None else None,

        # Detailed analysis
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'class_performance': class_performance,
        'confidence_analysis': confidence_analysis,
        'plot_files': plot_files
    }

    return results


def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray,
                             y_original: pd.Series, generate_plots: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation for regression models
    회귀 모델의 포괄적 평가
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Basic metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_var = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    # Additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100

    # Residual analysis
    residuals = y_test - y_pred
    residual_stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75))
    }

    # Performance categorization
    if r2 >= 0.9:
        performance_category = "우수"
    elif r2 >= 0.7:
        performance_category = "양호"
    elif r2 >= 0.5:
        performance_category = "보통"
    else:
        performance_category = "개선 필요"

    # Prediction vs actual analysis
    prediction_analysis = {
        'actual_mean': float(np.mean(y_test)),
        'actual_std': float(np.std(y_test)),
        'predicted_mean': float(np.mean(y_pred)),
        'predicted_std': float(np.std(y_pred)),
        'correlation': float(np.corrcoef(y_test, y_pred)[0, 1])
    }

    # Error distribution analysis
    error_percentiles = {
        'p10': float(np.percentile(np.abs(residuals), 10)),
        'p25': float(np.percentile(np.abs(residuals), 25)),
        'p50': float(np.percentile(np.abs(residuals), 50)),
        'p75': float(np.percentile(np.abs(residuals), 75)),
        'p90': float(np.percentile(np.abs(residuals), 90))
    }

    # Generate plots if requested
    plot_files = []
    if generate_plots:
        plot_files = generate_regression_plots(y_test, y_pred, residuals)

    results = {
        'task_type': 'regression',
        'test_samples': len(y_test),

        # Performance metrics
        'r2_score': round(float(r2), 4),
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'mape': round(float(mape), 2),
        'explained_variance': round(float(explained_var), 4),
        'max_error': round(float(max_err), 4),

        # Analysis
        'performance_category': performance_category,
        'residual_analysis': residual_stats,
        'prediction_analysis': prediction_analysis,
        'error_distribution': error_percentiles,
        'plot_files': plot_files
    }

    return results


def generate_classification_plots(y_test: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray],
                                conf_matrix: np.ndarray, n_classes: int) -> List[str]:
    """Generate classification evaluation plots"""
    plot_files = []

    try:
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plot_file = 'confusion_matrix.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

        # ROC Curve for binary classification
        if n_classes == 2 and y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plot_file = 'roc_curve.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)

        # Prediction confidence histogram
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            max_probas = np.max(y_pred_proba, axis=1)
            plt.hist(max_probas, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Confidence')
            plt.grid(True, alpha=0.3)
            plot_file = 'confidence_distribution.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)

    except Exception as e:
        print(f"경고: 플롯 생성 중 오류 발생: {e}")

    return plot_files


def generate_regression_plots(y_test: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> List[str]:
    """Generate regression evaluation plots"""
    plot_files = []

    try:
        # Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        plot_file = 'actual_vs_predicted.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

        # Residuals plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.grid(True, alpha=0.3)
        plot_file = 'residuals_plot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

        # Residuals histogram
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        plot_file = 'residuals_histogram.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

    except Exception as e:
        print(f"경고: 플롯 생성 중 오류 발생: {e}")

    return plot_files


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['model_file', 'test_data_file', 'target_column', 'task_type'])

        # Extract parameters
        model_file = options['model_file']
        test_data_file = options['test_data_file']
        target_column = options['target_column']
        task_type = options['task_type']
        generate_plots = options.get('generate_plots', True)

        # Load model and data
        model, X_test, y_test, y_original, model_info = load_model_and_data(
            model_file, test_data_file, target_column
        )

        # Perform evaluation based on task type
        if task_type == 'classification':
            evaluation_results = evaluate_classification_model(
                model, X_test, y_test, y_original, generate_plots
            )
        elif task_type == 'regression':
            evaluation_results = evaluate_regression_model(
                model, X_test, y_test, y_original, generate_plots
            )
        else:
            raise ValueError(f"지원하지 않는 작업 유형: {task_type}")

        # Get data info for final result
        df = load_data(test_data_file)
        data_info = get_data_info(df)

        # Add model information to results
        evaluation_results.update({
            'model_file': model_file,
            'model_type': model_info.get('model_type', 'unknown'),
            'target_column': target_column
        })

        # Create final result
        final_result = create_analysis_result(
            analysis_type="model_evaluation",
            data_info=data_info,
            results=evaluation_results,
            summary=f"{task_type} 모델 평가 완료 - 테스트 샘플: {len(y_test)}개"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "model_evaluation"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()