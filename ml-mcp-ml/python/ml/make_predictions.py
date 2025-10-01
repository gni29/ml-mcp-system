#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Predictions with Trained Models for ML MCP
ML MCP용 훈련된 모델로 예측 수행 스크립트
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


def load_trained_model(model_file: str) -> Dict[str, Any]:
    """
    Load trained model and its preprocessing components
    훈련된 모델과 전처리 구성 요소 로드
    """
    try:
        model_info = joblib.load(model_file)

        required_keys = ['model']
        missing_keys = [key for key in required_keys if key not in model_info]
        if missing_keys:
            raise ValueError(f"모델 파일에 필수 키가 없습니다: {missing_keys}")

        return model_info

    except Exception as e:
        raise ValueError(f"모델 파일 로드 실패: {str(e)}")


def preprocess_prediction_data(df: pd.DataFrame, model_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess input data for prediction using saved transformers
    저장된 변환기를 사용하여 예측용 입력 데이터 전처리
    """
    df_processed = df.copy()

    # Get preprocessing components
    scaler = model_info.get('scaler')
    label_encoders = model_info.get('label_encoders', {})
    feature_columns = model_info.get('feature_columns')

    # Ensure we have the expected columns
    if feature_columns is not None:
        missing_columns = set(feature_columns) - set(df_processed.columns)
        if missing_columns:
            print(f"경고: 다음 컬럼들이 누락되어 기본값으로 채웁니다: {missing_columns}")
            for col in missing_columns:
                df_processed[col] = 0  # Default value for missing columns

        # Select only the required columns in the correct order
        df_processed = df_processed[feature_columns]

    # Apply label encoding for categorical columns
    for col, encoder in label_encoders.items():
        if col in df_processed.columns:
            # Handle missing values
            df_processed[col] = df_processed[col].fillna('missing')

            # Handle unseen categories
            known_categories = set(encoder.classes_)
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in known_categories else encoder.classes_[0]
            )

            # Transform
            df_processed[col] = encoder.transform(df_processed[col])

    # Apply scaling if it was used during training
    if scaler is not None:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

    return df_processed


def make_predictions_with_analysis(model_info: Dict[str, Any], X_processed: pd.DataFrame,
                                  include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Make predictions and provide comprehensive analysis
    예측 수행 및 포괄적 분석 제공
    """
    model = model_info['model']
    model_type = model_info.get('model_type', 'unknown')
    task_type = model_info.get('task_type', 'unknown')
    target_encoder = model_info.get('target_encoder')

    # Make predictions
    predictions = model.predict(X_processed)

    # Get prediction probabilities for classification models
    prediction_probabilities = None
    if include_probabilities and hasattr(model, 'predict_proba'):
        prediction_probabilities = model.predict_proba(X_processed)

    # Decode predictions if target was encoded
    if target_encoder is not None:
        predictions_decoded = target_encoder.inverse_transform(predictions)
    else:
        predictions_decoded = predictions

    # Prediction analysis
    prediction_stats = {
        'total_predictions': len(predictions),
        'unique_predictions': len(np.unique(predictions)),
        'prediction_distribution': dict(zip(*np.unique(predictions_decoded, return_counts=True)))
    }

    # Additional statistics based on task type
    if task_type == 'regression' or (task_type == 'unknown' and np.issubdtype(predictions.dtype, np.number)):
        # Regression-specific statistics
        prediction_stats.update({
            'min_prediction': float(np.min(predictions)),
            'max_prediction': float(np.max(predictions)),
            'mean_prediction': float(np.mean(predictions)),
            'median_prediction': float(np.median(predictions)),
            'std_prediction': float(np.std(predictions)),
            'quartiles': {
                'q1': float(np.percentile(predictions, 25)),
                'q2': float(np.percentile(predictions, 50)),
                'q3': float(np.percentile(predictions, 75))
            }
        })

    # Confidence analysis for classification with probabilities
    confidence_analysis = None
    if prediction_probabilities is not None:
        max_probabilities = np.max(prediction_probabilities, axis=1)
        confidence_analysis = {
            'mean_confidence': float(np.mean(max_probabilities)),
            'min_confidence': float(np.min(max_probabilities)),
            'max_confidence': float(np.max(max_probabilities)),
            'std_confidence': float(np.std(max_probabilities)),
            'confidence_distribution': {
                'very_high': int(np.sum(max_probabilities > 0.9)),  # >90%
                'high': int(np.sum((max_probabilities > 0.7) & (max_probabilities <= 0.9))),  # 70-90%
                'medium': int(np.sum((max_probabilities > 0.5) & (max_probabilities <= 0.7))),  # 50-70%
                'low': int(np.sum(max_probabilities <= 0.5))  # <=50%
            }
        }

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_columns = model_info.get('feature_columns', [])
        if feature_columns and len(feature_columns) == len(model.feature_importances_):
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True))

    results = {
        'predictions': predictions_decoded.tolist() if hasattr(predictions_decoded, 'tolist') else list(predictions_decoded),
        'raw_predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
        'prediction_statistics': prediction_stats,
        'model_info': {
            'model_type': model_type,
            'task_type': task_type,
            'feature_count': X_processed.shape[1]
        }
    }

    # Add probabilities if requested and available
    if prediction_probabilities is not None:
        results['prediction_probabilities'] = prediction_probabilities.tolist()
        results['confidence_analysis'] = confidence_analysis

    # Add feature importance if available
    if feature_importance is not None:
        results['feature_importance'] = feature_importance

    return results


def save_predictions(predictions_data: Dict[str, Any], original_df: pd.DataFrame,
                    output_file: str, include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Save predictions to file with original data
    원본 데이터와 함께 예측 결과를 파일로 저장
    """
    # Create output dataframe
    output_df = original_df.copy()

    # Add predictions
    output_df['predictions'] = predictions_data['predictions']

    # Add probabilities if available
    if include_probabilities and 'prediction_probabilities' in predictions_data:
        probabilities = predictions_data['prediction_probabilities']
        for i, prob_array in enumerate(probabilities):
            if isinstance(prob_array, (list, np.ndarray)):
                for j, prob in enumerate(prob_array):
                    output_df[f'probability_class_{j}'] = probabilities[:, j] if isinstance(probabilities, np.ndarray) else [p[j] for p in probabilities]

    # Add confidence scores for classification
    if 'confidence_analysis' in predictions_data:
        probabilities = predictions_data['prediction_probabilities']
        max_probabilities = [max(prob) for prob in probabilities]
        output_df['prediction_confidence'] = max_probabilities

    # Save to file
    output_df.to_csv(output_file, index=False)

    return {
        'output_file': output_file,
        'total_rows': len(output_df),
        'columns_added': ['predictions'] +
                        ([f'probability_class_{i}' for i in range(len(predictions_data.get('prediction_probabilities', [[]])[0]))] if 'prediction_probabilities' in predictions_data else []) +
                        (['prediction_confidence'] if 'confidence_analysis' in predictions_data else [])
    }


def perform_prediction_task(model_file: str, input_data_file: str, output_file: str = 'predictions.csv',
                           include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Complete prediction workflow
    완전한 예측 워크플로우
    """
    # Load trained model
    model_info = load_trained_model(model_file)

    # Load input data
    input_df = load_data(input_data_file)

    # Preprocess input data
    X_processed = preprocess_prediction_data(input_df, model_info)

    # Make predictions with analysis
    prediction_results = make_predictions_with_analysis(
        model_info, X_processed, include_probabilities
    )

    # Save predictions
    save_info = save_predictions(
        prediction_results, input_df, output_file, include_probabilities
    )

    # Combine results
    final_results = {
        'model_file': model_file,
        'input_data_file': input_data_file,
        'output_file': output_file,
        'include_probabilities': include_probabilities,
        'predictions_count': prediction_results['prediction_statistics']['total_predictions'],
        'unique_predictions': prediction_results['prediction_statistics']['unique_predictions'],
        'prediction_results': prediction_results,
        'save_info': save_info,
        'success': True
    }

    return final_results


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['model_file', 'input_data_file'])

        # Extract parameters
        model_file = options['model_file']
        input_data_file = options['input_data_file']
        output_file = options.get('output_file', 'predictions.csv')
        include_probabilities = options.get('include_probabilities', False)

        # Perform prediction task
        results = perform_prediction_task(
            model_file=model_file,
            input_data_file=input_data_file,
            output_file=output_file,
            include_probabilities=include_probabilities
        )

        # Get data info for final result
        df = load_data(input_data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="make_predictions",
            data_info=data_info,
            results=results,
            summary=f"예측 완료 - {results['predictions_count']}개 샘플 예측됨"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "make_predictions"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()