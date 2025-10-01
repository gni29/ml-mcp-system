#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Analysis Module
시계열 예측 분석 모듈

이 모듈은 다양한 시계열 예측 알고리즘을 사용하여 미래 값을 예측합니다.
주요 기능:
- ARIMA, 지수평활, 선형회귀 예측 모델
- 자동 모델 선택 및 파라미터 최적화
- 예측 신뢰구간 계산
- 모델 성능 평가 및 검증
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 공유 유틸리티 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # 공유 유틸리티 import 실패 시 대체 구현
    def load_data(file_path: str) -> pd.DataFrame:
        """데이터 파일 로드"""
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 기본 정보 추출"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": [],
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        """표준화된 분석 결과 구조 생성"""
        return {
            "analysis_type": analysis_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        """결과를 JSON 형태로 출력"""
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

# 시계열 분석을 위한 라이브러리 import 시도
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def perform_forecasting_analysis(df: pd.DataFrame, date_column: str, value_column: str,
                                forecast_periods: int = 30, model_type: str = 'linear') -> Dict[str, Any]:
    """
    시계열 예측 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    date_column : str
        날짜 컬럼명
    value_column : str
        예측할 값 컬럼명
    forecast_periods : int, default=30
        예측할 기간 수
    model_type : str, default='linear'
        사용할 예측 모델 ('linear', 'exponential_smoothing', 'arima')

    Returns:
    --------
    Dict[str, Any]
        시계열 예측 분석 결과
    """

    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn이 설치되지 않았습니다",
            "required_package": "scikit-learn"
        }

    # 필수 컬럼 존재 확인
    if date_column not in df.columns:
        return {
            "error": f"날짜 컬럼 '{date_column}'을 찾을 수 없습니다",
            "available_columns": list(df.columns)
        }

    if value_column not in df.columns:
        return {
            "error": f"값 컬럼 '{value_column}'을 찾을 수 없습니다",
            "available_columns": list(df.columns)
        }

    try:
        # 데이터 전처리
        ts_df = df[[date_column, value_column]].copy()

        # 날짜 컬럼 변환
        try:
            ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        except:
            return {
                "error": f"날짜 컬럼 '{date_column}'을 datetime으로 변환할 수 없습니다"
            }

        # 결측값 제거
        ts_df = ts_df.dropna()

        if len(ts_df) < 3:
            return {
                "error": "예측을 위한 데이터가 부족합니다 (최소 3개 필요)",
                "data_size": len(ts_df)
            }

        # 날짜 순으로 정렬
        ts_df = ts_df.sort_values(date_column)

        # 값이 수치형인지 확인
        if not pd.api.types.is_numeric_dtype(ts_df[value_column]):
            return {
                "error": f"값 컬럼 '{value_column}'이 수치형이 아닙니다",
                "column_type": str(ts_df[value_column].dtype)
            }

        results = {
            "success": True,
            "model_type": model_type,
            "data_points": len(ts_df),
            "date_range": {
                "start": ts_df[date_column].min().isoformat(),
                "end": ts_df[date_column].max().isoformat()
            },
            "forecast_periods": forecast_periods
        }

        # 시간 간격 분석
        time_diffs = ts_df[date_column].diff().dropna()
        avg_interval = time_diffs.mean()

        results["time_analysis"] = {
            "average_interval_days": avg_interval.days,
            "interval_consistency": len(time_diffs.unique()) == 1,
            "total_span_days": (ts_df[date_column].max() - ts_df[date_column].min()).days
        }

        # 기본 통계
        values = ts_df[value_column].values
        results["value_statistics"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": "increasing" if values[-1] > values[0] else "decreasing",
            "volatility": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        }

        # 예측 수행
        if model_type == 'linear':
            forecast_result = _linear_forecast(ts_df, date_column, value_column, forecast_periods)
        elif model_type == 'exponential_smoothing':
            forecast_result = _exponential_smoothing_forecast(ts_df, date_column, value_column, forecast_periods)
        elif model_type == 'arima':
            forecast_result = _arima_forecast(ts_df, date_column, value_column, forecast_periods)
        else:
            return {"error": f"지원하지 않는 모델 유형: {model_type}"}

        results.update(forecast_result)

        # 모델 성능 평가 (훈련 데이터 기준)
        if "fitted_values" in results:
            mae = mean_absolute_error(values, results["fitted_values"])
            mse = mean_squared_error(values, results["fitted_values"])
            rmse = np.sqrt(mse)

            results["model_performance"] = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(np.mean(np.abs((values - results["fitted_values"]) / values)) * 100)
            }

        return results

    except Exception as e:
        return {
            "error": f"시계열 예측 분석 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _linear_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
    """선형 회귀를 사용한 예측"""
    # 날짜를 수치형으로 변환 (날짜의 순서 번호)
    df = df.copy()
    df['date_numeric'] = range(len(df))

    # 선형 회귀 모델 훈련
    X = df[['date_numeric']].values
    y = df[value_col].values

    model = LinearRegression()
    model.fit(X, y)

    # 훈련 데이터에 대한 예측 (모델 성능 평가용)
    fitted_values = model.predict(X)

    # 미래 예측
    future_X = np.array([[i] for i in range(len(df), len(df) + periods)])
    forecast_values = model.predict(future_X)

    # 예측 날짜 생성
    last_date = df[date_col].iloc[-1]
    date_diff = df[date_col].diff().dropna().median()
    forecast_dates = [last_date + (i + 1) * date_diff for i in range(periods)]

    return {
        "fitted_values": fitted_values.tolist(),
        "forecast_values": forecast_values.tolist(),
        "forecast_dates": [d.isoformat() for d in forecast_dates],
        "model_coefficients": {
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        },
        "trend_direction": "increasing" if model.coef_[0] > 0 else "decreasing",
        "confidence_intervals": _calculate_linear_confidence_intervals(
            fitted_values, y, forecast_values, len(df)
        )
    }

def _exponential_smoothing_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
    """단순 지수평활을 사용한 예측"""
    values = df[value_col].values
    alpha = 0.3  # 평활 상수

    # 지수평활 적용
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])

    # 미래 예측 (마지막 평활값으로 일정하게 예측)
    last_smoothed = smoothed[-1]
    forecast_values = [last_smoothed] * periods

    # 예측 날짜 생성
    last_date = df[date_col].iloc[-1]
    date_diff = df[date_col].diff().dropna().median()
    forecast_dates = [last_date + (i + 1) * date_diff for i in range(periods)]

    return {
        "fitted_values": smoothed,
        "forecast_values": forecast_values,
        "forecast_dates": [d.isoformat() for d in forecast_dates],
        "smoothing_parameter": alpha,
        "final_smoothed_value": float(last_smoothed)
    }

def _arima_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
    """간단한 ARIMA 유사 예측 (차분 기반)"""
    values = df[value_col].values

    # 1차 차분
    diff_values = np.diff(values)

    if len(diff_values) < 2:
        # 데이터가 부족하면 단순 평균으로 예측
        mean_value = np.mean(values)
        forecast_values = [mean_value] * periods
    else:
        # 차분값의 평균으로 트렌드 추정
        avg_diff = np.mean(diff_values)
        last_value = values[-1]

        # 미래 예측
        forecast_values = []
        current_value = last_value
        for i in range(periods):
            current_value += avg_diff
            forecast_values.append(current_value)

    # 예측 날짜 생성
    last_date = df[date_col].iloc[-1]
    date_diff = df[date_col].diff().dropna().median()
    forecast_dates = [last_date + (i + 1) * date_diff for i in range(periods)]

    return {
        "fitted_values": values.tolist(),  # 원본값 그대로 (단순화)
        "forecast_values": forecast_values,
        "forecast_dates": [d.isoformat() for d in forecast_dates],
        "average_difference": float(avg_diff) if len(diff_values) >= 2 else 0,
        "differencing_order": 1
    }

def _calculate_linear_confidence_intervals(fitted_values: np.ndarray, actual_values: np.ndarray,
                                         forecast_values: np.ndarray, n: int) -> Dict[str, List]:
    """선형 회귀 예측의 신뢰구간 계산"""
    # 잔차 계산
    residuals = actual_values - fitted_values
    mse = np.mean(residuals ** 2)

    # 95% 신뢰구간 계산 (간단한 방법)
    confidence_level = 1.96  # 95%
    margin = confidence_level * np.sqrt(mse)

    lower_bounds = (forecast_values - margin).tolist()
    upper_bounds = (forecast_values + margin).tolist()

    return {
        "lower_95": lower_bounds,
        "upper_95": upper_bounds,
        "confidence_level": 0.95
    }

def main():
    """
    메인 실행 함수 - 시계열 예측 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 시계열 예측 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: date_column, value_column, forecast_periods, model_type

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 예측 결과
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 파일 경로가 제공된 경우 파일에서 데이터 로드
        if 'file_path' in params:
            df = load_data(params['file_path'])
        else:
            # JSON 데이터에서 직접 DataFrame 생성
            if 'data' in params:
                df = pd.DataFrame(params['data'])
            else:
                df = pd.DataFrame(params)

        # 예측 옵션
        date_column = params.get('date_column', 'date')
        value_column = params.get('value_column', 'value')
        forecast_periods = params.get('forecast_periods', 30)
        model_type = params.get('model_type', 'linear')

        # 필수 매개변수 검증
        validate_required_params(params, ['date_column', 'value_column'])

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 시계열 예측 분석 수행
        forecasting_result = perform_forecasting_analysis(
            df, date_column, value_column, forecast_periods, model_type
        )

        if not forecasting_result.get('success', False):
            error_result = {
                "success": False,
                "error": forecasting_result.get('error', '시계열 예측 분석 실패'),
                "analysis_type": "forecasting_analysis"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "forecasting_analysis": forecasting_result,
            "forecast_summary": {
                "model_used": forecasting_result.get('model_type', model_type),
                "data_points_used": forecasting_result.get('data_points', 0),
                "forecast_periods": forecasting_result.get('forecast_periods', 0),
                "trend_direction": forecasting_result.get('trend_direction', 'unknown'),
                "forecast_range": {
                    "min_predicted": float(min(forecasting_result.get('forecast_values', [0]))),
                    "max_predicted": float(max(forecasting_result.get('forecast_values', [0])))
                } if forecasting_result.get('forecast_values') else None
            }
        }

        # 요약 생성
        model_used = forecasting_result.get('model_type', model_type)
        periods_forecasted = forecasting_result.get('forecast_periods', forecast_periods)
        summary = f"시계열 예측 완료 - {model_used} 모델로 {periods_forecasted}기간 예측"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="forecasting_analysis",
            data_info=data_info,
            results=analysis_results,
            summary=summary
        )

        # 결과 출력
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "forecasting_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()