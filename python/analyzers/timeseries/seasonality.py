#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Seasonality Analysis Module
시계열 계절성 분석 모듈

이 모듈은 시계열 데이터의 계절성 패턴을 분석합니다.
주요 기능:
- 계절성 강도 및 주기 탐지
- 계절 분해 (Seasonal Decomposition)
- 주기적 패턴 분석
- 계절성 제거 및 조정
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

# 계절성 분석을 위한 라이브러리 import 시도
try:
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def perform_seasonality_analysis(df: pd.DataFrame, date_column: str, value_column: str,
                                 expected_period: Optional[int] = None) -> Dict[str, Any]:
    """
    시계열 계절성 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    date_column : str
        날짜 컬럼명
    value_column : str
        분석할 값 컬럼명
    expected_period : int, optional
        예상되는 계절성 주기

    Returns:
    --------
    Dict[str, Any]
        계절성 분석 결과
    """

    if not SCIPY_AVAILABLE:
        return {
            "error": "scipy가 설치되지 않았습니다",
            "required_package": "scipy"
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

        if len(ts_df) < 4:
            return {
                "error": "계절성 분석을 위한 데이터가 부족합니다 (최소 4개 필요)",
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
            "data_points": len(ts_df),
            "date_range": {
                "start": ts_df[date_column].min().isoformat(),
                "end": ts_df[date_column].max().isoformat()
            },
            "expected_period": expected_period
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
            "coefficient_of_variation": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        }

        # FFT 기반 주파수 분석
        fft_analysis = _perform_fft_analysis(values)
        results["frequency_analysis"] = fft_analysis

        # 자기상관 기반 계절성 탐지
        autocorr_analysis = _perform_autocorrelation_analysis(values)
        results["autocorrelation_analysis"] = autocorr_analysis

        # 계절 분해 (단순 방법)
        decomposition = _perform_seasonal_decomposition(values, expected_period)
        results["seasonal_decomposition"] = decomposition

        # 계절성 강도 측정
        seasonality_strength = _measure_seasonality_strength(
            values,
            decomposition.get("seasonal_component", []),
            fft_analysis.get("dominant_frequencies", [])
        )
        results["seasonality_strength"] = seasonality_strength

        # 계절성 패턴 분석
        pattern_analysis = _analyze_seasonal_patterns(ts_df, date_column, value_column)
        results["pattern_analysis"] = pattern_analysis

        # 계절성 분류 및 해석
        classification = _classify_seasonality(results)
        results["seasonality_classification"] = classification

        return results

    except Exception as e:
        return {
            "error": f"계절성 분석 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _perform_fft_analysis(values: np.ndarray) -> Dict[str, Any]:
    """FFT를 사용한 주파수 분석"""
    n = len(values)
    if n < 4:
        return {
            "dominant_frequencies": [],
            "frequency_powers": [],
            "method": "insufficient_data"
        }

    # 평균 제거
    detrended = values - np.mean(values)

    # FFT 수행
    fft_values = fft(detrended)
    frequencies = fftfreq(n, d=1)

    # 파워 스펙트럼 계산
    power_spectrum = np.abs(fft_values) ** 2

    # 양의 주파수만 고려
    positive_freq_idx = frequencies > 0
    positive_frequencies = frequencies[positive_freq_idx]
    positive_powers = power_spectrum[positive_freq_idx]

    # 주요 주파수 탐지 (피크 찾기)
    if len(positive_powers) > 0:
        # 피크 탐지
        peaks, _ = find_peaks(positive_powers, height=np.max(positive_powers) * 0.1)

        # 상위 피크들 정렬
        peak_frequencies = positive_frequencies[peaks]
        peak_powers = positive_powers[peaks]

        # 파워 기준 정렬
        sorted_indices = np.argsort(peak_powers)[::-1]

        dominant_frequencies = []
        for i in sorted_indices[:5]:  # 상위 5개
            freq = peak_frequencies[i]
            power = peak_powers[i]
            period = 1.0 / freq if freq > 0 else float('inf')

            dominant_frequencies.append({
                "frequency": float(freq),
                "period": float(period),
                "power": float(power),
                "relative_power": float(power / np.max(positive_powers))
            })
    else:
        dominant_frequencies = []

    return {
        "dominant_frequencies": dominant_frequencies,
        "total_power": float(np.sum(positive_powers)) if len(positive_powers) > 0 else 0,
        "method": "fft"
    }

def _perform_autocorrelation_analysis(values: np.ndarray, max_lag: int = None) -> Dict[str, Any]:
    """자기상관 함수를 사용한 계절성 탐지"""
    n = len(values)
    if n < 4:
        return {
            "significant_lags": [],
            "max_autocorrelation": 0,
            "method": "insufficient_data"
        }

    if max_lag is None:
        max_lag = min(n // 2, 50)

    # 자기상관 계산
    autocorrelations = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break

        corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
        if not np.isnan(corr):
            autocorrelations.append({
                "lag": lag,
                "correlation": float(corr)
            })

    if not autocorrelations:
        return {
            "significant_lags": [],
            "max_autocorrelation": 0,
            "method": "autocorrelation"
        }

    # 유의한 자기상관 탐지 (임계값 기준)
    threshold = 2.0 / np.sqrt(n)  # 95% 신뢰구간
    significant_lags = [
        ac for ac in autocorrelations
        if abs(ac["correlation"]) > threshold
    ]

    # 최대 자기상관 찾기
    max_autocorr = max(autocorrelations, key=lambda x: abs(x["correlation"]))

    return {
        "autocorrelations": autocorrelations,
        "significant_lags": significant_lags,
        "max_autocorrelation": float(max_autocorr["correlation"]),
        "max_lag": int(max_autocorr["lag"]),
        "threshold": float(threshold),
        "method": "autocorrelation"
    }

def _perform_seasonal_decomposition(values: np.ndarray, expected_period: Optional[int] = None) -> Dict[str, Any]:
    """간단한 계절 분해"""
    n = len(values)

    # 주기 결정
    if expected_period is None:
        # 간단한 방법: 데이터 길이에 따라 자동 결정
        if n >= 12:
            period = 12  # 월별 데이터 가정
        elif n >= 7:
            period = 7   # 주별 데이터 가정
        else:
            period = min(4, n // 2)  # 최소 주기
    else:
        period = expected_period

    if period >= n or period < 2:
        return {
            "seasonal_component": values.tolist(),
            "trend_component": [np.mean(values)] * len(values),
            "residual_component": [0] * len(values),
            "period": period,
            "method": "no_decomposition"
        }

    try:
        # 단순 이동평균을 사용한 트렌드 추출
        if period % 2 == 0:
            # 짝수 주기
            window = period
        else:
            # 홀수 주기
            window = period

        # 트렌드 계산 (중앙 이동평균)
        trend = pd.Series(values).rolling(window=window, center=True).mean()
        trend = trend.fillna(method='bfill').fillna(method='ffill')

        # 디트렌드된 값
        detrended = values - trend.values

        # 계절성 컴포넌트 계산
        seasonal = np.zeros(len(values))
        for i in range(period):
            indices = np.arange(i, len(values), period)
            if len(indices) > 0:
                seasonal_avg = np.mean(detrended[indices])
                seasonal[indices] = seasonal_avg

        # 잔차 계산
        residual = values - trend.values - seasonal

        return {
            "seasonal_component": seasonal.tolist(),
            "trend_component": trend.tolist(),
            "residual_component": residual.tolist(),
            "period": period,
            "seasonal_strength": float(np.var(seasonal) / np.var(values)) if np.var(values) > 0 else 0,
            "trend_strength": float(np.var(trend) / np.var(values)) if np.var(values) > 0 else 0,
            "method": "moving_average_decomposition"
        }

    except Exception:
        # 실패 시 기본값 반환
        return {
            "seasonal_component": [0] * len(values),
            "trend_component": [np.mean(values)] * len(values),
            "residual_component": values.tolist(),
            "period": period,
            "method": "decomposition_failed"
        }

def _measure_seasonality_strength(values: np.ndarray, seasonal_component: List[float],
                                 dominant_frequencies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """계절성 강도 측정"""
    if len(seasonal_component) == 0:
        seasonal_component = [0] * len(values)

    seasonal_array = np.array(seasonal_component)

    # 계절성 강도 계산 방법들
    measures = {}

    # 1. 분산 비율 기반
    total_variance = np.var(values)
    seasonal_variance = np.var(seasonal_array)
    if total_variance > 0:
        measures["variance_ratio"] = float(seasonal_variance / total_variance)
    else:
        measures["variance_ratio"] = 0

    # 2. 주파수 도메인 기반
    if dominant_frequencies:
        # 가장 강한 주파수의 상대적 파워
        measures["frequency_strength"] = float(dominant_frequencies[0]["relative_power"])
        measures["dominant_period"] = float(dominant_frequencies[0]["period"])
    else:
        measures["frequency_strength"] = 0
        measures["dominant_period"] = None

    # 3. 전체 계절성 강도 (0-1)
    seasonality_strength = max(
        measures["variance_ratio"],
        measures["frequency_strength"]
    )
    measures["overall_strength"] = float(seasonality_strength)

    # 4. 계절성 분류
    if seasonality_strength >= 0.6:
        strength_level = "very_strong"
    elif seasonality_strength >= 0.4:
        strength_level = "strong"
    elif seasonality_strength >= 0.2:
        strength_level = "moderate"
    elif seasonality_strength >= 0.1:
        strength_level = "weak"
    else:
        strength_level = "very_weak"

    measures["strength_level"] = strength_level

    return measures

def _analyze_seasonal_patterns(df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
    """계절 패턴 분석"""
    try:
        # 날짜 정보 추출
        df_copy = df.copy()
        df_copy['month'] = df_copy[date_column].dt.month
        df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
        df_copy['day_of_year'] = df_copy[date_column].dt.dayofyear

        patterns = {}

        # 월별 패턴
        if len(df_copy['month'].unique()) > 1:
            monthly_stats = df_copy.groupby('month')[value_column].agg(['mean', 'std', 'count']).to_dict()
            patterns["monthly"] = {
                "means": {int(k): float(v) for k, v in monthly_stats['mean'].items()},
                "stds": {int(k): float(v) for k, v in monthly_stats['std'].items()},
                "counts": {int(k): int(v) for k, v in monthly_stats['count'].items()}
            }

        # 요일별 패턴
        if len(df_copy['day_of_week'].unique()) > 1:
            daily_stats = df_copy.groupby('day_of_week')[value_column].agg(['mean', 'std', 'count']).to_dict()
            patterns["daily"] = {
                "means": {int(k): float(v) for k, v in daily_stats['mean'].items()},
                "stds": {int(k): float(v) for k, v in daily_stats['std'].items()},
                "counts": {int(k): int(v) for k, v in daily_stats['count'].items()}
            }

        # 패턴 강도 계산
        pattern_strengths = {}
        for pattern_name, pattern_data in patterns.items():
            if 'means' in pattern_data:
                means = list(pattern_data['means'].values())
                if len(means) > 1:
                    pattern_strengths[pattern_name] = float(np.std(means) / np.mean(means)) if np.mean(means) != 0 else 0

        return {
            "patterns": patterns,
            "pattern_strengths": pattern_strengths,
            "strongest_pattern": max(pattern_strengths.items(), key=lambda x: x[1])[0] if pattern_strengths else None
        }

    except Exception:
        return {
            "patterns": {},
            "pattern_strengths": {},
            "strongest_pattern": None
        }

def _classify_seasonality(results: Dict[str, Any]) -> Dict[str, Any]:
    """계절성 분류 및 해석"""
    strength = results.get("seasonality_strength", {})
    overall_strength = strength.get("overall_strength", 0)
    strength_level = strength.get("strength_level", "very_weak")
    dominant_period = strength.get("dominant_period")

    # 계절성 유형 분류
    if dominant_period:
        if 11 <= dominant_period <= 13:
            seasonality_type = "yearly"
        elif 6 <= dominant_period <= 8:
            seasonality_type = "weekly"
        elif 3 <= dominant_period <= 5:
            seasonality_type = "short_cycle"
        else:
            seasonality_type = "custom"
    else:
        seasonality_type = "none"

    # 해석 생성
    interpretation = _generate_seasonality_interpretation(strength_level, seasonality_type, dominant_period)

    # 권장사항 생성
    recommendations = _generate_seasonality_recommendations(strength_level, seasonality_type)

    return {
        "strength_level": strength_level,
        "seasonality_type": seasonality_type,
        "dominant_period": dominant_period,
        "overall_strength": overall_strength,
        "interpretation": interpretation,
        "recommendations": recommendations
    }

def _generate_seasonality_interpretation(strength_level: str, seasonality_type: str, period: Optional[float]) -> str:
    """계절성 해석 텍스트 생성"""
    interpretations = {
        "very_strong": {
            "yearly": "매우 강한 연간 계절성 패턴이 관찰됩니다",
            "weekly": "매우 강한 주간 계절성 패턴이 관찰됩니다",
            "short_cycle": "매우 강한 단기 주기 패턴이 관찰됩니다",
            "custom": f"매우 강한 {period:.1f} 주기의 계절성 패턴이 관찰됩니다" if period else "매우 강한 계절성 패턴이 관찰됩니다",
            "none": "계절성이 매우 강하지만 명확한 주기는 식별되지 않았습니다"
        },
        "strong": {
            "yearly": "강한 연간 계절성 패턴이 나타납니다",
            "weekly": "강한 주간 계절성 패턴이 나타납니다",
            "short_cycle": "강한 단기 주기 패턴이 나타납니다",
            "custom": f"강한 {period:.1f} 주기의 계절성 패턴이 나타납니다" if period else "강한 계절성 패턴이 나타납니다",
            "none": "강한 계절성이 있지만 명확한 주기는 식별되지 않았습니다"
        },
        "moderate": {
            "yearly": "보통 수준의 연간 계절성이 있습니다",
            "weekly": "보통 수준의 주간 계절성이 있습니다",
            "short_cycle": "보통 수준의 단기 주기 패턴이 있습니다",
            "custom": f"보통 수준의 {period:.1f} 주기 계절성이 있습니다" if period else "보통 수준의 계절성이 있습니다",
            "none": "약간의 계절성이 있지만 패턴이 불분명합니다"
        },
        "weak": {
            "yearly": "약한 연간 계절성 경향이 있습니다",
            "weekly": "약한 주간 계절성 경향이 있습니다",
            "short_cycle": "약한 단기 주기 경향이 있습니다",
            "custom": f"약한 {period:.1f} 주기 계절성 경향이 있습니다" if period else "약한 계절성 경향이 있습니다",
            "none": "계절성이 매우 약합니다"
        },
        "very_weak": {
            "none": "명확한 계절성 패턴이 관찰되지 않습니다"
        }
    }

    return interpretations.get(strength_level, {}).get(seasonality_type,
                                                      interpretations.get(strength_level, {}).get("none",
                                                                                                   "계절성 분석 결과를 해석할 수 없습니다"))

def _generate_seasonality_recommendations(strength_level: str, seasonality_type: str) -> List[str]:
    """계절성 기반 권장사항 생성"""
    recommendations = []

    if strength_level in ["very_strong", "strong"]:
        recommendations.append("강한 계절성이 있으므로 계절 조정을 고려하세요")
        if seasonality_type == "yearly":
            recommendations.append("연간 패턴을 활용한 예측 모델을 사용하세요")
        elif seasonality_type == "weekly":
            recommendations.append("주간 패턴을 고려한 운영 계획을 수립하세요")

    elif strength_level == "moderate":
        recommendations.append("계절성을 고려한 분석을 수행하세요")

    elif strength_level in ["weak", "very_weak"]:
        recommendations.append("계절성이 약하므로 트렌드 분석에 집중하세요")

    if seasonality_type == "none":
        recommendations.append("다른 주기성이나 외부 요인을 검토해보세요")

    if not recommendations:
        recommendations.append("계절성 패턴을 지속적으로 모니터링하세요")

    return recommendations

def main():
    """
    메인 실행 함수 - 계절성 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 계절성 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: date_column, value_column, expected_period

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 계절성 분석 결과
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

        # 분석 옵션
        date_column = params.get('date_column', 'date')
        value_column = params.get('value_column', 'value')
        expected_period = params.get('expected_period')

        # 필수 매개변수 검증
        validate_required_params(params, ['date_column', 'value_column'])

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 계절성 분석 수행
        seasonality_result = perform_seasonality_analysis(df, date_column, value_column, expected_period)

        if not seasonality_result.get('success', False):
            error_result = {
                "success": False,
                "error": seasonality_result.get('error', '계절성 분석 실패'),
                "analysis_type": "seasonality_analysis"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "seasonality_analysis": seasonality_result,
            "seasonality_summary": {
                "data_points_analyzed": seasonality_result.get('data_points', 0),
                "seasonality_strength": seasonality_result.get('seasonality_strength', {}).get('overall_strength', 0),
                "strength_level": seasonality_result.get('seasonality_classification', {}).get('strength_level', 'unknown'),
                "seasonality_type": seasonality_result.get('seasonality_classification', {}).get('seasonality_type', 'unknown'),
                "dominant_period": seasonality_result.get('seasonality_strength', {}).get('dominant_period'),
                "significant_patterns": len(seasonality_result.get('pattern_analysis', {}).get('patterns', {}))
            }
        }

        # 요약 생성
        strength_level = seasonality_result.get('seasonality_classification', {}).get('strength_level', 'unknown')
        seasonality_type = seasonality_result.get('seasonality_classification', {}).get('seasonality_type', 'unknown')
        summary = f"계절성 분석 완료 - {strength_level} {seasonality_type} 계절성 패턴 탐지"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="seasonality_analysis",
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
            "analysis_type": "seasonality_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()