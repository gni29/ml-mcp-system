#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Trend Analysis Module
시계열 트렌드 분석 모듈

이 모듈은 시계열 데이터의 트렌드를 분석하고 변화점을 탐지합니다.
주요 기능:
- 선형/비선형 트렌드 분석
- 트렌드 변화점(Change Point) 탐지
- 계절성 제거 후 트렌드 추출
- 트렌드 강도 및 방향 분석
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

# 트렌드 분석을 위한 라이브러리 import 시도
try:
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    from scipy.signal import savgol_filter
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def perform_trend_analysis(df: pd.DataFrame, date_column: str, value_column: str,
                          trend_method: str = 'linear') -> Dict[str, Any]:
    """
    시계열 트렌드 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    date_column : str
        날짜 컬럼명
    value_column : str
        분석할 값 컬럼명
    trend_method : str, default='linear'
        트렌드 분석 방법 ('linear', 'polynomial', 'moving_average')

    Returns:
    --------
    Dict[str, Any]
        트렌드 분석 결과
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
                "error": "트렌드 분석을 위한 데이터가 부족합니다 (최소 3개 필요)",
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
            "method": trend_method,
            "data_points": len(ts_df),
            "date_range": {
                "start": ts_df[date_column].min().isoformat(),
                "end": ts_df[date_column].max().isoformat()
            }
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
            "range": float(np.max(values) - np.min(values)),
            "coefficient_of_variation": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        }

        # 트렌드 분석 수행
        if trend_method == 'linear':
            trend_result = _linear_trend_analysis(ts_df, date_column, value_column)
        elif trend_method == 'polynomial':
            trend_result = _polynomial_trend_analysis(ts_df, date_column, value_column)
        elif trend_method == 'moving_average':
            trend_result = _moving_average_trend_analysis(ts_df, date_column, value_column)
        else:
            return {"error": f"지원하지 않는 트렌드 분석 방법: {trend_method}"}

        results.update(trend_result)

        # 변화점 탐지
        change_points = _detect_change_points(values)
        results["change_point_analysis"] = change_points

        # 트렌드 강도 분석
        trend_strength = _analyze_trend_strength(values, results.get("trend_line", values))
        results["trend_strength"] = trend_strength

        # 트렌드 분류
        trend_classification = _classify_trend(results)
        results["trend_classification"] = trend_classification

        return results

    except Exception as e:
        return {
            "error": f"트렌드 분석 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _linear_trend_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
    """선형 트렌드 분석"""
    # 날짜를 수치형으로 변환 (날짜의 순서 번호)
    df = df.copy()
    df['date_numeric'] = range(len(df))

    # 선형 회귀 모델
    X = df[['date_numeric']].values
    y = df[value_col].values

    model = LinearRegression()
    model.fit(X, y)

    # 트렌드 라인 계산
    trend_line = model.predict(X)

    # 통계적 유의성 검증
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)

    return {
        "trend_type": "linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "standard_error": float(std_err),
        "trend_line": trend_line.tolist(),
        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
        "trend_significance": "significant" if p_value < 0.05 else "not_significant",
        "trend_strength": abs(r_value)
    }

def _polynomial_trend_analysis(df: pd.DataFrame, date_col: str, value_col: str, degree: int = 2) -> Dict[str, Any]:
    """다항식 트렌드 분석"""
    df = df.copy()
    df['date_numeric'] = range(len(df))

    X = df['date_numeric'].values
    y = df[value_col].values

    # 다항식 피팅
    coefficients = np.polyfit(X, y, degree)
    poly_func = np.poly1d(coefficients)
    trend_line = poly_func(X)

    # R² 계산
    ss_res = np.sum((y - trend_line) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        "trend_type": "polynomial",
        "degree": degree,
        "coefficients": coefficients.tolist(),
        "r_squared": float(r_squared),
        "trend_line": trend_line.tolist(),
        "curvature": "concave_up" if coefficients[0] > 0 else "concave_down" if coefficients[0] < 0 else "linear",
        "turning_points": _find_turning_points(coefficients, len(X))
    }

def _moving_average_trend_analysis(df: pd.DataFrame, date_col: str, value_col: str, window: int = None) -> Dict[str, Any]:
    """이동평균 기반 트렌드 분석"""
    values = df[value_col].values

    # 윈도우 크기 자동 결정
    if window is None:
        window = max(3, len(values) // 10)

    # 이동평균 계산
    if len(values) >= window:
        moving_avg = pd.Series(values).rolling(window=window, center=True).mean()
        trend_line = moving_avg.fillna(method='bfill').fillna(method='ffill').values
    else:
        trend_line = values

    # 트렌드 방향 분석
    trend_changes = np.diff(trend_line)
    increasing_count = np.sum(trend_changes > 0)
    decreasing_count = np.sum(trend_changes < 0)

    if increasing_count > decreasing_count:
        overall_direction = "increasing"
    elif decreasing_count > increasing_count:
        overall_direction = "decreasing"
    else:
        overall_direction = "mixed"

    return {
        "trend_type": "moving_average",
        "window_size": window,
        "trend_line": trend_line.tolist(),
        "trend_direction": overall_direction,
        "trend_changes": {
            "increasing_periods": int(increasing_count),
            "decreasing_periods": int(decreasing_count),
            "stable_periods": int(len(trend_changes) - increasing_count - decreasing_count)
        },
        "smoothness_factor": float(1 - (np.std(np.diff(trend_line)) / np.std(np.diff(values)))) if np.std(np.diff(values)) > 0 else 1.0
    }

def _detect_change_points(values: np.ndarray) -> Dict[str, Any]:
    """변화점 탐지 (간단한 방법)"""
    if len(values) < 6:
        return {
            "change_points": [],
            "n_change_points": 0,
            "method": "insufficient_data"
        }

    # 기울기 변화 기반 변화점 탐지
    window_size = max(3, len(values) // 10)
    change_points = []

    for i in range(window_size, len(values) - window_size):
        # 이전 구간과 이후 구간의 기울기 계산
        before_slope = np.polyfit(range(window_size), values[i-window_size:i], 1)[0]
        after_slope = np.polyfit(range(window_size), values[i:i+window_size], 1)[0]

        # 기울기 변화가 큰 지점을 변화점으로 탐지
        slope_diff = abs(after_slope - before_slope)
        if slope_diff > np.std(values) / 2:  # 임계값
            change_points.append({
                "index": int(i),
                "before_slope": float(before_slope),
                "after_slope": float(after_slope),
                "slope_change": float(slope_diff)
            })

    return {
        "change_points": change_points,
        "n_change_points": len(change_points),
        "method": "slope_based",
        "window_size": window_size
    }

def _analyze_trend_strength(original_values: np.ndarray, trend_line: np.ndarray) -> Dict[str, Any]:
    """트렌드 강도 분석"""
    # 트렌드 성분과 잔차 분석
    residuals = original_values - trend_line
    trend_variance = np.var(trend_line)
    residual_variance = np.var(residuals)
    total_variance = np.var(original_values)

    # 트렌드 강도 계산 (0~1)
    trend_strength = trend_variance / total_variance if total_variance > 0 else 0

    # 트렌드 일관성 (낮은 잔차 분산 = 높은 일관성)
    trend_consistency = 1 - (residual_variance / total_variance) if total_variance > 0 else 1

    return {
        "trend_strength": float(trend_strength),
        "trend_consistency": float(trend_consistency),
        "trend_variance_ratio": float(trend_variance / total_variance) if total_variance > 0 else 0,
        "residual_variance_ratio": float(residual_variance / total_variance) if total_variance > 0 else 0,
        "signal_to_noise_ratio": float(trend_variance / residual_variance) if residual_variance > 0 else float('inf')
    }

def _classify_trend(results: Dict[str, Any]) -> Dict[str, Any]:
    """트렌드 분류 및 해석"""
    trend_strength = results.get("trend_strength", {}).get("trend_strength", 0)
    trend_direction = results.get("trend_direction", "unknown")
    r_squared = results.get("r_squared", 0)

    # 트렌드 강도 분류
    if trend_strength >= 0.7:
        strength_category = "very_strong"
    elif trend_strength >= 0.5:
        strength_category = "strong"
    elif trend_strength >= 0.3:
        strength_category = "moderate"
    elif trend_strength >= 0.1:
        strength_category = "weak"
    else:
        strength_category = "very_weak"

    # 트렌드 품질 평가
    if r_squared >= 0.8:
        trend_quality = "excellent"
    elif r_squared >= 0.6:
        trend_quality = "good"
    elif r_squared >= 0.4:
        trend_quality = "fair"
    else:
        trend_quality = "poor"

    return {
        "strength_category": strength_category,
        "direction": trend_direction,
        "quality": trend_quality,
        "interpretation": _get_trend_interpretation(strength_category, trend_direction, trend_quality),
        "recommendations": _get_trend_recommendations(strength_category, trend_direction, trend_quality)
    }

def _find_turning_points(coefficients: np.ndarray, data_length: int) -> List[Dict[str, Any]]:
    """다항식 함수의 변곡점 찾기"""
    if len(coefficients) < 3:
        return []

    # 1차 도함수의 계수 계산
    derivative_coeffs = []
    for i, coef in enumerate(coefficients[:-1]):
        derivative_coeffs.append(coef * (len(coefficients) - 1 - i))

    # 실근 찾기 (변곡점)
    if len(derivative_coeffs) >= 2:
        roots = np.roots(derivative_coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and 0 <= r.real < data_length]

        turning_points = []
        for root in real_roots:
            turning_points.append({
                "x": float(root),
                "type": "turning_point"
            })

        return turning_points

    return []

def _get_trend_interpretation(strength: str, direction: str, quality: str) -> str:
    """트렌드 해석 텍스트 생성"""
    interpretations = {
        ("very_strong", "increasing"): "매우 강한 상승 트렌드가 관찰됩니다",
        ("very_strong", "decreasing"): "매우 강한 하락 트렌드가 관찰됩니다",
        ("strong", "increasing"): "명확한 상승 트렌드가 나타납니다",
        ("strong", "decreasing"): "명확한 하락 트렌드가 나타납니다",
        ("moderate", "increasing"): "보통 수준의 상승 경향을 보입니다",
        ("moderate", "decreasing"): "보통 수준의 하락 경향을 보입니다",
        ("weak", "increasing"): "약한 상승 경향이 있습니다",
        ("weak", "decreasing"): "약한 하락 경향이 있습니다",
        ("very_weak", "flat"): "명확한 트렌드가 없이 평평한 패턴입니다"
    }

    key = (strength, direction)
    return interpretations.get(key, f"{strength} {direction} 트렌드입니다")

def _get_trend_recommendations(strength: str, direction: str, quality: str) -> List[str]:
    """트렌드 기반 권장사항 생성"""
    recommendations = []

    if strength in ["very_strong", "strong"]:
        if direction == "increasing":
            recommendations.append("상승 트렌드가 강하므로 성장 기회를 활용할 수 있습니다")
        elif direction == "decreasing":
            recommendations.append("하락 트렌드가 강하므로 개선 조치가 필요합니다")

    if quality == "poor":
        recommendations.append("트렌드의 설명력이 낮으므로 다른 분석 방법을 고려해보세요")

    if strength == "very_weak":
        recommendations.append("트렌드가 약하므로 단기 변동에 주의하세요")

    if not recommendations:
        recommendations.append("현재 트렌드 패턴을 지속적으로 모니터링하세요")

    return recommendations

def main():
    """
    메인 실행 함수 - 트렌드 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 트렌드 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: date_column, value_column, trend_method

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 트렌드 분석 결과
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
        trend_method = params.get('trend_method', 'linear')

        # 필수 매개변수 검증
        validate_required_params(params, ['date_column', 'value_column'])

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 트렌드 분석 수행
        trend_result = perform_trend_analysis(df, date_column, value_column, trend_method)

        if not trend_result.get('success', False):
            error_result = {
                "success": False,
                "error": trend_result.get('error', '트렌드 분석 실패'),
                "analysis_type": "trend_analysis"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "trend_analysis": trend_result,
            "trend_summary": {
                "method_used": trend_result.get('method', trend_method),
                "data_points_analyzed": trend_result.get('data_points', 0),
                "trend_direction": trend_result.get('trend_direction', 'unknown'),
                "trend_strength": trend_result.get('trend_strength', {}).get('trend_strength', 0),
                "trend_significance": trend_result.get('trend_significance', 'unknown'),
                "change_points_detected": trend_result.get('change_point_analysis', {}).get('n_change_points', 0)
            }
        }

        # 요약 생성
        method_used = trend_result.get('method', trend_method)
        direction = trend_result.get('trend_direction', 'unknown')
        strength = trend_result.get('trend_classification', {}).get('strength_category', 'unknown')
        summary = f"트렌드 분석 완료 - {method_used} 방법으로 {strength} {direction} 트렌드 탐지"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="trend_analysis",
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
            "analysis_type": "trend_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()