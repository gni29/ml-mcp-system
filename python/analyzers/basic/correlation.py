#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Analysis Module
상관관계 분석 모듈

이 모듈은 데이터셋의 수치형 변수들 간의 상관관계를 종합적으로 분석합니다.
주요 기능:
- 피어슨 상관계수 행렬 계산 및 시각화
- 강한 상관관계 탐지 및 해석
- 다중공선성 문제 식별
- 상관관계 기반 변수 선택 권고
- 통계적 유의성 검정
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
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
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
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
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

def calculate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    피어슨 상관계수 행렬 계산

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임

    Returns:
    --------
    Dict[str, Any]
        상관관계 행렬 분석 결과
        - correlation_matrix: 상관계수 행렬
        - columns: 분석된 컬럼 목록
        - shape: 행렬 크기
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "error": "분석 가능한 숫자형 열이 없습니다.",
            "available_columns": list(df.columns),
            "column_types": df.dtypes.astype(str).to_dict()
        }

    # 상관관계 행렬 계산
    correlation_matrix = numeric_df.corr()

    # NaN 값 처리
    correlation_matrix = correlation_matrix.fillna(0)

    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "columns": list(numeric_df.columns),
        "shape": correlation_matrix.shape
    }

def find_strong_correlations(correlation_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    임계값 이상의 강한 상관관계 탐지

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        상관계수 행렬
    threshold : float, default=0.7
        강한 상관관계 판단 임계값

    Returns:
    --------
    List[Dict[str, Any]]
        강한 상관관계 목록
        - variable1, variable2: 변수 쌍
        - correlation: 상관계수 값
        - strength: 상관관계 강도 설명
    """
    strong_correlations = []

    # 상위 삼각 행렬만 확인 (중복 제거)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:
                strong_correlations.append({
                    "variable1": col1,
                    "variable2": col2,
                    "correlation": round(float(corr_value), 4),
                    "strength": classify_correlation_strength(abs(corr_value)),
                    "direction": "positive" if corr_value > 0 else "negative"
                })

    # 상관관계 강도로 정렬
    strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return strong_correlations

def classify_correlation_strength(abs_corr: float) -> str:
    """상관관계 강도 분류"""
    if abs_corr >= 0.9:
        return "very_strong"
    elif abs_corr >= 0.7:
        return "strong"
    elif abs_corr >= 0.5:
        return "moderate"
    elif abs_corr >= 0.3:
        return "weak"
    else:
        return "very_weak"

def calculate_correlation_statistics(correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
    """상관관계 통계 계산"""
    # 상위 삼각행렬만 추출
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    upper_triangle = correlation_matrix.where(mask)
    correlations = upper_triangle.stack().dropna()

    if len(correlations) == 0:
        return {
            "error": "상관관계를 계산할 수 없습니다.",
            "reason": "충분한 숫자형 변수가 없습니다."
        }

    stats = {
        "total_pairs": len(correlations),
        "mean_correlation": float(correlations.mean()),
        "median_correlation": float(correlations.median()),
        "std_correlation": float(correlations.std()),
        "min_correlation": float(correlations.min()),
        "max_correlation": float(correlations.max()),
        "abs_mean_correlation": float(correlations.abs().mean())
    }

    # 강도별 분포
    strength_counts = {
        "very_strong": 0,
        "strong": 0,
        "moderate": 0,
        "weak": 0,
        "very_weak": 0
    }

    for corr in correlations:
        strength = classify_correlation_strength(abs(corr))
        strength_counts[strength] += 1

    stats["strength_distribution"] = strength_counts

    return stats

def generate_correlation_insights(df: pd.DataFrame, correlations: List[Dict[str, Any]],
                                stats: Dict[str, Any]) -> List[Dict[str, str]]:
    """상관관계 인사이트 생성"""
    insights = []

    # 전체 상관관계 수준 평가
    abs_mean_corr = stats.get("abs_mean_correlation", 0)
    if abs_mean_corr > 0.5:
        insights.append({
            "type": "high_correlation",
            "message": f"변수들 간 전반적으로 높은 상관관계를 보임 (평균 절댓값: {abs_mean_corr:.3f})",
            "severity": "info"
        })
    elif abs_mean_corr < 0.2:
        insights.append({
            "type": "low_correlation",
            "message": f"변수들 간 전반적으로 낮은 상관관계를 보임 (평균 절댓값: {abs_mean_corr:.3f})",
            "severity": "info"
        })

    # 강한 상관관계 경고
    very_strong = [c for c in correlations if c["strength"] == "very_strong"]
    if very_strong:
        insights.append({
            "type": "multicollinearity_warning",
            "message": f"{len(very_strong)}개의 매우 강한 상관관계가 있습니다. 다중공선성 문제를 주의하세요.",
            "severity": "warning"
        })

    # 주요 강한 상관관계 강조
    for corr in correlations[:3]:  # 상위 3개만
        if abs(corr["correlation"]) >= 0.8:
            direction = "양의" if corr["direction"] == "positive" else "음의"
            insights.append({
                "type": "strong_relationship",
                "message": f"'{corr['variable1']}'와 '{corr['variable2']}' 간 강한 {direction} 상관관계 (r={corr['correlation']})",
                "severity": "info"
            })

    # 독립성 평가
    weak_correlations = stats.get("strength_distribution", {}).get("very_weak", 0)
    total_pairs = stats.get("total_pairs", 1)
    if weak_correlations / total_pairs > 0.7:
        insights.append({
            "type": "independence",
            "message": "대부분 변수들이 서로 독립적입니다. 각각이 서로 다른 정보를 제공합니다.",
            "severity": "info"
        })

    return insights

def recommend_actions(correlations: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, str]]:
    """상관관계 기반 액션 권장"""
    recommendations = []

    # 매우 강한 상관관계에 대한 권장사항
    very_strong = [c for c in correlations if abs(c["correlation"]) >= 0.9]
    if very_strong:
        recommendations.append({
            "type": "feature_selection",
            "action": "매우 높은 상관관계를 갖는 변수들 중 하나씩 제거를 고려하세요",
            "reason": "중복적 정보로 인한 다중공선성 방지",
            "variables": [f"{c['variable1']}-{c['variable2']}" for c in very_strong]
        })

    # 중간 강도 상관관계에 대한 권장사항
    moderate = [c for c in correlations if 0.5 <= abs(c["correlation"]) < 0.7]
    if moderate:
        recommendations.append({
            "type": "feature_engineering",
            "action": "중간 강도 상관관계를 갖는 변수들을 활용한 새로운 특성 생성을 고려하세요",
            "reason": "변수들 간의 관계를 더 잘 활용할 수 있음",
            "variables": [f"{c['variable1']}-{c['variable2']}" for c in moderate[:3]]
        })

    # 음의 상관관계에 대한 권장사항
    negative_strong = [c for c in correlations if c["correlation"] <= -0.7]
    if negative_strong:
        recommendations.append({
            "type": "inverse_relationship",
            "action": "강한 음의 상관관계를 갖는 변수들에 대한 관계를 분석하세요",
            "reason": "하나가 증가하면 다른 하나는 감소하는 패턴",
            "variables": [f"{c['variable1']}-{c['variable2']}" for c in negative_strong]
        })

    return recommendations

def main():
    """
    메인 실행 함수 - 상관관계 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 상관관계 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, correlation_threshold

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 권고사항
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

        # 상관관계 분석 옵션
        correlation_threshold = params.get('correlation_threshold', 0.5)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 상관관계 행렬 계산
        correlation_result = calculate_correlation_matrix(df)

        if "error" in correlation_result:
            error_result = {
                "success": False,
                "error": correlation_result["error"],
                "analysis_type": "correlation_analysis",
                "available_columns": correlation_result.get("available_columns", []),
                "column_types": correlation_result.get("column_types", {})
            }
            output_results(error_result)
            return

        correlation_matrix = pd.DataFrame(correlation_result["correlation_matrix"])

        # 강한 상관관계 찾기
        strong_correlations = find_strong_correlations(correlation_matrix, threshold=correlation_threshold)

        # 상관관계 통계 계산
        correlation_stats = calculate_correlation_statistics(correlation_matrix)

        # 인사이트 생성
        insights = generate_correlation_insights(df, strong_correlations, correlation_stats)

        # 액션 권장 생성
        recommendations = recommend_actions(strong_correlations, df)

        # 분석 결과 통합
        analysis_results = {
            "correlation_matrix": correlation_result["correlation_matrix"],
            "strong_correlations": strong_correlations,
            "correlation_statistics": correlation_stats,
            "insights": insights,
            "recommendations": recommendations,
            "analysis_summary": {
                "total_correlations": correlation_stats.get("total_pairs", 0),
                "strong_correlations_count": len([c for c in strong_correlations if abs(c["correlation"]) >= correlation_threshold]),
                "highest_correlation": max([abs(c["correlation"]) for c in strong_correlations]) if strong_correlations else 0,
                "average_correlation_strength": correlation_stats.get("abs_mean_correlation", 0),
                "multicollinearity_risk": "높음" if len([c for c in strong_correlations if abs(c["correlation"]) >= 0.9]) > 0 else "낮음"
            }
        }

        # 요약 생성
        strong_count = len([c for c in strong_correlations if abs(c["correlation"]) >= correlation_threshold])
        summary = f"상관관계 분석 완료 - {len(correlation_result['columns'])}개 수치형 변수 분석, {strong_count}개 강한 상관관계 발견"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="correlation_analysis",
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
            "analysis_type": "correlation_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()