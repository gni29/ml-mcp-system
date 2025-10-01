#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Analysis for Lightweight Analysis MCP
경량 분석 MCP용 상관관계 분석 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params, get_numeric_columns
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
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

    def get_numeric_columns(df: pd.DataFrame, min_unique: int = 2) -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if df[col].nunique() >= min_unique]


def calculate_correlation_analysis(df: pd.DataFrame, method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
    """
    Perform comprehensive correlation analysis
    포괄적인 상관관계 분석 수행
    """
    # Get numeric columns
    numeric_cols = get_numeric_columns(df, min_unique=2)

    if len(numeric_cols) < 2:
        return {
            "error": "상관관계 분석을 위해서는 최소 2개의 수치형 변수가 필요합니다",
            "available_numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "total_columns": len(df.columns)
        }

    subset_df = df[numeric_cols]
    results = {}

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = subset_df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = subset_df.corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = subset_df.corr(method='kendall')
    else:
        corr_matrix = subset_df.corr(method='pearson')
        method = 'pearson'

    # Handle NaN values
    corr_matrix = corr_matrix.fillna(0)

    results["correlation_matrix"] = corr_matrix.to_dict()
    results["method"] = method
    results["analyzed_variables"] = len(numeric_cols)

    # Find strong correlations
    strong_correlations = find_strong_correlations(corr_matrix, threshold)
    results["strong_correlations"] = strong_correlations

    # Calculate correlation statistics
    correlation_stats = calculate_correlation_statistics(corr_matrix)
    results["correlation_statistics"] = correlation_stats

    # Find maximum correlation
    upper_triangle = np.triu(corr_matrix.values, k=1)
    max_corr_idx = np.unravel_index(np.argmax(np.abs(upper_triangle)), upper_triangle.shape)
    max_correlation = corr_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
    results["max_correlation"] = float(max_correlation)

    # Generate insights
    insights = generate_correlation_insights(strong_correlations, correlation_stats, threshold)
    results["insights"] = insights

    # Generate recommendations
    recommendations = generate_recommendations(strong_correlations, method, threshold)
    results["recommendations"] = recommendations

    return results


def find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
    """
    Find correlations above threshold
    임계값 이상의 상관관계 찾기
    """
    strong_correlations = []

    # Get upper triangle to avoid duplicates
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:
                strong_correlations.append({
                    "variable1": corr_matrix.columns[i],
                    "variable2": corr_matrix.columns[j],
                    "correlation": round(float(corr_value), 4),
                    "absolute_correlation": round(abs(float(corr_value)), 4),
                    "strength": classify_correlation_strength(abs(corr_value)),
                    "direction": "positive" if corr_value > 0 else "negative"
                })

    # Sort by absolute correlation value
    strong_correlations.sort(key=lambda x: x["absolute_correlation"], reverse=True)

    return strong_correlations


def classify_correlation_strength(abs_corr: float) -> str:
    """
    Classify correlation strength
    상관관계 강도 분류
    """
    if abs_corr >= 0.9:
        return "매우 강함"
    elif abs_corr >= 0.7:
        return "강함"
    elif abs_corr >= 0.5:
        return "보통"
    elif abs_corr >= 0.3:
        return "약함"
    else:
        return "매우 약함"


def calculate_correlation_statistics(corr_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall correlation statistics
    전체 상관관계 통계 계산
    """
    # Get upper triangle correlations (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_correlations = corr_matrix.where(mask).stack().dropna()

    if len(upper_correlations) == 0:
        return {
            "error": "상관관계 통계를 계산할 수 없습니다",
            "total_pairs": 0
        }

    stats_dict = {
        "total_pairs": len(upper_correlations),
        "mean_correlation": float(upper_correlations.mean()),
        "median_correlation": float(upper_correlations.median()),
        "std_correlation": float(upper_correlations.std()),
        "min_correlation": float(upper_correlations.min()),
        "max_correlation": float(upper_correlations.max()),
        "abs_mean_correlation": float(upper_correlations.abs().mean()),
        "abs_median_correlation": float(upper_correlations.abs().median())
    }

    # Distribution of correlation strengths
    strength_distribution = {
        "매우 강함": len(upper_correlations[upper_correlations.abs() >= 0.9]),
        "강함": len(upper_correlations[(upper_correlations.abs() >= 0.7) & (upper_correlations.abs() < 0.9)]),
        "보통": len(upper_correlations[(upper_correlations.abs() >= 0.5) & (upper_correlations.abs() < 0.7)]),
        "약함": len(upper_correlations[(upper_correlations.abs() >= 0.3) & (upper_correlations.abs() < 0.5)]),
        "매우 약함": len(upper_correlations[upper_correlations.abs() < 0.3])
    }

    stats_dict["strength_distribution"] = strength_distribution

    return stats_dict


def generate_correlation_insights(strong_correlations: List[Dict], correlation_stats: Dict, threshold: float) -> List[Dict[str, str]]:
    """
    Generate insights about correlations
    상관관계에 대한 인사이트 생성
    """
    insights = []

    # Overall correlation level
    abs_mean_corr = correlation_stats.get("abs_mean_correlation", 0)
    if abs_mean_corr > 0.5:
        insights.append({
            "type": "높은 상관관계",
            "message": f"변수들 간 전반적으로 높은 상관관계 (평균 절댓값: {abs_mean_corr:.3f})",
            "severity": "정보"
        })
    elif abs_mean_corr < 0.2:
        insights.append({
            "type": "낮은 상관관계",
            "message": f"변수들 간 전반적으로 낮은 상관관계 (평균 절댓값: {abs_mean_corr:.3f})",
            "severity": "정보"
        })

    # Strong correlation warnings
    very_strong = [c for c in strong_correlations if c["absolute_correlation"] >= 0.9]
    if very_strong:
        insights.append({
            "type": "다중공선성 경고",
            "message": f"{len(very_strong)}개의 매우 강한 상관관계 발견. 다중공선성 문제 주의 필요",
            "severity": "경고"
        })

    # Highlight top correlations
    for i, corr in enumerate(strong_correlations[:3]):
        if corr["absolute_correlation"] >= threshold:
            direction = "양의" if corr["direction"] == "positive" else "음의"
            insights.append({
                "type": "주요 상관관계",
                "message": f"'{corr['variable1']}'와 '{corr['variable2']}' 간 {corr['strength']} {direction} 상관관계 (r={corr['correlation']})",
                "severity": "정보"
            })

    # Independence assessment
    total_pairs = correlation_stats.get("total_pairs", 1)
    very_weak_count = correlation_stats.get("strength_distribution", {}).get("매우 약함", 0)
    if very_weak_count / total_pairs > 0.7:
        insights.append({
            "type": "변수 독립성",
            "message": "대부분 변수들이 서로 독립적이며 각각 고유한 정보를 제공합니다",
            "severity": "정보"
        })

    return insights


def generate_recommendations(strong_correlations: List[Dict], method: str, threshold: float) -> List[Dict[str, str]]:
    """
    Generate recommendations based on correlation analysis
    상관관계 분석 결과를 바탕으로 권장사항 생성
    """
    recommendations = []

    # Very strong correlations
    very_strong = [c for c in strong_correlations if c["absolute_correlation"] >= 0.9]
    if very_strong:
        var_pairs = [f"{c['variable1']}-{c['variable2']}" for c in very_strong[:3]]
        recommendations.append({
            "type": "특성 선택",
            "action": "매우 높은 상관관계 변수들 중 일부 제거 고려",
            "reason": "다중공선성 방지 및 모델 성능 향상",
            "variables": ", ".join(var_pairs)
        })

    # Moderate correlations
    moderate = [c for c in strong_correlations if 0.5 <= c["absolute_correlation"] < 0.7]
    if moderate:
        var_pairs = [f"{c['variable1']}-{c['variable2']}" for c in moderate[:3]]
        recommendations.append({
            "type": "특성 공학",
            "action": "중간 강도 상관관계 변수들로 새로운 특성 생성 고려",
            "reason": "변수 간 관계를 더 효과적으로 활용",
            "variables": ", ".join(var_pairs)
        })

    # Negative correlations
    negative_strong = [c for c in strong_correlations if c["correlation"] <= -threshold]
    if negative_strong:
        var_pairs = [f"{c['variable1']}-{c['variable2']}" for c in negative_strong[:3]]
        recommendations.append({
            "type": "역관계 분석",
            "action": "강한 음의 상관관계에 대한 비즈니스 로직 검토",
            "reason": "상반된 트렌드 패턴 이해",
            "variables": ", ".join(var_pairs)
        })

    # Method-specific recommendations
    if method == 'pearson' and strong_correlations:
        recommendations.append({
            "type": "분석 방법",
            "action": "비선형 관계 확인을 위해 Spearman 상관계수도 고려해보세요",
            "reason": "Pearson은 선형관계만 측정하므로 비선형 관계 놓칠 수 있음",
            "variables": "모든 변수"
        })

    return recommendations


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['file_path'])

        # Load data
        df = load_data(options['file_path'])

        # Get basic data info
        data_info = get_data_info(df)

        # Perform correlation analysis
        method = options.get('method', 'pearson')
        threshold = options.get('threshold', 0.7)

        analysis_results = calculate_correlation_analysis(df, method, threshold)

        if "error" in analysis_results:
            error_result = {
                "success": False,
                "error": analysis_results["error"],
                "analysis_type": "correlation_analysis",
                **{k: v for k, v in analysis_results.items() if k != "error"}
            }
            output_results(error_result)
            return

        # Create final result
        result = create_analysis_result(
            analysis_type="correlation_analysis",
            data_info=data_info,
            results=analysis_results,
            summary=f"{method} 상관관계 분석 완료 - {analysis_results['analyzed_variables']}개 변수, 임계값 {threshold}"
        )

        # Output results
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "correlation_analysis"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()