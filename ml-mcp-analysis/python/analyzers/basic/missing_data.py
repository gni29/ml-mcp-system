#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missing Data Analysis for Lightweight Analysis MCP
경량 분석 MCP용 결측치 분석 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
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
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def analyze_missing_data_comprehensive(df: pd.DataFrame, suggest_imputation: bool = True) -> Dict[str, Any]:
    """
    Comprehensive missing data analysis
    포괄적인 결측치 분석
    """
    results = {}

    # Overall missing data summary
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_percentage = (total_missing / total_cells) * 100

    results["missing_summary"] = {
        "total_missing": int(total_missing),
        "total_cells": int(total_cells),
        "missing_percentage": round(missing_percentage, 2),
        "complete_rows": int(len(df) - df.isnull().any(axis=1).sum()),
        "complete_row_percentage": round(((len(df) - df.isnull().any(axis=1).sum()) / len(df)) * 100, 2)
    }

    # Column-wise missing analysis
    columns_with_missing = []
    column_analysis = {}

    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100

        column_info = {
            "missing_count": int(missing_count),
            "missing_percentage": round(missing_pct, 2),
            "data_type": str(df[column].dtype),
            "non_null_count": int(df[column].count()),
            "unique_values": int(df[column].nunique()),
            "missing_severity": classify_missing_severity(missing_pct)
        }

        column_analysis[column] = column_info

        if missing_count > 0:
            columns_with_missing.append(column)

    results["column_analysis"] = column_analysis
    results["columns_with_missing"] = columns_with_missing

    # Missing data patterns
    missing_patterns = analyze_missing_patterns(df)
    results["missing_patterns"] = missing_patterns

    # Missing data insights
    insights = generate_missing_insights(column_analysis, results["missing_summary"])
    results["insights"] = insights

    # Imputation suggestions if requested
    if suggest_imputation:
        imputation_suggestions = suggest_imputation_methods(df, column_analysis)
        results["imputation_suggestions"] = imputation_suggestions

    # Quality assessment
    quality_assessment = assess_data_quality(results["missing_summary"], column_analysis)
    results["quality_assessment"] = quality_assessment

    return results


def classify_missing_severity(missing_percentage: float) -> str:
    """
    Classify missing data severity
    결측치 심각도 분류
    """
    if missing_percentage == 0:
        return "없음"
    elif missing_percentage < 5:
        return "경미함"
    elif missing_percentage < 15:
        return "보통"
    elif missing_percentage < 30:
        return "심각함"
    else:
        return "매우 심각함"


def analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing data patterns
    결측치 패턴 분석
    """
    # Get missing patterns
    missing_pattern_counts = df.isnull().value_counts()

    patterns = {}
    for i, (pattern, count) in enumerate(missing_pattern_counts.head(10).items()):
        pattern_dict = {}
        for j, col in enumerate(df.columns):
            pattern_dict[col] = bool(pattern[j])

        patterns[f"pattern_{i+1}"] = {
            "missing_columns": pattern_dict,
            "count": int(count),
            "percentage": round((count / len(df)) * 100, 2),
            "description": describe_pattern(pattern_dict)
        }

    # Pattern analysis summary
    total_patterns = len(missing_pattern_counts)
    complete_cases = int(missing_pattern_counts.iloc[0]) if not df.isnull().any().any() or (len(missing_pattern_counts) > 0 and not missing_pattern_counts.index[0].any()) else 0

    pattern_summary = {
        "total_unique_patterns": total_patterns,
        "complete_cases": complete_cases,
        "incomplete_cases": len(df) - complete_cases,
        "most_common_pattern": patterns.get("pattern_1", {}).get("description", "알 수 없음")
    }

    return {
        "patterns": patterns,
        "summary": pattern_summary
    }


def describe_pattern(pattern_dict: Dict[str, bool]) -> str:
    """
    Describe a missing pattern
    결측치 패턴 설명
    """
    missing_cols = [col for col, is_missing in pattern_dict.items() if is_missing]

    if not missing_cols:
        return "완전한 데이터"
    elif len(missing_cols) == 1:
        return f"'{missing_cols[0]}' 컬럼만 결측"
    elif len(missing_cols) == len(pattern_dict):
        return "모든 컬럼 결측"
    else:
        return f"{len(missing_cols)}개 컬럼 결측: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}"


def generate_missing_insights(column_analysis: Dict[str, Any], missing_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate insights about missing data
    결측치에 대한 인사이트 생성
    """
    insights = []

    # Overall data quality insight
    overall_missing_pct = missing_summary["missing_percentage"]
    if overall_missing_pct < 5:
        insights.append({
            "type": "데이터 품질",
            "message": f"전체적으로 우수한 데이터 품질 (결측률: {overall_missing_pct:.1f}%)",
            "severity": "positive"
        })
    elif overall_missing_pct < 15:
        insights.append({
            "type": "데이터 품질",
            "message": f"양호한 데이터 품질 (결측률: {overall_missing_pct:.1f}%)",
            "severity": "info"
        })
    else:
        insights.append({
            "type": "데이터 품질",
            "message": f"데이터 품질 개선 필요 (결측률: {overall_missing_pct:.1f}%)",
            "severity": "warning"
        })

    # High missing columns
    high_missing_cols = [col for col, info in column_analysis.items()
                        if info["missing_percentage"] > 30]
    if high_missing_cols:
        insights.append({
            "type": "높은 결측률 컬럼",
            "message": f"{len(high_missing_cols)}개 컬럼의 결측률이 30% 이상: {', '.join(high_missing_cols[:3])}",
            "severity": "warning"
        })

    # Complete columns
    complete_cols = [col for col, info in column_analysis.items()
                    if info["missing_percentage"] == 0]
    if complete_cols:
        insights.append({
            "type": "완전한 컬럼",
            "message": f"{len(complete_cols)}개 컬럼에 결측치 없음 (전체 {len(column_analysis)}개 중)",
            "severity": "positive"
        })

    # Complete rows insight
    complete_row_pct = missing_summary["complete_row_percentage"]
    if complete_row_pct > 80:
        insights.append({
            "type": "완전한 행",
            "message": f"행의 {complete_row_pct:.1f}%가 완전한 데이터 보유",
            "severity": "positive"
        })
    elif complete_row_pct < 50:
        insights.append({
            "type": "불완전한 행",
            "message": f"행의 {100-complete_row_pct:.1f}%에 결측치 존재",
            "severity": "warning"
        })

    return insights


def suggest_imputation_methods(df: pd.DataFrame, column_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest appropriate imputation methods for each column
    각 컬럼에 대한 적절한 대체 방법 제안
    """
    suggestions = {}

    for column, analysis in column_analysis.items():
        if analysis["missing_count"] == 0:
            continue

        missing_pct = analysis["missing_percentage"]
        data_type = analysis["data_type"]

        column_suggestions = {
            "missing_percentage": missing_pct,
            "severity": analysis["missing_severity"],
            "recommended_methods": [],
            "considerations": []
        }

        # Determine recommendations based on missing percentage
        if missing_pct > 50:
            column_suggestions["recommended_methods"].append({
                "method": "컬럼 제거",
                "priority": "높음",
                "description": "결측률이 50% 이상으로 정보가 제한적",
                "pros": "분석 복잡도 감소",
                "cons": "잠재적 정보 손실"
            })
            column_suggestions["considerations"].append("컬럼 제거 후 도메인 전문가와 상의 권장")

        elif missing_pct > 20:
            # High missing - predictive imputation
            if pd.api.types.is_numeric_dtype(df[column]):
                column_suggestions["recommended_methods"].append({
                    "method": "예측 모델 기반 대체",
                    "priority": "높음",
                    "description": "다른 변수들을 이용한 회귀/분류 모델로 예측",
                    "pros": "정확한 대체값 추정",
                    "cons": "계산 복잡도 높음"
                })

            column_suggestions["recommended_methods"].append({
                "method": "다중 대체법 (Multiple Imputation)",
                "priority": "높음",
                "description": "여러 번의 대체를 통한 불확실성 고려",
                "pros": "통계적 불확실성 반영",
                "cons": "구현 복잡도 높음"
            })

        else:
            # Low to moderate missing
            if pd.api.types.is_numeric_dtype(df[column]):
                # Numeric column suggestions
                column_suggestions["recommended_methods"].extend([
                    {
                        "method": "중위값 대체",
                        "priority": "높음",
                        "description": "중위값으로 결측치 대체",
                        "pros": "이상치에 강함",
                        "cons": "분포 왜곡 가능"
                    },
                    {
                        "method": "평균값 대체",
                        "priority": "보통",
                        "description": "평균값으로 결측치 대체",
                        "pros": "구현 간단",
                        "cons": "이상치에 민감"
                    }
                ])

                # Check for correlation with other variables
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    try:
                        corr_matrix = df[numeric_cols].corr()
                        if column in corr_matrix.columns:
                            high_corr_cols = []
                            for other_col in corr_matrix.columns:
                                if other_col != column and abs(corr_matrix.loc[column, other_col]) > 0.7:
                                    high_corr_cols.append(other_col)

                            if high_corr_cols:
                                column_suggestions["recommended_methods"].append({
                                    "method": "상관관계 기반 회귀 대체",
                                    "priority": "높음",
                                    "description": f"높은 상관관계 변수 활용: {', '.join(high_corr_cols[:2])}",
                                    "pros": "정확한 예측",
                                    "cons": "변수 간 의존성 증가"
                                })
                    except:
                        pass

            else:
                # Categorical column suggestions
                column_suggestions["recommended_methods"].extend([
                    {
                        "method": "최빈값 대체",
                        "priority": "높음",
                        "description": "가장 빈번한 범주로 대체",
                        "pros": "단순하고 직관적",
                        "cons": "편향 증가 가능"
                    },
                    {
                        "method": "별도 범주 생성",
                        "priority": "보통",
                        "description": "'Missing' 또는 'Unknown' 범주 생성",
                        "pros": "결측 패턴 정보 보존",
                        "cons": "새로운 범주 추가"
                    }
                ])

        # Add general considerations
        if missing_pct < 10:
            column_suggestions["considerations"].append("결측률이 낮아 행 제거도 고려 가능")

        if analysis["unique_values"] / (len(df) - analysis["missing_count"]) > 0.8:
            column_suggestions["considerations"].append("고유값 비율이 높아 범주형 처리 고려")

        suggestions[column] = column_suggestions

    return suggestions


def assess_data_quality(missing_summary: Dict[str, Any], column_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess overall data quality based on missing patterns
    결측 패턴을 바탕으로 전체 데이터 품질 평가
    """
    overall_missing_pct = missing_summary["missing_percentage"]
    complete_row_pct = missing_summary["complete_row_percentage"]

    # Calculate quality scores
    completeness_score = max(0, 100 - overall_missing_pct)

    # Count columns by severity
    severity_counts = {}
    for col, analysis in column_analysis.items():
        severity = analysis["missing_severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Overall quality grade
    if overall_missing_pct < 5 and complete_row_pct > 80:
        quality_grade = "우수"
        recommendations = ["현재 데이터 품질이 우수하여 추가적인 전처리가 최소화됨"]
    elif overall_missing_pct < 15 and complete_row_pct > 60:
        quality_grade = "양호"
        recommendations = ["일부 결측치 처리 후 분석 진행 가능"]
    elif overall_missing_pct < 30:
        quality_grade = "보통"
        recommendations = ["체계적인 결측치 처리 전략 수립 필요", "도메인 전문가와 결측 원인 검토"]
    else:
        quality_grade = "개선 필요"
        recommendations = ["대규모 결측치 처리 계획 수립", "데이터 수집 프로세스 점검", "추가 데이터 확보 고려"]

    return {
        "overall_grade": quality_grade,
        "completeness_score": round(completeness_score, 1),
        "severity_distribution": severity_counts,
        "usable_data_percentage": round(complete_row_pct, 1),
        "recommendations": recommendations,
        "actionable_insights": generate_actionable_insights(overall_missing_pct, severity_counts)
    }


def generate_actionable_insights(overall_missing_pct: float, severity_counts: Dict[str, int]) -> List[str]:
    """
    Generate actionable insights for data quality improvement
    데이터 품질 개선을 위한 실행 가능한 인사이트 생성
    """
    insights = []

    if severity_counts.get("매우 심각함", 0) > 0:
        insights.append(f"{severity_counts['매우 심각함']}개 컬럼의 결측률 30% 초과 - 컬럼 제거 또는 추가 데이터 수집 고려")

    if severity_counts.get("심각함", 0) > 0:
        insights.append(f"{severity_counts['심각함']}개 컬럼의 결측률 15-30% - 고급 대체 방법 적용 권장")

    if overall_missing_pct > 20:
        insights.append("전체 결측률이 20% 초과 - 데이터 수집 프로세스 검토 필요")

    if severity_counts.get("없음", 0) > len(severity_counts) * 0.7:
        insights.append("대부분 컬럼이 완전 - 우수한 데이터 수집 프로세스")

    return insights


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

        # Perform missing data analysis
        suggest_imputation = options.get('suggest_imputation', True)
        analysis_results = analyze_missing_data_comprehensive(df, suggest_imputation)

        # Create final result
        result = create_analysis_result(
            analysis_type="missing_data_analysis",
            data_info=data_info,
            results=analysis_results,
            summary=f"결측치 분석 완료 - 전체 결측률: {analysis_results['missing_summary']['missing_percentage']:.1f}%"
        )

        # Output results
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "missing_data_analysis"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()