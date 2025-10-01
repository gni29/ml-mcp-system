#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missing Data Analysis Module
결측치 분석 및 처리 모듈

이 모듈은 데이터셋의 결측치를 종합적으로 분석하고 처리 방안을 제시합니다.
주요 기능:
- 전체 및 컬럼별 결측치 통계 분석
- 결측치 패턴 분석 및 시각화
- 데이터 타입별 최적 대체방법 제안
- 데이터 품질 평가 및 개선 권고사항 제공
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

def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    결측치 종합 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임

    Returns:
    --------
    Dict[str, Any]
        결측치 분석 결과를 포함한 딕셔너리
        - missing_overview: 전체 결측치 요약 통계
        - missing_by_column: 컬럼별 세부 결측치 정보
        - missing_patterns: 결측치 패턴 분석
        - recommendations: 데이터 처리 권고사항
    """
    results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_overview": {},
        "missing_patterns": {},
        "recommendations": []
    }

    # 전체 결측치 통계 계산
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_percentage = (total_missing / total_cells) * 100

    results["missing_overview"] = {
        "total_missing_values": int(total_missing),
        "total_cells": int(total_cells),
        "missing_percentage": round(missing_percentage, 2)
    }

    # 컬럼별 결측치 분석
    missing_by_column = {}
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100

        missing_by_column[column] = {
            "missing_count": int(missing_count),
            "missing_percentage": round(missing_pct, 2),
            "data_type": str(df[column].dtype),
            "non_null_count": int(df[column].count())
        }

    results["missing_by_column"] = missing_by_column

    # 결측치 패턴 분석
    missing_patterns = df.isnull().value_counts()
    pattern_analysis = {}

    for i, (pattern, count) in enumerate(missing_patterns.head(10).items()):
        pattern_dict = {}
        for j, col in enumerate(df.columns):
            pattern_dict[col] = bool(pattern[j])

        pattern_analysis[f"pattern_{i+1}"] = {
            "columns_missing": pattern_dict,
            "count": int(count),
            "percentage": round((count / len(df)) * 100, 2)
        }

    results["missing_patterns"] = pattern_analysis

    # 추천사항 생성
    recommendations = []

    # 높은 결측치 비율 컬럼
    high_missing_cols = [col for col, info in missing_by_column.items()
                        if info["missing_percentage"] > 50]
    if high_missing_cols:
        recommendations.append({
            "type": "high_missing_columns",
            "description": f"결측치 비율이 50% 이상인 컬럼들: {', '.join(high_missing_cols)}",
            "action": "컬럼 제거 또는 대체 데이터 수집 고려"
        })

    # 중간 수준 결측치 컬럼
    medium_missing_cols = [col for col, info in missing_by_column.items()
                          if 10 <= info["missing_percentage"] <= 50]
    if medium_missing_cols:
        recommendations.append({
            "type": "medium_missing_columns",
            "description": f"결측치 비율이 10-50%인 컬럼들: {', '.join(medium_missing_cols)}",
            "action": "적절한 대체방법 적용 (평균, 중위값, 최빈값, 예측 모델 등)"
        })

    # 낮은 수준 결측치 컬럼
    low_missing_cols = [col for col, info in missing_by_column.items()
                       if 0 < info["missing_percentage"] < 10]
    if low_missing_cols:
        recommendations.append({
            "type": "low_missing_columns",
            "description": f"결측치 비율이 10% 미만인 컬럼들: {', '.join(low_missing_cols)}",
            "action": "행 제거 또는 간단한 대체방법 적용"
        })

    # 완전한 데이터
    complete_cols = [col for col, info in missing_by_column.items()
                    if info["missing_percentage"] == 0]
    if complete_cols:
        recommendations.append({
            "type": "complete_columns",
            "description": f"결측치가 없는 컬럼들: {', '.join(complete_cols)}",
            "action": "분석에 안전하게 사용 가능"
        })

    results["recommendations"] = recommendations

    return results

def suggest_imputation_methods(df: pd.DataFrame) -> Dict[str, Any]:
    """
    컬럼별 최적 결측치 대체방법 제안

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임

    Returns:
    --------
    Dict[str, Any]
        각 컬럼에 대한 최적 대체방법 제안
        - 데이터 타입별 맞춤형 대체 방법
        - 각 방법의 장단점 분석
        - 상관관계 기반 고급 대체방법 제안
    """
    suggestions = {}

    for column in df.columns:
        if df[column].isnull().sum() == 0:
            continue

        col_info = {
            "data_type": str(df[column].dtype),
            "missing_count": int(df[column].isnull().sum()),
            "missing_percentage": round((df[column].isnull().sum() / len(df)) * 100, 2),
            "suggested_methods": []
        }

        # 숫자형 데이터
        if pd.api.types.is_numeric_dtype(df[column]):
            col_info["suggested_methods"] = [
                {
                    "method": "mean_imputation",
                    "description": "평균값으로 대체",
                    "pros": "간단하고 빠름",
                    "cons": "분포가 왜곡될 수 있음"
                },
                {
                    "method": "median_imputation",
                    "description": "중위값으로 대체",
                    "pros": "이상치에 강함",
                    "cons": "정보 손실"
                },
                {
                    "method": "mode_imputation",
                    "description": "최빈값으로 대체",
                    "pros": "범주형에 적합",
                    "cons": "연속형에는 부적절할 수 있음"
                }
            ]

            # 상관관계가 높은 다른 컬럼이 있는지 확인
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_cols = []
                if column in corr_matrix.columns:
                    for other_col in corr_matrix.columns:
                        if other_col != column and abs(corr_matrix.loc[column, other_col]) > 0.7:
                            high_corr_cols.append(other_col)

                if high_corr_cols:
                    col_info["suggested_methods"].append({
                        "method": "regression_imputation",
                        "description": f"회귀 모델 사용 (상관관계 높은 컬럼: {', '.join(high_corr_cols)})",
                        "pros": "정확한 예측값",
                        "cons": "계산 복잡도 높음"
                    })

        # 범주형 데이터
        else:
            col_info["suggested_methods"] = [
                {
                    "method": "mode_imputation",
                    "description": "최빈값으로 대체",
                    "pros": "가장 일반적인 값 사용",
                    "cons": "편향 증가 가능"
                },
                {
                    "method": "constant_imputation",
                    "description": "'Unknown' 또는 'Missing' 카테고리 생성",
                    "pros": "결측치 패턴 보존",
                    "cons": "새로운 카테고리 추가"
                }
            ]

        suggestions[column] = col_info

    return suggestions

def main():
    """
    메인 실행 함수 - 결측치 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 결측치 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, analysis_options

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 요약 및 권고사항
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

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 결측치 분석 수행
        missing_analysis = analyze_missing_data(df)

        # 대체방법 제안
        imputation_suggestions = suggest_imputation_methods(df)

        # 분석 결과 통합
        analysis_results = {
            "missing_data_analysis": missing_analysis,
            "imputation_suggestions": imputation_suggestions,
            "data_quality_assessment": {
                "total_columns_with_missing": len([col for col in df.columns if df[col].isnull().sum() > 0]),
                "columns_complete": len([col for col in df.columns if df[col].isnull().sum() == 0]),
                "overall_missing_percentage": missing_analysis["missing_overview"]["missing_percentage"],
                "data_quality_grade": "우수" if missing_analysis["missing_overview"]["missing_percentage"] < 5 else
                                    "양호" if missing_analysis["missing_overview"]["missing_percentage"] < 20 else "주의"
            }
        }

        # 요약 생성
        missing_cols_count = len([col for col in df.columns if df[col].isnull().sum() > 0])
        summary = f"결측치 분석 완료 - {missing_cols_count}개 컬럼에서 결측치 발견, 전체 {missing_analysis['missing_overview']['missing_percentage']:.1f}% 결측"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="missing_data_analysis",
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
            "analysis_type": "missing_data_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()