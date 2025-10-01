#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box Plot Generator
박스 플롯 생성기

이 모듈은 데이터의 분포와 이상값을 시각화하는 박스 플롯을 생성합니다.
주요 기능:
- 단일/그룹별 박스 플롯
- 이상값 탐지 및 표시
- 통계 정보 제공 (사분위수, IQR 등)
- 분포 비교 분석
- 한국어 해석 및 인사이트
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_box_plot(data: Dict[str, Any], y_column: str, x_column: Optional[str] = None,
                   title: str = "", output_path: str = "box_plot.png",
                   show_outliers: bool = True, show_means: bool = True,
                   palette: str = "Set2", figsize: tuple = (10, 6)) -> Dict[str, Any]:
    """
    박스 플롯 생성

    Parameters:
    -----------
    data : Dict[str, Any]
        시각화할 데이터 (JSON 형태)
    y_column : str
        Y축 컬럼명 (수치형)
    x_column : str, optional
        X축 컬럼명 (범주형, 그룹별 비교용)
    title : str, default=""
        차트 제목
    output_path : str, default="box_plot.png"
        출력 파일 경로
    show_outliers : bool, default=True
        이상값 표시 여부
    show_means : bool, default=True
        평균값 표시 여부
    palette : str, default="Set2"
        색상 팔레트
    figsize : tuple, default=(10, 6)
        그림 크기

    Returns:
    --------
    Dict[str, Any]
        생성 결과 및 통계 분석
    """

    try:
        # 데이터프레임 생성
        df = pd.DataFrame(data)

        # Y축 컬럼 존재 확인
        if y_column not in df.columns:
            return {
                "success": False,
                "error": f"Y축 컬럼 '{y_column}'이 데이터에 없습니다",
                "available_columns": df.columns.tolist()
            }

        # Y축이 수치형인지 확인
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return {
                "success": False,
                "error": f"Y축 컬럼 '{y_column}'이 수치형이 아닙니다",
                "column_dtype": str(df[y_column].dtype)
            }

        # X축 컬럼 확인 (있는 경우)
        if x_column and x_column not in df.columns:
            return {
                "success": False,
                "error": f"X축 컬럼 '{x_column}'이 데이터에 없습니다",
                "available_columns": df.columns.tolist()
            }

        # 결측값 제거
        columns_to_use = [y_column] + ([x_column] if x_column else [])
        df_clean = df[columns_to_use].dropna()

        if len(df_clean) == 0:
            return {
                "success": False,
                "error": "유효한 데이터가 없습니다"
            }

        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)

        # 박스 플롯 생성
        if x_column:
            # 그룹별 박스 플롯
            sns.boxplot(data=df_clean, x=x_column, y=y_column,
                       palette=palette, ax=ax, showfliers=show_outliers)

            # 평균값 표시
            if show_means:
                means = df_clean.groupby(x_column)[y_column].mean()
                for i, (group, mean_val) in enumerate(means.items()):
                    ax.scatter(i, mean_val, color='red', marker='D', s=50, zorder=3)

            # X축 라벨 회전
            if len(df_clean[x_column].unique()) > 8:
                plt.xticks(rotation=45, ha='right')

        else:
            # 단일 박스 플롯
            sns.boxplot(data=df_clean, y=y_column, ax=ax, showfliers=show_outliers)

            # 평균값 표시
            if show_means:
                mean_val = df_clean[y_column].mean()
                ax.scatter(0, mean_val, color='red', marker='D', s=50, zorder=3)

        # 제목 및 라벨 설정
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            if x_column:
                ax.set_title(f'{x_column}별 {y_column} 박스 플롯', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'{y_column} 분포 박스 플롯', fontsize=14, fontweight='bold')

        ax.set_ylabel(y_column, fontsize=12)
        if x_column:
            ax.set_xlabel(x_column, fontsize=12)

        # 그리드 추가
        ax.grid(True, alpha=0.3)

        # 레이아웃 최적화
        plt.tight_layout()

        # 파일 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 통계 분석 수행
        statistical_analysis = perform_box_plot_analysis(df_clean, y_column, x_column)

        return {
            "success": True,
            "output_path": output_path,
            "chart_type": "box_plot",
            "y_variable": y_column,
            "x_variable": x_column,
            "groups_analyzed": df_clean[x_column].unique().tolist() if x_column else None,
            "statistical_analysis": statistical_analysis,
            "data_insights": generate_box_plot_insights(statistical_analysis, y_column, x_column)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"박스 플롯 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def perform_box_plot_analysis(df: pd.DataFrame, y_column: str, x_column: Optional[str]) -> Dict[str, Any]:
    """박스 플롯 통계 분석"""

    analysis = {}

    if x_column:
        # 그룹별 분석
        groups = df[x_column].unique()
        analysis["group_statistics"] = {}

        for group in groups:
            group_data = df[df[x_column] == group][y_column]
            group_stats = calculate_box_stats(group_data)
            analysis["group_statistics"][str(group)] = group_stats

        # 그룹 간 비교
        analysis["group_comparison"] = compare_groups(df, y_column, x_column)

    else:
        # 단일 그룹 분석
        analysis["overall_statistics"] = calculate_box_stats(df[y_column])

    return analysis

def calculate_box_stats(series: pd.Series) -> Dict[str, Any]:
    """박스 플롯 통계 계산"""

    # 기본 통계
    stats = {
        "count": len(series),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max())
    }

    # 사분위수
    quartiles = series.quantile([0.25, 0.5, 0.75])
    stats.update({
        "q1": float(quartiles[0.25]),
        "q2_median": float(quartiles[0.5]),
        "q3": float(quartiles[0.75]),
        "iqr": float(quartiles[0.75] - quartiles[0.25])
    })

    # 이상값 탐지
    iqr = stats["iqr"]
    lower_fence = stats["q1"] - 1.5 * iqr
    upper_fence = stats["q3"] + 1.5 * iqr

    outliers = series[(series < lower_fence) | (series > upper_fence)]
    stats.update({
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "outlier_count": len(outliers),
        "outlier_percentage": float(len(outliers) / len(series) * 100),
        "outliers": outliers.tolist() if len(outliers) <= 20 else outliers.tolist()[:20]
    })

    # 분포 특성
    stats["distribution_properties"] = {
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "range": float(stats["max"] - stats["min"]),
        "coefficient_of_variation": float(stats["std"] / stats["mean"]) if stats["mean"] != 0 else 0
    }

    return stats

def compare_groups(df: pd.DataFrame, y_column: str, x_column: str) -> Dict[str, Any]:
    """그룹 간 비교 분석"""

    groups = df[x_column].unique()
    comparison = {
        "group_count": len(groups),
        "median_comparison": {},
        "variance_comparison": {},
        "outlier_comparison": {}
    }

    group_medians = []
    group_variances = []
    group_outlier_rates = []

    for group in groups:
        group_data = df[df[x_column] == group][y_column]

        median_val = group_data.median()
        variance_val = group_data.var()

        # 이상값 비율
        q1, q3 = group_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_rate = len(group_data[(group_data < lower_fence) | (group_data > upper_fence)]) / len(group_data) * 100

        group_medians.append(median_val)
        group_variances.append(variance_val)
        group_outlier_rates.append(outlier_rate)

        comparison["median_comparison"][str(group)] = float(median_val)
        comparison["variance_comparison"][str(group)] = float(variance_val)
        comparison["outlier_comparison"][str(group)] = float(outlier_rate)

    # 전체 비교 통계
    comparison["summary"] = {
        "median_range": float(max(group_medians) - min(group_medians)),
        "highest_median_group": str(groups[np.argmax(group_medians)]),
        "lowest_median_group": str(groups[np.argmin(group_medians)]),
        "most_variable_group": str(groups[np.argmax(group_variances)]),
        "least_variable_group": str(groups[np.argmin(group_variances)]),
        "highest_outlier_group": str(groups[np.argmax(group_outlier_rates)]),
        "variance_ratio": float(max(group_variances) / min(group_variances)) if min(group_variances) > 0 else float('inf')
    }

    return comparison

def generate_box_plot_insights(analysis: Dict[str, Any], y_column: str, x_column: Optional[str]) -> List[str]:
    """박스 플롯 인사이트 생성"""

    insights = []

    if x_column and "group_comparison" in analysis:
        # 그룹별 분석 인사이트
        comparison = analysis["group_comparison"]
        summary = comparison["summary"]

        insights.append(f"중위값이 가장 높은 그룹: {summary['highest_median_group']}")
        insights.append(f"중위값이 가장 낮은 그룹: {summary['lowest_median_group']}")

        if summary["variance_ratio"] > 4:
            insights.append(f"그룹 간 분산 차이가 큽니다 (비율: {summary['variance_ratio']:.1f}:1)")
        elif summary["variance_ratio"] < 2:
            insights.append("그룹 간 분산이 비슷합니다")

        # 이상값 분석
        outlier_rates = list(comparison["outlier_comparison"].values())
        max_outlier_rate = max(outlier_rates)
        if max_outlier_rate > 10:
            insights.append(f"'{summary['highest_outlier_group']}' 그룹에서 이상값이 많이 발견됩니다 ({max_outlier_rate:.1f}%)")

    else:
        # 단일 그룹 분석 인사이트
        overall_stats = analysis["overall_statistics"]

        if overall_stats["outlier_percentage"] > 5:
            insights.append(f"이상값이 {overall_stats['outlier_count']}개 ({overall_stats['outlier_percentage']:.1f}%) 발견되었습니다")

        # 분포 특성
        skewness = overall_stats["distribution_properties"]["skewness"]
        if abs(skewness) > 1:
            direction = "오른쪽" if skewness > 0 else "왼쪽"
            insights.append(f"데이터가 {direction}으로 치우쳐 있습니다 (왜도: {skewness:.2f})")
        else:
            insights.append("데이터가 대칭적으로 분포되어 있습니다")

        # 변동성
        cv = overall_stats["distribution_properties"]["coefficient_of_variation"]
        if cv > 1:
            insights.append("데이터의 변동성이 큽니다")
        elif cv < 0.3:
            insights.append("데이터의 변동성이 낮습니다")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 박스 플롯 생성
        result = create_box_plot(**params)

        # JSON으로 결과 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()