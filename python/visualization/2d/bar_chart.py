#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar Chart Generator
막대 그래프 생성기

이 모듈은 범주형 데이터의 시각화를 위한 막대 그래프를 생성합니다.
주요 기능:
- 세로/가로 막대 그래프
- 그룹별 막대 그래프
- 값 라벨 표시
- 색상 커스터마이징
- 정렬 옵션
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

def create_bar_chart(data: Dict[str, Any], x_column: str, y_column: str,
                    title: str = "", output_path: str = "bar_chart.png",
                    horizontal: bool = False, sort_by: str = "none",
                    color_palette: str = "Set3", show_values: bool = True,
                    figsize: tuple = (10, 6)) -> Dict[str, Any]:
    """
    막대 그래프 생성

    Parameters:
    -----------
    data : Dict[str, Any]
        시각화할 데이터 (JSON 형태)
    x_column : str
        X축 컬럼명 (범주형)
    y_column : str
        Y축 컬럼명 (수치형)
    title : str, default=""
        차트 제목
    output_path : str, default="bar_chart.png"
        출력 파일 경로
    horizontal : bool, default=False
        가로 막대 그래프 여부
    sort_by : str, default="none"
        정렬 방식 ("none", "ascending", "descending", "alphabetical")
    color_palette : str, default="Set3"
        색상 팔레트
    show_values : bool, default=True
        막대 위에 값 표시 여부
    figsize : tuple, default=(10, 6)
        그림 크기

    Returns:
    --------
    Dict[str, Any]
        생성 결과 및 통계 정보
    """

    try:
        # 데이터프레임 생성
        df = pd.DataFrame(data)

        # 컬럼 존재 확인
        if x_column not in df.columns:
            return {
                "success": False,
                "error": f"X축 컬럼 '{x_column}'이 데이터에 없습니다",
                "available_columns": df.columns.tolist()
            }

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

        # 결측값 제거
        df_clean = df[[x_column, y_column]].dropna()

        if len(df_clean) == 0:
            return {
                "success": False,
                "error": "유효한 데이터가 없습니다"
            }

        # 그룹별 집계 (같은 카테고리의 값들 합계)
        df_grouped = df_clean.groupby(x_column)[y_column].sum().reset_index()

        # 정렬
        if sort_by == "ascending":
            df_grouped = df_grouped.sort_values(y_column, ascending=True)
        elif sort_by == "descending":
            df_grouped = df_grouped.sort_values(y_column, ascending=False)
        elif sort_by == "alphabetical":
            df_grouped = df_grouped.sort_values(x_column)

        # 통계 정보 계산
        stats = {
            "total_categories": len(df_grouped),
            "total_value": float(df_grouped[y_column].sum()),
            "mean_value": float(df_grouped[y_column].mean()),
            "max_value": float(df_grouped[y_column].max()),
            "min_value": float(df_grouped[y_column].min()),
            "max_category": df_grouped.loc[df_grouped[y_column].idxmax(), x_column],
            "min_category": df_grouped.loc[df_grouped[y_column].idxmin(), x_column]
        }

        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)

        # 색상 설정
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(df_grouped)))

        if horizontal:
            # 가로 막대 그래프
            bars = ax.barh(df_grouped[x_column], df_grouped[y_column], color=colors)

            # 값 라벨
            if show_values:
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + stats["max_value"] * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.1f}', ha='left', va='center', fontsize=9)

            ax.set_xlabel(y_column, fontsize=12)
            ax.set_ylabel(x_column, fontsize=12)

        else:
            # 세로 막대 그래프
            bars = ax.bar(df_grouped[x_column], df_grouped[y_column], color=colors)

            # 값 라벨
            if show_values:
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + stats["max_value"] * 0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel(y_column, fontsize=12)

            # X축 라벨 회전 (카테고리가 많은 경우)
            if len(df_grouped) > 8:
                plt.xticks(rotation=45, ha='right')

        # 제목 설정
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{x_column}별 {y_column} 분포', fontsize=14, fontweight='bold')

        # 그리드 추가
        ax.grid(True, alpha=0.3)

        # 레이아웃 최적화
        plt.tight_layout()

        # 파일 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 카테고리별 분석
        category_analysis = analyze_categories(df_grouped, x_column, y_column)

        return {
            "success": True,
            "output_path": output_path,
            "chart_type": "bar_chart",
            "orientation": "horizontal" if horizontal else "vertical",
            "categories_analyzed": df_grouped[x_column].tolist(),
            "values": df_grouped[y_column].tolist(),
            "statistics": stats,
            "category_analysis": category_analysis,
            "data_insights": generate_bar_chart_insights(stats, category_analysis)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"막대 그래프 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def analyze_categories(df: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, Any]:
    """카테고리별 상세 분석"""

    total_value = df[y_column].sum()

    analysis = {
        "category_contributions": {},
        "top_3_categories": [],
        "bottom_3_categories": [],
        "value_distribution": {}
    }

    # 카테고리별 기여도
    for _, row in df.iterrows():
        category = row[x_column]
        value = row[y_column]
        percentage = (value / total_value) * 100 if total_value > 0 else 0

        analysis["category_contributions"][str(category)] = {
            "value": float(value),
            "percentage": float(percentage)
        }

    # 상위/하위 카테고리
    df_sorted = df.sort_values(y_column, ascending=False)

    analysis["top_3_categories"] = [
        {
            "category": str(row[x_column]),
            "value": float(row[y_column]),
            "percentage": float((row[y_column] / total_value) * 100) if total_value > 0 else 0
        }
        for _, row in df_sorted.head(3).iterrows()
    ]

    analysis["bottom_3_categories"] = [
        {
            "category": str(row[x_column]),
            "value": float(row[y_column]),
            "percentage": float((row[y_column] / total_value) * 100) if total_value > 0 else 0
        }
        for _, row in df_sorted.tail(3).iterrows()
    ]

    # 값 분포 분석
    values = df[y_column]
    analysis["value_distribution"] = {
        "range": float(values.max() - values.min()),
        "coefficient_of_variation": float(values.std() / values.mean()) if values.mean() != 0 else 0,
        "concentration_index": float(values.max() / values.sum()) if values.sum() > 0 else 0
    }

    return analysis

def generate_bar_chart_insights(stats: Dict[str, Any], category_analysis: Dict[str, Any]) -> List[str]:
    """막대 그래프 인사이트 생성"""

    insights = []

    # 최고/최저 카테고리
    insights.append(f"가장 높은 값을 가진 카테고리는 '{stats['max_category']}'({stats['max_value']:.1f})입니다.")
    insights.append(f"가장 낮은 값을 가진 카테고리는 '{stats['min_category']}'({stats['min_value']:.1f})입니다.")

    # 분포의 균등성
    cv = category_analysis["value_distribution"]["coefficient_of_variation"]
    if cv < 0.3:
        insights.append("카테고리별 값의 분포가 비교적 균등합니다.")
    elif cv > 0.7:
        insights.append("카테고리별 값의 편차가 큽니다.")
    else:
        insights.append("카테고리별 값의 분포가 적당한 편차를 보입니다.")

    # 집중도 분석
    concentration = category_analysis["value_distribution"]["concentration_index"]
    if concentration > 0.5:
        insights.append("전체 값의 상당 부분이 한 카테고리에 집중되어 있습니다.")
    elif concentration < 0.2:
        insights.append("값이 여러 카테고리에 골고루 분산되어 있습니다.")

    # 상위 카테고리 분석
    if len(category_analysis["top_3_categories"]) >= 3:
        top_3_total = sum([cat["percentage"] for cat in category_analysis["top_3_categories"]])
        if top_3_total > 70:
            insights.append("상위 3개 카테고리가 전체의 70% 이상을 차지합니다.")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 막대 그래프 생성
        result = create_bar_chart(**params)

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