#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histogram Generator
히스토그램 생성기

이 모듈은 데이터의 분포를 시각화하는 히스토그램을 생성합니다.
주요 기능:
- 단변량 분포 시각화
- 통계 정보 표시 (평균, 중위값, 표준편차)
- 정규분포 곡선 오버레이
- 다양한 구간(bin) 설정 옵션
- 한국어 라벨 및 주석
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_histogram(data: Dict[str, Any], column: str, bins: int = 30,
                    title: str = "", output_path: str = "histogram.png",
                    show_stats: bool = True, show_normal_curve: bool = True,
                    figsize: tuple = (10, 6)) -> Dict[str, Any]:
    """
    히스토그램 생성

    Parameters:
    -----------
    data : Dict[str, Any]
        시각화할 데이터 (JSON 형태)
    column : str
        히스토그램을 생성할 컬럼명
    bins : int, default=30
        히스토그램 구간 수
    title : str, default=""
        차트 제목
    output_path : str, default="histogram.png"
        출력 파일 경로
    show_stats : bool, default=True
        통계 정보 표시 여부
    show_normal_curve : bool, default=True
        정규분포 곡선 표시 여부
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

        if column not in df.columns:
            return {
                "success": False,
                "error": f"컬럼 '{column}'이 데이터에 없습니다",
                "available_columns": df.columns.tolist()
            }

        # 수치형 데이터 확인
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "success": False,
                "error": f"컬럼 '{column}'이 수치형이 아닙니다",
                "column_dtype": str(df[column].dtype)
            }

        # 결측값 제거
        series_data = df[column].dropna()

        if len(series_data) == 0:
            return {
                "success": False,
                "error": f"컬럼 '{column}'에 유효한 데이터가 없습니다"
            }

        # 통계 정보 계산
        stats = {
            "count": len(series_data),
            "mean": float(series_data.mean()),
            "median": float(series_data.median()),
            "std": float(series_data.std()),
            "min": float(series_data.min()),
            "max": float(series_data.max()),
            "q25": float(series_data.quantile(0.25)),
            "q75": float(series_data.quantile(0.75)),
            "skewness": float(series_data.skew()),
            "kurtosis": float(series_data.kurtosis())
        }

        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)

        # 히스토그램 생성
        n, bins_edges, patches = ax.hist(
            series_data, bins=bins, alpha=0.7, color='skyblue',
            edgecolor='black', density=True
        )

        # 정규분포 곡선 오버레이
        if show_normal_curve:
            x = np.linspace(series_data.min(), series_data.max(), 100)
            normal_curve = (1 / (stats["std"] * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x - stats["mean"]) / stats["std"]) ** 2)
            ax.plot(x, normal_curve, 'r-', linewidth=2, label='정규분포 곡선')

        # 통계선 추가
        ax.axvline(stats["mean"], color='red', linestyle='--', linewidth=2, label=f'평균: {stats["mean"]:.2f}')
        ax.axvline(stats["median"], color='green', linestyle='--', linewidth=2, label=f'중위값: {stats["median"]:.2f}')

        # 제목 및 라벨 설정
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{column} 분포 히스토그램', fontsize=14, fontweight='bold')

        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('밀도 (Density)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 통계 정보 텍스트 박스
        if show_stats:
            stats_text = f"""통계 정보:
평균: {stats["mean"]:.2f}
중위값: {stats["median"]:.2f}
표준편차: {stats["std"]:.2f}
왜도: {stats["skewness"]:.2f}
첨도: {stats["kurtosis"]:.2f}"""

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)

        # 레이아웃 최적화
        plt.tight_layout()

        # 파일 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "success": True,
            "output_path": output_path,
            "chart_type": "histogram",
            "column_analyzed": column,
            "bins_used": bins,
            "statistics": stats,
            "distribution_analysis": {
                "is_normal": abs(stats["skewness"]) < 0.5,
                "symmetry": "대칭" if abs(stats["skewness"]) < 0.5 else
                          ("왼쪽 치우침" if stats["skewness"] < -0.5 else "오른쪽 치우침"),
                "tail_heaviness": "정상" if abs(stats["kurtosis"]) < 3 else
                                ("가벼운 꼬리" if stats["kurtosis"] < -3 else "무거운 꼬리")
            },
            "interpretation": generate_histogram_interpretation(stats)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"히스토그램 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def generate_histogram_interpretation(stats: Dict[str, Any]) -> Dict[str, str]:
    """히스토그램 해석 생성"""

    interpretations = []

    # 분포의 중심 경향
    if abs(stats["mean"] - stats["median"]) < stats["std"] * 0.1:
        interpretations.append("평균과 중위값이 유사하여 대칭적인 분포를 보입니다.")
    elif stats["mean"] > stats["median"]:
        interpretations.append("평균이 중위값보다 커서 오른쪽 꼬리가 긴 분포입니다.")
    else:
        interpretations.append("평균이 중위값보다 작아서 왼쪽 꼬리가 긴 분포입니다.")

    # 변동성
    cv = stats["std"] / stats["mean"] if stats["mean"] != 0 else float('inf')
    if cv < 0.1:
        interpretations.append("변동성이 매우 낮은 안정적인 분포입니다.")
    elif cv < 0.3:
        interpretations.append("변동성이 낮은 편입니다.")
    elif cv < 0.7:
        interpretations.append("중간 정도의 변동성을 보입니다.")
    else:
        interpretations.append("변동성이 높은 분포입니다.")

    # 정규성
    if abs(stats["skewness"]) < 0.5 and abs(stats["kurtosis"]) < 3:
        interpretations.append("정규분포에 가까운 분포 특성을 보입니다.")
    else:
        interpretations.append("정규분포와 다른 분포 특성을 보입니다.")

    # 이상값 가능성
    iqr = stats["q75"] - stats["q25"]
    lower_fence = stats["q25"] - 1.5 * iqr
    upper_fence = stats["q75"] + 1.5 * iqr

    if stats["min"] < lower_fence or stats["max"] > upper_fence:
        interpretations.append("이상값이 존재할 가능성이 있습니다.")

    return {
        "summary": " ".join(interpretations),
        "distribution_type": "정규분포 유사" if abs(stats["skewness"]) < 0.5 else
                           ("왼쪽 치우침" if stats["skewness"] < 0 else "오른쪽 치우침"),
        "variability": "낮음" if cv < 0.3 else ("보통" if cv < 0.7 else "높음"),
        "outlier_risk": "있음" if (stats["min"] < lower_fence or stats["max"] > upper_fence) else "없음"
    }

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 히스토그램 생성
        result = create_histogram(**params)

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