#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance Visualization
특성 중요도 시각화

이 모듈은 머신러닝 모델의 특성 중요도를 시각화합니다.
주요 기능:
- 수평/수직 막대 그래프로 특성 중요도 표시
- 상위 K개 특성 선택적 표시
- 중요도 값 및 비율 계산
- 특성 그룹별 분석
- 한국어 해석 및 인사이트
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_feature_importance_plot(feature_names: List[str], importance_values: List[float],
                                  top_k: int = 15, title: str = "Feature Importance",
                                  output_path: str = "feature_importance.png",
                                  horizontal: bool = True, figsize: tuple = (10, 8),
                                  color_scheme: str = "importance_gradient") -> Dict[str, Any]:
    """
    특성 중요도 플롯 생성

    Parameters:
    -----------
    feature_names : List[str]
        특성 이름 목록
    importance_values : List[float]
        특성 중요도 값
    top_k : int, default=15
        표시할 상위 특성 개수
    title : str, default="Feature Importance"
        차트 제목
    output_path : str, default="feature_importance.png"
        출력 파일 경로
    horizontal : bool, default=True
        수평 막대 그래프 여부
    figsize : tuple, default=(10, 8)
        그림 크기
    color_scheme : str, default="importance_gradient"
        색상 스키마

    Returns:
    --------
    Dict[str, Any]
        특성 중요도 분석 결과
    """

    try:
        # 입력 데이터 검증
        if len(feature_names) != len(importance_values):
            return {
                "success": False,
                "error": "특성 이름과 중요도 값의 개수가 일치하지 않습니다",
                "feature_names_count": len(feature_names),
                "importance_values_count": len(importance_values)
            }

        if len(feature_names) == 0:
            return {
                "success": False,
                "error": "입력 데이터가 비어있습니다"
            }

        # 데이터프레임 생성 및 정렬
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        })

        # 중요도의 절댓값으로 정렬
        df['abs_importance'] = df['importance'].abs()
        df = df.sort_values('abs_importance', ascending=True)

        # 상위 K개 특성 선택
        if top_k < len(df):
            df = df.tail(top_k)

        # 색상 설정
        if color_scheme == "importance_gradient":
            # 중요도에 따른 그라디언트
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df)))
        elif color_scheme == "positive_negative":
            # 양수/음수에 따른 색상
            colors = ['red' if x < 0 else 'green' for x in df['importance']]
        else:
            # 기본 색상
            colors = plt.cm.Set3(np.linspace(0, 1, len(df)))

        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)

        if horizontal:
            # 수평 막대 그래프
            bars = ax.barh(range(len(df)), df['importance'], color=colors)

            # 특성 이름 설정
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'], fontsize=10)

            # 값 라벨 추가
            for i, (bar, value) in enumerate(zip(bars, df['importance'])):
                width = bar.get_width()
                ax.text(width + (0.01 * abs(df['importance'].max())),
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.4f}', ha='left', va='center', fontsize=9)

            ax.set_xlabel('중요도 (Importance)', fontsize=12)
            ax.set_ylabel('특성 (Features)', fontsize=12)

        else:
            # 수직 막대 그래프
            bars = ax.bar(range(len(df)), df['importance'], color=colors)

            # 특성 이름 설정 (회전)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=10)

            # 값 라벨 추가
            for i, (bar, value) in enumerate(zip(bars, df['importance'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2,
                       height + (0.01 * abs(df['importance'].max())),
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9, rotation=0)

            ax.set_ylabel('중요도 (Importance)', fontsize=12)
            ax.set_xlabel('특성 (Features)', fontsize=12)

        # 제목 설정
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 0 기준선 추가
        if horizontal:
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        else:
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)

        # 그리드 추가
        ax.grid(True, alpha=0.3)

        # 레이아웃 최적화
        plt.tight_layout()

        # 파일 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 특성 중요도 분석
        importance_analysis = analyze_feature_importance(feature_names, importance_values, top_k)

        return {
            "success": True,
            "output_path": output_path,
            "chart_type": "feature_importance",
            "total_features": len(feature_names),
            "features_displayed": len(df),
            "top_features": df[['feature', 'importance']].to_dict('records'),
            "importance_analysis": importance_analysis,
            "insights": generate_feature_importance_insights(importance_analysis)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"특성 중요도 플롯 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def analyze_feature_importance(feature_names: List[str], importance_values: List[float], top_k: int) -> Dict[str, Any]:
    """특성 중요도 분석"""

    importance_array = np.array(importance_values)

    analysis = {
        "basic_statistics": {
            "total_features": len(feature_names),
            "mean_importance": float(np.mean(importance_array)),
            "std_importance": float(np.std(importance_array)),
            "max_importance": float(np.max(importance_array)),
            "min_importance": float(np.min(importance_array)),
            "sum_absolute_importance": float(np.sum(np.abs(importance_array))),
            "positive_features": int(np.sum(importance_array > 0)),
            "negative_features": int(np.sum(importance_array < 0)),
            "zero_features": int(np.sum(importance_array == 0))
        },
        "top_features": {},
        "importance_distribution": {},
        "feature_groups": {}
    }

    # 정렬된 인덱스
    sorted_indices = np.argsort(np.abs(importance_array))[::-1]

    # 상위 특성 분석
    top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
    analysis["top_features"] = {
        "features": [feature_names[i] for i in top_indices],
        "importance_values": [float(importance_array[i]) for i in top_indices],
        "absolute_values": [float(abs(importance_array[i])) for i in top_indices],
        "cumulative_importance": []
    }

    # 누적 중요도 계산
    total_abs_importance = analysis["basic_statistics"]["sum_absolute_importance"]
    cumulative = 0
    for i in top_indices:
        cumulative += abs(importance_array[i])
        cumulative_percent = (cumulative / total_abs_importance * 100) if total_abs_importance > 0 else 0
        analysis["top_features"]["cumulative_importance"].append(float(cumulative_percent))

    # 중요도 분포 분석
    abs_importance = np.abs(importance_array)
    percentiles = [10, 25, 50, 75, 90]
    analysis["importance_distribution"] = {
        f"percentile_{p}": float(np.percentile(abs_importance, p)) for p in percentiles
    }

    # 고중요도 vs 저중요도 특성
    high_threshold = np.percentile(abs_importance, 75)
    low_threshold = np.percentile(abs_importance, 25)

    high_importance_indices = np.where(abs_importance >= high_threshold)[0]
    low_importance_indices = np.where(abs_importance <= low_threshold)[0]

    analysis["feature_groups"] = {
        "high_importance": {
            "count": len(high_importance_indices),
            "features": [feature_names[i] for i in high_importance_indices],
            "threshold": float(high_threshold)
        },
        "low_importance": {
            "count": len(low_importance_indices),
            "features": [feature_names[i] for i in low_importance_indices],
            "threshold": float(low_threshold)
        }
    }

    # 특성 이름 패턴 분석 (prefix 기반)
    feature_prefixes = {}
    for name in feature_names:
        prefix = name.split('_')[0] if '_' in name else name[:3]
        if prefix not in feature_prefixes:
            feature_prefixes[prefix] = []
        feature_prefixes[prefix].append(name)

    # 접두사별 평균 중요도
    prefix_importance = {}
    for prefix, names in feature_prefixes.items():
        if len(names) > 1:  # 2개 이상의 특성을 가진 접두사만
            indices = [feature_names.index(name) for name in names]
            avg_importance = float(np.mean([abs(importance_array[i]) for i in indices]))
            prefix_importance[prefix] = {
                "feature_count": len(names),
                "avg_importance": avg_importance,
                "features": names
            }

    analysis["feature_groups"]["prefix_analysis"] = prefix_importance

    return analysis

def generate_feature_importance_insights(analysis: Dict[str, Any]) -> List[str]:
    """특성 중요도 인사이트 생성"""

    insights = []
    basic_stats = analysis["basic_statistics"]
    top_features = analysis["top_features"]

    # 전체적인 중요도 분포
    if basic_stats["std_importance"] / abs(basic_stats["mean_importance"]) > 1 if basic_stats["mean_importance"] != 0 else False:
        insights.append("특성 간 중요도 편차가 큽니다")
    else:
        insights.append("특성 간 중요도가 비교적 균등합니다")

    # 상위 특성 집중도
    if len(top_features["cumulative_importance"]) >= 3:
        top_3_cumulative = top_features["cumulative_importance"][2]  # 상위 3개 특성의 누적 중요도
        if top_3_cumulative > 50:
            insights.append(f"상위 3개 특성이 전체 중요도의 {top_3_cumulative:.1f}%를 차지합니다")
        elif top_3_cumulative < 25:
            insights.append("특성 중요도가 골고루 분산되어 있습니다")

    # 최고 중요도 특성
    if top_features["features"]:
        most_important = top_features["features"][0]
        most_important_value = top_features["importance_values"][0]
        insights.append(f"가장 중요한 특성: '{most_important}' (중요도: {most_important_value:.4f})")

    # 양수/음수 특성 비율
    total_features = basic_stats["total_features"]
    positive_ratio = basic_stats["positive_features"] / total_features
    if positive_ratio > 0.8:
        insights.append("대부분의 특성이 양의 중요도를 가집니다")
    elif positive_ratio < 0.2:
        insights.append("대부분의 특성이 음의 중요도를 가집니다")
    else:
        insights.append(f"양의 중요도 특성: {basic_stats['positive_features']}개, 음의 중요도 특성: {basic_stats['negative_features']}개")

    # 0 중요도 특성
    if basic_stats["zero_features"] > 0:
        insights.append(f"{basic_stats['zero_features']}개의 특성이 0 중요도를 가집니다")

    # 특성 그룹 분석
    feature_groups = analysis["feature_groups"]
    high_count = feature_groups["high_importance"]["count"]
    low_count = feature_groups["low_importance"]["count"]

    if high_count < total_features * 0.2:
        insights.append(f"소수의 특성({high_count}개)이 높은 중요도를 가집니다")

    if low_count > total_features * 0.5:
        insights.append(f"많은 특성({low_count}개)이 낮은 중요도를 가집니다")

    # 접두사 패턴 분석
    prefix_analysis = feature_groups.get("prefix_analysis", {})
    if prefix_analysis:
        best_prefix = max(prefix_analysis.items(), key=lambda x: x[1]["avg_importance"])
        insights.append(f"'{best_prefix[0]}' 접두사를 가진 특성들이 높은 중요도를 보입니다")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 특성 중요도 플롯 생성
        result = create_feature_importance_plot(**params)

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