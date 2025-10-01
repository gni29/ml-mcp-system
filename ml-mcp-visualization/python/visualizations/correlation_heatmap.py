#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Heatmap Visualization Module
상관관계 히트맵 시각화 모듈

이 모듈은 데이터의 상관관계를 히트맵으로 시각화합니다.
주요 기능:
- 다양한 상관계수 방법 지원 (Pearson, Spearman, Kendall)
- 클러스터링 기반 변수 재정렬
- 강한 상관관계 하이라이팅
- 맞춤형 색상 스키마
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 공유 유틸리티 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, output_results
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

    def output_results(results: Dict[str, Any]):
        """결과를 JSON 형태로 출력"""
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

def create_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson',
                              figure_size: Tuple[float, float] = (12, 10),
                              color_scheme: str = 'coolwarm',
                              output_file: str = 'correlation_heatmap.png') -> Dict[str, Any]:
    """
    상관관계 히트맵 생성

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    method : str
        상관계수 계산 방법 ('pearson', 'spearman', 'kendall')
    figure_size : Tuple[float, float]
        그림 크기 (가로, 세로)
    color_scheme : str
        색상 스키마
    output_file : str
        출력 파일명

    Returns:
    --------
    Dict[str, Any]
        히트맵 생성 결과
    """

    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "error": "상관관계 분석을 위해 최소 2개의 수치형 컬럼이 필요합니다",
            "available_columns": df.columns.tolist(),
            "numeric_columns": numeric_cols
        }

    try:
        # 상관관계 계산
        df_numeric = df[numeric_cols].dropna()

        if df_numeric.empty:
            return {
                "error": "유효한 데이터가 없습니다 (모든 행에 결측값 존재)",
                "variable_count": len(numeric_cols)
            }

        correlation_matrix = df_numeric.corr(method=method)

        # 결과 분석
        results = {
            "success": True,
            "method": method,
            "variable_count": len(numeric_cols),
            "data_points": len(df_numeric),
            "output_file": output_file
        }

        # 상관관계 통계
        corr_values = correlation_matrix.values
        mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
        upper_triangle = corr_values[mask]

        results["correlation_stats"] = {
            "max_correlation": float(np.max(upper_triangle)),
            "min_correlation": float(np.min(upper_triangle)),
            "mean_correlation": float(np.mean(np.abs(upper_triangle))),
            "strong_correlations": int(np.sum(np.abs(upper_triangle) > 0.7)),
            "moderate_correlations": int(np.sum((np.abs(upper_triangle) > 0.3) & (np.abs(upper_triangle) <= 0.7))),
            "weak_correlations": int(np.sum(np.abs(upper_triangle) <= 0.3))
        }

        # 강한 상관관계 찾기
        strong_pairs = _find_strong_correlations(correlation_matrix, threshold=0.7)
        results["strong_correlation_pairs"] = strong_pairs

        # 히트맵 생성
        _create_heatmap(correlation_matrix, method, figure_size, color_scheme, output_file, results)

        # 클러스터링된 히트맵 생성 (변수가 많은 경우)
        if len(numeric_cols) > 5:
            _create_clustered_heatmap(correlation_matrix, method, figure_size, color_scheme, results)

        # 상관관계 네트워크 다이어그램 생성
        if len(numeric_cols) <= 20:  # 너무 많으면 복잡해짐
            _create_correlation_network(correlation_matrix, results)

        return results

    except Exception as e:
        return {
            "error": f"상관관계 히트맵 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """강한 상관관계 쌍 찾기"""
    strong_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_pairs.append({
                    "variable1": corr_matrix.columns[i],
                    "variable2": corr_matrix.columns[j],
                    "correlation": float(corr_value),
                    "strength": "매우 강함" if abs(corr_value) >= 0.9 else "강함"
                })

    # 상관계수 절댓값 기준 정렬
    strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return strong_pairs

def _create_heatmap(corr_matrix: pd.DataFrame, method: str, figure_size: Tuple[float, float],
                   color_scheme: str, output_file: str, results: Dict[str, Any]):
    """기본 상관관계 히트맵 생성"""

    fig, ax = plt.subplots(figsize=figure_size)

    # 마스크 생성 (상삼각 행렬 숨기기)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 히트맵 그리기
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap=color_scheme,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        fmt='.2f',
        annot_kws={'size': 8 if len(corr_matrix) > 10 else 10}
    )

    # 제목과 레이블 설정
    method_names = {
        'pearson': 'Pearson 상관계수',
        'spearman': 'Spearman 순위 상관계수',
        'kendall': 'Kendall 순위 상관계수'
    }

    plt.title(f'{method_names.get(method, method)} 히트맵\n(n={results["data_points"]})',
              fontsize=16, fontweight='bold', pad=20)

    # 축 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 강한 상관관계 하이라이팅
    strong_pairs = results.get("strong_correlation_pairs", [])
    if strong_pairs:
        info_text = f"강한 상관관계: {len(strong_pairs)}쌍"
        if len(strong_pairs) > 0:
            top_pair = strong_pairs[0]
            info_text += f"\n최고: {top_pair['variable1']} ↔ {top_pair['variable2']} ({top_pair['correlation']:.3f})"

        plt.figtext(0.02, 0.02, info_text, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if "generated_files" not in results:
        results["generated_files"] = []
    results["generated_files"].append(output_file)

def _create_clustered_heatmap(corr_matrix: pd.DataFrame, method: str,
                             figure_size: Tuple[float, float], color_scheme: str,
                             results: Dict[str, Any]):
    """클러스터링된 상관관계 히트맵 생성"""

    try:
        # 계층적 클러스터링으로 변수 재정렬
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        # 거리 행렬 계산 (1 - |correlation|)
        distance_matrix = 1 - np.abs(corr_matrix)
        condensed_distances = squareform(distance_matrix, checks=False)

        # 계층적 클러스터링
        linkage_matrix = linkage(condensed_distances, method='average')

        # 클러스터링된 히트맵 그리기
        g = sns.clustermap(
            corr_matrix,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            cmap=color_scheme,
            center=0,
            annot=True,
            fmt='.2f',
            figsize=figure_size,
            cbar_kws={"shrink": 0.8},
            annot_kws={'size': 6 if len(corr_matrix) > 15 else 8}
        )

        g.fig.suptitle(f'클러스터링된 {method} 상관관계 히트맵', y=1.02, fontsize=14, fontweight='bold')

        output_file = output_file.replace('.png', '_clustered.png')
        g.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(output_file)

    except ImportError:
        # scipy가 없는 경우 건너뛰기
        pass
    except Exception as e:
        results["clustering_error"] = str(e)

def _create_correlation_network(corr_matrix: pd.DataFrame, results: Dict[str, Any]):
    """상관관계 네트워크 다이어그램 생성"""

    try:
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=(12, 12))

        n_vars = len(corr_matrix.columns)
        # 변수들을 원형으로 배치
        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # 노드 그리기 (변수들)
        for i, (var, x, y) in enumerate(zip(corr_matrix.columns, x_pos, y_pos)):
            circle = patches.Circle((x, y), 0.1, facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(i+1), ha='center', va='center', fontweight='bold', fontsize=8)

        # 엣지 그리기 (상관관계)
        lines = []
        line_widths = []
        line_colors = []

        threshold = 0.3  # 표시할 최소 상관관계 강도

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    lines.append([(x_pos[i], y_pos[i]), (x_pos[j], y_pos[j])])
                    line_widths.append(abs(corr_val) * 5)  # 상관계수에 비례한 선 두께
                    line_colors.append('red' if corr_val > 0 else 'blue')

        # 선 그리기
        for line, width, color in zip(lines, line_widths, line_colors):
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                   color=color, linewidth=width, alpha=0.6)

        # 범례 및 레이블
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # 변수명 범례
        legend_text = "\n".join([f"{i+1}: {var}" for i, var in enumerate(corr_matrix.columns)])
        ax.text(1.2, 0, legend_text, fontsize=8, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 색상 범례
        ax.text(-1.2, -1.2, "빨간색: 양의 상관관계\n파란색: 음의 상관관계\n선 두께: 상관관계 강도",
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.title('상관관계 네트워크 다이어그램', fontsize=14, fontweight='bold', pad=20)

        output_file = results["output_file"].replace('.png', '_network.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(output_file)

    except Exception as e:
        results["network_error"] = str(e)

def main():
    """
    메인 실행 함수 - 상관관계 히트맵 생성의 진입점
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 데이터 파일 로드
        data_file = params.get('data_file')
        if not data_file:
            raise ValueError("data_file 매개변수가 필요합니다")

        df = load_data(data_file)

        # 매개변수 추출
        method = params.get('method', 'pearson')
        figure_size = params.get('figure_size', [12, 10])
        color_scheme = params.get('color_scheme', 'coolwarm')
        output_file = params.get('output_file', 'correlation_heatmap.png')

        # 상관관계 히트맵 생성
        result = create_correlation_heatmap(df, method, tuple(figure_size), color_scheme, output_file)

        # 결과 출력
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()