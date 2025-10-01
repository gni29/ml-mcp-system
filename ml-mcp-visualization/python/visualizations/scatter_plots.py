#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter Plot Visualization Module
산점도 시각화 모듈

이 모듈은 다차원 산점도 분석과 시각화를 제공합니다.
주요 기능:
- 2D/3D 산점도 생성
- 산점도 매트릭스 (pair plot)
- 범주별 색상 구분
- 추세선 및 회귀선 분석
- 이상치 탐지 및 표시
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

def create_scatter_plots(df: pd.DataFrame, x_column: Optional[str] = None,
                        y_column: Optional[str] = None, color_column: Optional[str] = None,
                        create_matrix: bool = True, add_trendline: bool = True,
                        output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    산점도 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        시각화할 데이터프레임
    x_column : str, optional
        X축 컬럼 (개별 산점도용)
    y_column : str, optional
        Y축 컬럼 (개별 산점도용)
    color_column : str, optional
        색상 구분 컬럼
    create_matrix : bool
        산점도 매트릭스 생성 여부
    add_trendline : bool
        추세선 추가 여부
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        산점도 생성 결과
    """

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 수치형 컬럼 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "error": "산점도 생성을 위해 최소 2개의 수치형 컬럼이 필요합니다",
            "available_columns": df.columns.tolist(),
            "numeric_columns": numeric_cols
        }

    results = {
        "success": True,
        "numeric_columns": numeric_cols,
        "generated_files": []
    }

    try:
        # 스타일 설정
        sns.set_style("whitegrid")
        plt.style.use('default')

        # 1. 개별 산점도 생성 (X, Y 컬럼이 지정된 경우)
        if x_column and y_column and x_column in numeric_cols and y_column in numeric_cols:
            _create_individual_scatter(df, x_column, y_column, color_column, add_trendline, output_dir, results)

        # 2. 산점도 매트릭스 생성
        if create_matrix and len(numeric_cols) > 1:
            _create_scatter_matrix(df, numeric_cols, color_column, output_dir, results)

        # 3. 3D 산점도 생성 (3개 이상의 수치형 컬럼이 있는 경우)
        if len(numeric_cols) >= 3:
            _create_3d_scatter(df, numeric_cols, color_column, output_dir, results)

        # 4. 이상치 탐지 산점도
        if len(numeric_cols) >= 2:
            _create_outlier_scatter(df, numeric_cols, output_dir, results)

        # 5. 상관관계 강도별 산점도
        if len(numeric_cols) > 2:
            _create_correlation_scatter(df, numeric_cols, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"산점도 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_individual_scatter(df: pd.DataFrame, x_column: str, y_column: str,
                              color_column: Optional[str], add_trendline: bool,
                              output_dir: str, results: Dict[str, Any]):
    """개별 산점도 생성"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 색상 설정
    if color_column and color_column in df.columns:
        # 범주형 변수인 경우
        if df[color_column].dtype == 'object' or df[color_column].nunique() < 10:
            scatter = ax.scatter(df[x_column], df[y_column], c=df[color_column].astype('category').cat.codes,
                               cmap='tab10', alpha=0.7, s=50)

            # 범례 추가
            unique_values = df[color_column].unique()
            if len(unique_values) <= 10:  # 범례가 너무 많으면 생략
                for i, val in enumerate(unique_values):
                    if pd.notna(val):
                        ax.scatter([], [], c=plt.cm.tab10(i), label=str(val), s=50)
                ax.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # 연속형 변수인 경우
            scatter = ax.scatter(df[x_column], df[y_column], c=df[color_column],
                               cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax, label=color_column)
    else:
        # 단색 산점도
        ax.scatter(df[x_column], df[y_column], alpha=0.7, s=50, color='skyblue', edgecolors='black')

    # 추세선 추가
    if add_trendline:
        try:
            # 다항식 회귀선 (1차)
            z = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[x_column].min(), df[x_column].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'추세선 (기울기: {z[0]:.3f})')

            # 상관계수 표시
            corr = df[x_column].corr(df[y_column])
            ax.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

            ax.legend()
        except Exception as e:
            results["trendline_error"] = str(e)

    # 축 레이블 및 제목
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    title = f'{x_column} vs {y_column}'
    if color_column:
        title += f' (색상: {color_column})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 격자 표시
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / f'scatter_{x_column}_vs_{y_column}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_scatter_matrix(df: pd.DataFrame, numeric_cols: List[str],
                          color_column: Optional[str], output_dir: str, results: Dict[str, Any]):
    """산점도 매트릭스 생성 (pair plot)"""

    # 컬럼 수가 너무 많으면 제한
    if len(numeric_cols) > 8:
        numeric_cols = numeric_cols[:8]
        results["matrix_note"] = f"성능상 이유로 처음 8개 컬럼만 표시: {', '.join(numeric_cols)}"

    # 데이터 준비
    plot_data = df[numeric_cols].copy()

    # 색상 컬럼 추가 (있는 경우)
    hue_column = None
    if color_column and color_column in df.columns:
        if df[color_column].dtype == 'object' or df[color_column].nunique() < 10:
            plot_data[color_column] = df[color_column]
            hue_column = color_column

    try:
        # Seaborn pairplot 사용
        if hue_column:
            g = sns.pairplot(plot_data, hue=hue_column, diag_kind='hist',
                           plot_kws={'alpha': 0.7, 's': 30}, diag_kws={'alpha': 0.7})
        else:
            g = sns.pairplot(plot_data, diag_kind='hist',
                           plot_kws={'alpha': 0.7, 's': 30, 'color': 'skyblue'},
                           diag_kws={'alpha': 0.7, 'color': 'skyblue'})

        g.fig.suptitle('산점도 매트릭스 (Pair Plot)', y=1.02, fontsize=16, fontweight='bold')

        # 상관계수 추가
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if i != j:
                    corr = plot_data[numeric_cols[i]].corr(plot_data[numeric_cols[j]])
                    g.axes[i, j].text(0.05, 0.95, f'r={corr:.2f}',
                                     transform=g.axes[i, j].transAxes,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                     fontsize=8)

        output_file = Path(output_dir) / 'scatter_matrix.png'
        g.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

    except Exception as e:
        results["matrix_error"] = str(e)

def _create_3d_scatter(df: pd.DataFrame, numeric_cols: List[str],
                      color_column: Optional[str], output_dir: str, results: Dict[str, Any]):
    """3D 산점도 생성"""

    # 처음 3개 컬럼 사용
    x_col, y_col, z_col = numeric_cols[:3]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 색상 설정
    if color_column and color_column in df.columns:
        if df[color_column].dtype == 'object' or df[color_column].nunique() < 10:
            # 범주형
            unique_values = df[color_column].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))

            for i, val in enumerate(unique_values):
                if pd.notna(val):
                    mask = df[color_column] == val
                    ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], df.loc[mask, z_col],
                             c=[colors[i]], label=str(val), alpha=0.7, s=50)
            ax.legend()
        else:
            # 연속형
            scatter = ax.scatter(df[x_col], df[y_col], df[z_col], c=df[color_column],
                               cmap='viridis', alpha=0.7, s=50)
            fig.colorbar(scatter, ax=ax, label=color_column, shrink=0.8)
    else:
        # 단색
        ax.scatter(df[x_col], df[y_col], df[z_col], alpha=0.7, s=50, color='skyblue')

    # 축 레이블
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_zlabel(z_col, fontsize=10)

    title = f'3D 산점도: {x_col} × {y_col} × {z_col}'
    if color_column:
        title += f' (색상: {color_column})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = Path(output_dir) / 'scatter_3d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_outlier_scatter(df: pd.DataFrame, numeric_cols: List[str],
                           output_dir: str, results: Dict[str, Any]):
    """이상치 탐지 산점도"""

    # 처음 2개 컬럼 사용
    x_col, y_col = numeric_cols[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 기본 산점도
    ax1.scatter(df[x_col], df[y_col], alpha=0.7, s=50, color='skyblue', edgecolors='black')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title('기본 산점도', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. 이상치 강조 산점도
    # IQR 방법으로 이상치 탐지
    outliers_x = _detect_outliers(df[x_col])
    outliers_y = _detect_outliers(df[y_col])
    outliers = outliers_x | outliers_y

    # 정상값과 이상치 구분 표시
    ax2.scatter(df.loc[~outliers, x_col], df.loc[~outliers, y_col],
               alpha=0.7, s=50, color='skyblue', edgecolors='black', label='정상값')

    if outliers.sum() > 0:
        ax2.scatter(df.loc[outliers, x_col], df.loc[outliers, y_col],
                   alpha=0.8, s=80, color='red', edgecolors='darkred',
                   marker='D', label=f'이상치 ({outliers.sum()}개)')

    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title('이상치 탐지 산점도', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 이상치 통계 텍스트 추가
    outlier_info = f"이상치: {outliers.sum()}개 ({outliers.sum()/len(df)*100:.1f}%)"
    ax2.text(0.02, 0.98, outlier_info, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            verticalalignment='top', fontsize=9)

    plt.tight_layout()
    output_file = Path(output_dir) / 'scatter_outliers.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))
    results["outlier_detection"] = {
        "total_outliers": int(outliers.sum()),
        "outlier_percentage": float(outliers.sum()/len(df)*100),
        "method": "IQR"
    }

def _create_correlation_scatter(df: pd.DataFrame, numeric_cols: List[str],
                               output_dir: str, results: Dict[str, Any]):
    """상관관계 강도별 산점도"""

    # 모든 쌍의 상관계수 계산
    correlations = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr = df[numeric_cols[i]].corr(df[numeric_cols[j]])
            correlations.append({
                'var1': numeric_cols[i],
                'var2': numeric_cols[j],
                'correlation': abs(corr),
                'correlation_raw': corr
            })

    # 상관계수 절댓값으로 정렬
    correlations.sort(key=lambda x: x['correlation'], reverse=True)

    # 상위 4개 쌍 표시 (또는 전체가 4개 미만이면 전체)
    top_pairs = correlations[:min(4, len(correlations))]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, pair in enumerate(top_pairs):
        if i >= 4:
            break

        ax = axes[i]
        x_col, y_col = pair['var1'], pair['var2']
        corr = pair['correlation_raw']

        # 산점도 그리기
        ax.scatter(df[x_col], df[y_col], alpha=0.7, s=50, color='skyblue', edgecolors='black')

        # 회귀선 추가
        try:
            z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except:
            pass

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}\n상관계수: {corr:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 상관관계 강도 표시
        if abs(corr) >= 0.8:
            strength = "매우 강함"
            color = "red"
        elif abs(corr) >= 0.6:
            strength = "강함"
            color = "orange"
        elif abs(corr) >= 0.4:
            strength = "보통"
            color = "yellow"
        else:
            strength = "약함"
            color = "lightblue"

        ax.text(0.02, 0.98, strength, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
               verticalalignment='top', fontsize=9)

    # 빈 서브플롯 제거
    for i in range(len(top_pairs), 4):
        fig.delaxes(axes[i])

    plt.suptitle('상관관계 강도별 산점도 (상위 4쌍)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = Path(output_dir) / 'scatter_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))
    results["correlation_analysis"] = {
        "top_correlations": [
            {
                "variables": f"{pair['var1']} - {pair['var2']}",
                "correlation": pair['correlation_raw'],
                "strength": "매우 강함" if abs(pair['correlation_raw']) >= 0.8 else
                          "강함" if abs(pair['correlation_raw']) >= 0.6 else
                          "보통" if abs(pair['correlation_raw']) >= 0.4 else "약함"
            }
            for pair in top_pairs
        ]
    }

def _detect_outliers(series: pd.Series) -> pd.Series:
    """IQR 방법으로 이상치 탐지"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

def main():
    """
    메인 실행 함수 - 산점도 시각화의 진입점
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
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        color_column = params.get('color_column')
        create_matrix = params.get('create_matrix', True)
        add_trendline = params.get('add_trendline', True)
        output_dir = params.get('output_dir', 'visualizations')

        # 산점도 생성
        result = create_scatter_plots(df, x_column, y_column, color_column,
                                     create_matrix, add_trendline, output_dir)

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