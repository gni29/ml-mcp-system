#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Plots Visualization Module
분포 시각화 모듈

이 모듈은 데이터 분포를 다양한 방법으로 시각화합니다.
주요 기능:
- 히스토그램, 박스플롯, 바이올린 플롯
- 밀도 곡선 및 Q-Q 플롯
- 다중 변수 분포 비교
- 자동 레이아웃 및 스타일링
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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

    def output_results(results: Dict[str, Any]):
        """결과를 JSON 형태로 출력"""
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

def create_distribution_plots(df: pd.DataFrame, columns: Optional[List[str]] = None,
                             plot_types: List[str] = None, output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    분포 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        시각화할 데이터프레임
    columns : List[str], optional
        시각화할 컬럼들 (None이면 모든 수치형 컬럼)
    plot_types : List[str]
        생성할 플롯 유형들 ['histogram', 'boxplot', 'violin', 'density', 'qq']
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        시각화 결과 정보
    """

    if plot_types is None:
        plot_types = ['histogram', 'boxplot']

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 수치형 컬럼 선택
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        return {
            "error": "시각화할 수치형 컬럼이 없습니다",
            "available_columns": df.columns.tolist()
        }

    results = {
        "success": True,
        "analyzed_columns": len(numeric_cols),
        "plot_types": plot_types,
        "generated_files": []
    }

    try:
        # 스타일 설정
        sns.set_style("whitegrid")
        plt.style.use('default')

        # 각 플롯 유형별 생성
        for plot_type in plot_types:
            if plot_type == 'histogram':
                _create_histograms(df, numeric_cols, output_dir, results)
            elif plot_type == 'boxplot':
                _create_boxplots(df, numeric_cols, output_dir, results)
            elif plot_type == 'violin':
                _create_violin_plots(df, numeric_cols, output_dir, results)
            elif plot_type == 'density':
                _create_density_plots(df, numeric_cols, output_dir, results)
            elif plot_type == 'qq':
                _create_qq_plots(df, numeric_cols, output_dir, results)

        # 종합 분포 비교 차트
        if len(numeric_cols) > 1:
            _create_distribution_comparison(df, numeric_cols, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"분포 시각화 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_histograms(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """히스토그램 생성"""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]
            data = df[col].dropna()

            # 히스토그램 그리기
            ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{col} 분포', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('빈도')

            # 통계 정보 추가
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'평균: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.2f}')
            ax.legend(fontsize=8)

    # 빈 서브플롯 제거
    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    output_file = Path(output_dir) / 'histograms.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_boxplots(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """박스플롯 생성"""
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), 6))

    # 데이터 준비 (정규화)
    data_for_plot = []
    labels = []

    for col in columns:
        data = df[col].dropna()
        if len(data) > 0:
            data_for_plot.append(data)
            labels.append(col)

    if data_for_plot:
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

        # 색상 설정
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title('변수별 박스플롯 비교', fontsize=14, fontweight='bold')
        ax.set_ylabel('값')
        plt.xticks(rotation=45)

        # 이상치 정보 추가
        outlier_info = []
        for i, col in enumerate(labels):
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            outlier_info.append(f"{col}: {len(outliers)}개")

        # 이상치 정보를 텍스트로 추가
        info_text = "이상치 개수: " + ", ".join(outlier_info)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = Path(output_dir) / 'boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_violin_plots(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """바이올린 플롯 생성"""
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), 6))

    # 데이터 준비
    plot_data = df[columns].dropna()

    if not plot_data.empty:
        # 바이올린 플롯 생성
        parts = ax.violinplot([plot_data[col].values for col in columns],
                             positions=range(1, len(columns) + 1),
                             showmeans=True, showmedians=True)

        # 색상 설정
        colors = plt.cm.Set2(np.linspace(0, 1, len(columns)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(columns) + 1))
        ax.set_xticklabels(columns, rotation=45)
        ax.set_title('변수별 바이올린 플롯 (분포 밀도)', fontsize=14, fontweight='bold')
        ax.set_ylabel('값')

        # 범례 추가
        ax.text(0.02, 0.02, '빨간선: 중위수, 검은선: 평균', transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_file = Path(output_dir) / 'violin_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_density_plots(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """밀도 곡선 플롯 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))

    for col, color in zip(columns, colors):
        data = df[col].dropna()
        if len(data) > 1:
            # 표준화된 데이터로 밀도 플롯
            normalized_data = (data - data.mean()) / data.std()
            ax.hist(normalized_data, bins=50, alpha=0.3, density=True,
                   color=color, label=f'{col} (표준화)')

            # KDE 곡선 추가
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(normalized_data)
                x_range = np.linspace(normalized_data.min(), normalized_data.max(), 100)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2)
            except:
                pass

    ax.set_title('변수별 정규화 분포 비교', fontsize=14, fontweight='bold')
    ax.set_xlabel('표준화된 값 (Z-score)')
    ax.set_ylabel('밀도')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / 'density_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_qq_plots(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Q-Q 플롯 생성 (정규성 검정)"""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    try:
        from scipy import stats

        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                data = df[col].dropna()

                if len(data) > 3:
                    # Q-Q 플롯
                    stats.probplot(data, dist="norm", plot=ax)
                    ax.set_title(f'{col} Q-Q 플롯 (정규성 검정)', fontsize=10)

                    # Shapiro-Wilk 검정
                    if len(data) <= 5000:  # 샘플 크기 제한
                        statistic, p_value = stats.shapiro(data)
                        interpretation = "정규분포" if p_value > 0.05 else "비정규분포"
                        ax.text(0.05, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}\n({interpretation})',
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                else:
                    ax.text(0.5, 0.5, f'{col}\n데이터 부족', ha='center', va='center', transform=ax.transAxes)

        # 빈 서브플롯 제거
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])

    except ImportError:
        # scipy가 없는 경우 간단한 정규성 시각화
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                data = df[col].dropna()

                # 히스토그램과 정규분포 곡선 비교
                ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue')

                # 정규분포 곡선 그리기
                mu, sigma = data.mean(), data.std()
                x = np.linspace(data.min(), data.max(), 100)
                normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ax.plot(x, normal_curve, 'r-', linewidth=2, label='정규분포 곡선')

                ax.set_title(f'{col} 정규성 비교', fontsize=10)
                ax.legend()

    plt.tight_layout()
    output_file = Path(output_dir) / 'qq_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_distribution_comparison(df: pd.DataFrame, columns: List[str], output_dir: str, results: Dict[str, Any]):
    """분포 비교 종합 차트"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 정규화된 히스토그램 오버레이
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    for col, color in zip(columns, colors):
        data = df[col].dropna()
        if len(data) > 0:
            normalized_data = (data - data.mean()) / data.std()
            ax1.hist(normalized_data, bins=30, alpha=0.4, label=col, color=color, density=True)

    ax1.set_title('정규화된 분포 비교', fontweight='bold')
    ax1.set_xlabel('표준화된 값')
    ax1.set_ylabel('밀도')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 박스플롯 비교
    data_for_box = [df[col].dropna() for col in columns]
    bp = ax2.boxplot(data_for_box, labels=columns, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('박스플롯 비교', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45)

    # 3. 분위수 비교
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    x_pos = np.arange(len(quantiles))
    width = 0.8 / len(columns)

    for i, col in enumerate(columns):
        data = df[col].dropna()
        if len(data) > 0:
            q_values = [data.quantile(q) for q in quantiles]
            ax3.bar(x_pos + i * width, q_values, width, label=col, alpha=0.7, color=colors[i])

    ax3.set_title('분위수 비교', fontweight='bold')
    ax3.set_xlabel('분위수')
    ax3.set_ylabel('값')
    ax3.set_xticks(x_pos + width * (len(columns) - 1) / 2)
    ax3.set_xticklabels([f'{q*100:.0f}%' for q in quantiles])
    ax3.legend()

    # 4. 통계 요약 테이블
    ax4.axis('tight')
    ax4.axis('off')

    stats_data = []
    for col in columns:
        data = df[col].dropna()
        if len(data) > 0:
            stats_data.append([
                col,
                f"{data.mean():.2f}",
                f"{data.std():.2f}",
                f"{data.min():.2f}",
                f"{data.max():.2f}",
                f"{data.skew():.2f}" if hasattr(data, 'skew') else "N/A"
            ])

    table = ax4.table(cellText=stats_data,
                     colLabels=['변수', '평균', '표준편차', '최솟값', '최댓값', '왜도'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('기술통계 요약', fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = Path(output_dir) / 'distribution_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def main():
    """
    메인 실행 함수 - 분포 시각화의 진입점
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
        columns = params.get('columns')
        plot_types = params.get('plot_types', ['histogram', 'boxplot'])
        output_dir = params.get('output_dir', 'visualizations')

        # 분포 시각화 생성
        result = create_distribution_plots(df, columns, plot_types, output_dir)

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