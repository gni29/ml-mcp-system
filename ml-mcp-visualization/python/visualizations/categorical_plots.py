#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorical Plots Visualization Module
범주형 플롯 시각화 모듈

이 모듈은 범주형 데이터의 포괄적인 시각화를 제공합니다.
주요 기능:
- 막대 차트 (Bar Charts)
- 파이 차트 (Pie Charts)
- 박스 플롯 (Box Plots)
- 바이올린 플롯 (Violin Plots)
- 히트맵 (Heatmaps)
- 스트립 플롯 (Strip Plots)
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
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

def create_categorical_plots(df: pd.DataFrame, categorical_columns: List[str],
                           numeric_columns: List[str] = None,
                           plot_types: List[str] = None,
                           output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    범주형 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    categorical_columns : List[str]
        범주형 컬럼들
    numeric_columns : List[str], optional
        수치형 컬럼들 (박스플롯, 바이올린플롯 등에 사용)
    plot_types : List[str], optional
        생성할 플롯 유형들 ['bar', 'pie', 'box', 'violin', 'heatmap', 'strip']
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        범주형 시각화 결과
    """

    if plot_types is None:
        plot_types = ['bar', 'pie', 'box']

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 기본 유효성 검사
    for col in categorical_columns:
        if col not in df.columns:
            return {
                "error": f"범주형 컬럼 '{col}'을 찾을 수 없습니다",
                "available_columns": df.columns.tolist()
            }

    try:
        results = {
            "success": True,
            "categorical_columns": categorical_columns,
            "numeric_columns": numeric_columns,
            "plot_types": plot_types,
            "data_points": len(df),
            "generated_files": [],
            "statistics": {}
        }

        # 스타일 설정
        sns.set_style("whitegrid")
        plt.style.use('default')

        # 각 플롯 유형별 생성
        for plot_type in plot_types:
            if plot_type == 'bar':
                _create_bar_charts(df, categorical_columns, numeric_columns, output_dir, results)
            elif plot_type == 'pie':
                _create_pie_charts(df, categorical_columns, output_dir, results)
            elif plot_type == 'box':
                _create_box_plots(df, categorical_columns, numeric_columns, output_dir, results)
            elif plot_type == 'violin':
                _create_violin_plots(df, categorical_columns, numeric_columns, output_dir, results)
            elif plot_type == 'heatmap':
                _create_heatmaps(df, categorical_columns, output_dir, results)
            elif plot_type == 'strip':
                _create_strip_plots(df, categorical_columns, numeric_columns, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"범주형 시각화 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_bar_charts(df: pd.DataFrame, categorical_columns: List[str],
                      numeric_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """막대 차트 생성"""

    for cat_col in categorical_columns:
        # 범주별 빈도수 계산
        value_counts = df[cat_col].value_counts()

        if len(value_counts) == 0:
            continue

        # 1. 기본 빈도수 막대 차트
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 빈도수 막대 차트
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        bars = ax1.bar(range(len(value_counts)), value_counts.values, color=colors)
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.set_title(f'{cat_col} 빈도수', fontweight='bold')
        ax1.set_ylabel('빈도수')

        # 값 라벨 추가
        for bar, count in zip(bars, value_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        # 2. 수평 막대 차트
        ax2 = axes[0, 1]
        y_pos = range(len(value_counts))
        bars_h = ax2.barh(y_pos, value_counts.values, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(value_counts.index)
        ax2.set_title(f'{cat_col} 빈도수 (수평)', fontweight='bold')
        ax2.set_xlabel('빈도수')

        # 값 라벨 추가
        for bar, count in zip(bars_h, value_counts.values):
            ax2.text(bar.get_width() + max(value_counts.values) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center')

        # 3. 비율 막대 차트
        ax3 = axes[1, 0]
        percentages = (value_counts / value_counts.sum()) * 100
        bars_pct = ax3.bar(range(len(percentages)), percentages.values, color=colors)
        ax3.set_xticks(range(len(percentages)))
        ax3.set_xticklabels(percentages.index, rotation=45, ha='right')
        ax3.set_title(f'{cat_col} 비율 (%)', fontweight='bold')
        ax3.set_ylabel('비율 (%)')

        for bar, pct in zip(bars_pct, percentages.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{pct:.1f}%', ha='center', va='bottom')

        # 4. 수치 데이터와의 관계 (첫 번째 수치 컬럼)
        if numeric_columns:
            ax4 = axes[1, 1]
            grouped_mean = df.groupby(cat_col)[numeric_columns[0]].mean()
            bars_mean = ax4.bar(range(len(grouped_mean)), grouped_mean.values, color=colors)
            ax4.set_xticks(range(len(grouped_mean)))
            ax4.set_xticklabels(grouped_mean.index, rotation=45, ha='right')
            ax4.set_title(f'{cat_col}별 {numeric_columns[0]} 평균', fontweight='bold')
            ax4.set_ylabel(f'{numeric_columns[0]} 평균')

            for bar, mean_val in zip(bars_mean, grouped_mean.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{mean_val:.2f}', ha='center', va='bottom')

            # 통계 정보 저장
            results["statistics"].setdefault("bar_charts", {})[cat_col] = {
                "unique_categories": len(value_counts),
                "most_frequent": value_counts.index[0],
                "most_frequent_count": int(value_counts.iloc[0]),
                "least_frequent": value_counts.index[-1],
                "least_frequent_count": int(value_counts.iloc[-1]),
                "mean_by_category": grouped_mean.to_dict()
            }
        else:
            ax4.text(0.5, 0.5, '수치 데이터 없음', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)
            ax4.set_title('수치 데이터 관계 분석', fontweight='bold')

        plt.suptitle(f'{cat_col} 막대 차트 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'categorical_bar_{cat_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _create_pie_charts(df: pd.DataFrame, categorical_columns: List[str],
                      output_dir: str, results: Dict[str, Any]):
    """파이 차트 생성"""

    for cat_col in categorical_columns:
        value_counts = df[cat_col].value_counts()

        if len(value_counts) == 0:
            continue

        # 너무 많은 범주가 있는 경우 상위 10개만 표시
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            other_count = value_counts.tail(-10).sum()
            if other_count > 0:
                top_values['기타'] = other_count
            value_counts = top_values

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 기본 파이 차트
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax1.pie(value_counts.values, labels=value_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title(f'{cat_col} 분포 (파이 차트)', fontweight='bold')

        # 텍스트 스타일 조정
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 2. 도넛 차트
        ax2 = axes[1]
        wedges2, texts2, autotexts2 = ax2.pie(value_counts.values, labels=value_counts.index,
                                             autopct='%1.1f%%', colors=colors, startangle=90,
                                             wedgeprops=dict(width=0.7))
        ax2.set_title(f'{cat_col} 분포 (도넛 차트)', fontweight='bold')

        # 중앙에 총 개수 표시
        ax2.text(0, 0, f'총 {value_counts.sum()}개', ha='center', va='center',
                fontsize=12, fontweight='bold')

        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        output_file = Path(output_dir) / f'categorical_pie_{cat_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

        # 통계 정보 저장
        results["statistics"].setdefault("pie_charts", {})[cat_col] = {
            "categories_shown": len(value_counts),
            "largest_segment": value_counts.index[0],
            "largest_percentage": float(value_counts.iloc[0] / value_counts.sum() * 100),
            "entropy": float(-sum(p * np.log2(p) for p in value_counts/value_counts.sum() if p > 0))
        }

def _create_box_plots(df: pd.DataFrame, categorical_columns: List[str],
                     numeric_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """박스 플롯 생성"""

    if not numeric_columns:
        return

    for cat_col in categorical_columns:
        n_numeric = len(numeric_columns)
        fig, axes = plt.subplots((n_numeric + 1) // 2, 2, figsize=(16, 6 * ((n_numeric + 1) // 2)))

        if n_numeric == 1:
            axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, num_col in enumerate(numeric_columns):
            if i >= len(axes):
                break

            ax = axes[i]

            # 박스플롯 생성
            box_plot = ax.boxplot([df[df[cat_col] == cat][num_col].dropna()
                                  for cat in df[cat_col].unique()],
                                 labels=df[cat_col].unique(), patch_artist=True)

            # 색상 설정
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f'{cat_col}별 {num_col} 분포', fontweight='bold')
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            # 통계 정보 계산
            group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])

            results["statistics"].setdefault("box_plots", {}).setdefault(cat_col, {})[num_col] = {
                "group_statistics": group_stats.to_dict(),
                "overall_mean": float(df[num_col].mean()),
                "overall_std": float(df[num_col].std())
            }

        # 빈 서브플롯 숨기기
        for i in range(n_numeric, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{cat_col} 그룹별 박스 플롯', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'categorical_box_{cat_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _create_violin_plots(df: pd.DataFrame, categorical_columns: List[str],
                        numeric_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """바이올린 플롯 생성"""

    if not numeric_columns:
        return

    for cat_col in categorical_columns:
        n_numeric = min(len(numeric_columns), 4)  # 최대 4개까지만
        fig, axes = plt.subplots((n_numeric + 1) // 2, 2, figsize=(16, 6 * ((n_numeric + 1) // 2)))

        if n_numeric == 1:
            axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, num_col in enumerate(numeric_columns[:n_numeric]):
            ax = axes[i]

            try:
                # 바이올린 플롯 생성
                sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax, palette='Set3')
                ax.set_title(f'{cat_col}별 {num_col} 분포 (바이올린)', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

                # 평균값 점 추가
                group_means = df.groupby(cat_col)[num_col].mean()
                for j, (category, mean_val) in enumerate(group_means.items()):
                    ax.plot(j, mean_val, 'ro', markersize=8, markerfacecolor='red',
                           markeredgecolor='darkred', markeredgewidth=2)

            except Exception as e:
                ax.text(0.5, 0.5, f'바이올린 플롯 생성 실패:\n{str(e)}',
                       transform=ax.transAxes, ha='center', va='center')

        # 빈 서브플롯 숨기기
        for i in range(n_numeric, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{cat_col} 그룹별 바이올린 플롯', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'categorical_violin_{cat_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _create_heatmaps(df: pd.DataFrame, categorical_columns: List[str],
                    output_dir: str, results: Dict[str, Any]):
    """히트맵 생성 (범주형 변수 간 관계)"""

    if len(categorical_columns) < 2:
        return

    # 범주형 변수들 간의 교차표 생성
    for i, cat1 in enumerate(categorical_columns):
        for cat2 in categorical_columns[i+1:]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 1. 빈도수 교차표
            ax1 = axes[0]
            crosstab_count = pd.crosstab(df[cat1], df[cat2])
            sns.heatmap(crosstab_count, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'{cat1} vs {cat2} 빈도수', fontweight='bold')

            # 2. 비율 교차표
            ax2 = axes[1]
            crosstab_prop = pd.crosstab(df[cat1], df[cat2], normalize='index')
            sns.heatmap(crosstab_prop, annot=True, fmt='.2f', cmap='Reds', ax=ax2)
            ax2.set_title(f'{cat1} vs {cat2} 비율', fontweight='bold')

            plt.tight_layout()

            output_file = Path(output_dir) / f'categorical_heatmap_{cat1}_{cat2}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            results["generated_files"].append(str(output_file))

            # 통계 정보 저장 (카이제곱 검정)
            try:
                from scipy.stats import chi2_contingency
                chi2, p_value, dof, expected = chi2_contingency(crosstab_count)

                results["statistics"].setdefault("heatmaps", {})[f"{cat1}_vs_{cat2}"] = {
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_value),
                    "degrees_of_freedom": int(dof),
                    "significant": p_value < 0.05
                }
            except ImportError:
                results["statistics"].setdefault("heatmaps", {})[f"{cat1}_vs_{cat2}"] = {
                    "note": "scipy를 사용할 수 없어 통계 검정을 수행하지 못했습니다"
                }

def _create_strip_plots(df: pd.DataFrame, categorical_columns: List[str],
                       numeric_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """스트립 플롯 생성"""

    if not numeric_columns:
        return

    for cat_col in categorical_columns:
        n_numeric = min(len(numeric_columns), 4)
        fig, axes = plt.subplots((n_numeric + 1) // 2, 2, figsize=(16, 6 * ((n_numeric + 1) // 2)))

        if n_numeric == 1:
            axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, num_col in enumerate(numeric_columns[:n_numeric]):
            ax = axes[i]

            try:
                # 스트립 플롯 + 박스 플롯 조합
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax,
                           palette='Set3', width=0.6, showfliers=False)
                sns.stripplot(data=df, x=cat_col, y=num_col, ax=ax,
                             color='black', alpha=0.6, size=4)

                ax.set_title(f'{cat_col}별 {num_col} 분포 (스트립)', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'스트립 플롯 생성 실패:\n{str(e)}',
                       transform=ax.transAxes, ha='center', va='center')

        # 빈 서브플롯 숨기기
        for i in range(n_numeric, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{cat_col} 그룹별 스트립 플롯', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'categorical_strip_{cat_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def main():
    """
    메인 실행 함수 - 범주형 시각화의 진입점
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
        categorical_columns = params.get('categorical_columns')
        numeric_columns = params.get('numeric_columns')
        plot_types = params.get('plot_types', ['bar', 'pie', 'box'])
        output_dir = params.get('output_dir', 'visualizations')

        if not categorical_columns:
            raise ValueError("categorical_columns 매개변수가 필요합니다")

        # 범주형 시각화 생성
        result = create_categorical_plots(df, categorical_columns, numeric_columns, plot_types, output_dir)

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