#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Plots Visualization Module
통계 플롯 시각화 모듈

이 모듈은 통계 분석을 위한 포괄적인 시각화를 제공합니다.
주요 기능:
- 분포 플롯 (Distribution Plots)
- Q-Q 플롯 (Quantile-Quantile Plots)
- 잔차 플롯 (Residual Plots)
- 확률 플롯 (Probability Plots)
- 통계 검정 시각화
- 신뢰구간 플롯
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

def create_statistical_plots(df: pd.DataFrame, numeric_columns: List[str],
                           target_column: str = None,
                           plot_types: List[str] = None,
                           output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    통계 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    numeric_columns : List[str]
        분석할 수치형 컬럼들
    target_column : str, optional
        목표 변수 (회귀 분석용)
    plot_types : List[str], optional
        생성할 플롯 유형들 ['distribution', 'qq', 'residual', 'probability', 'confidence']
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        통계 시각화 결과
    """

    if plot_types is None:
        plot_types = ['distribution', 'qq', 'probability']

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 기본 유효성 검사
    for col in numeric_columns:
        if col not in df.columns:
            return {
                "error": f"수치형 컬럼 '{col}'을 찾을 수 없습니다",
                "available_columns": df.columns.tolist()
            }

    try:
        results = {
            "success": True,
            "numeric_columns": numeric_columns,
            "target_column": target_column,
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
            if plot_type == 'distribution':
                _create_distribution_plots(df, numeric_columns, output_dir, results)
            elif plot_type == 'qq':
                _create_qq_plots(df, numeric_columns, output_dir, results)
            elif plot_type == 'residual' and target_column:
                _create_residual_plots(df, numeric_columns, target_column, output_dir, results)
            elif plot_type == 'probability':
                _create_probability_plots(df, numeric_columns, output_dir, results)
            elif plot_type == 'confidence':
                _create_confidence_plots(df, numeric_columns, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"통계 시각화 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_distribution_plots(df: pd.DataFrame, numeric_columns: List[str],
                              output_dir: str, results: Dict[str, Any]):
    """분포 플롯 생성"""

    for col in numeric_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 히스토그램 + KDE
        ax1 = axes[0, 0]
        ax1.hist(data, bins=30, alpha=0.7, color='skyblue', density=True, label='히스토그램')

        # KDE 추가
        try:
            from scipy import stats
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax1.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        except ImportError:
            pass

        ax1.set_title(f'{col} 분포', fontweight='bold')
        ax1.set_xlabel(col)
        ax1.set_ylabel('밀도')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 박스플롯
        ax2 = axes[0, 1]
        box_plot = ax2.boxplot(data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        ax2.set_title(f'{col} 박스플롯', fontweight='bold')
        ax2.set_ylabel(col)
        ax2.grid(True, alpha=0.3)

        # 3. 바이올린 플롯
        ax3 = axes[0, 2]
        ax3.violinplot(data, showmeans=True, showmedians=True)
        ax3.set_title(f'{col} 바이올린플롯', fontweight='bold')
        ax3.set_ylabel(col)
        ax3.grid(True, alpha=0.3)

        # 4. ECDF (경험적 누적분포함수)
        ax4 = axes[1, 0]
        sorted_data = np.sort(data)
        ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.step(sorted_data, ecdf_y, where='post', linewidth=2)
        ax4.set_title(f'{col} ECDF', fontweight='bold')
        ax4.set_xlabel(col)
        ax4.set_ylabel('누적확률')
        ax4.grid(True, alpha=0.3)

        # 5. 정규분포와 비교
        ax5 = axes[1, 1]
        ax5.hist(data, bins=30, alpha=0.7, color='skyblue', density=True, label='실제 분포')

        # 정규분포 오버레이
        mu, sigma = data.mean(), data.std()
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = ((1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mu)/sigma)**2))
        ax5.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'정규분포 (μ={mu:.2f}, σ={sigma:.2f})')

        ax5.set_title(f'{col} vs 정규분포', fontweight='bold')
        ax5.set_xlabel(col)
        ax5.set_ylabel('밀도')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 통계 요약
        ax6 = axes[1, 2]
        ax6.axis('off')

        # 기술통계
        stats_text = f"""
        기술통계 요약:
        평균: {data.mean():.3f}
        중앙값: {data.median():.3f}
        표준편차: {data.std():.3f}
        분산: {data.var():.3f}
        최솟값: {data.min():.3f}
        최댓값: {data.max():.3f}
        왜도: {data.skew():.3f}
        첨도: {data.kurtosis():.3f}
        IQR: {data.quantile(0.75) - data.quantile(0.25):.3f}
        """

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle(f'{col} 분포 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'statistical_distribution_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

        # 통계 정보 저장
        results["statistics"].setdefault("distributions", {})[col] = {
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "variance": float(data.var()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "min": float(data.min()),
            "max": float(data.max()),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25))
        }

def _create_qq_plots(df: pd.DataFrame, numeric_columns: List[str],
                    output_dir: str, results: Dict[str, Any]):
    """Q-Q 플롯 생성"""

    for col in numeric_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        try:
            from scipy import stats
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy:
            # 1. 정규분포 Q-Q 플롯
            ax1 = axes[0, 0]
            stats.probplot(data, dist="norm", plot=ax1)
            ax1.set_title(f'{col} vs 정규분포 Q-Q', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 2. 균등분포 Q-Q 플롯
            ax2 = axes[0, 1]
            stats.probplot(data, dist="uniform", plot=ax2)
            ax2.set_title(f'{col} vs 균등분포 Q-Q', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 3. 지수분포 Q-Q 플롯 (양수 데이터만)
            ax3 = axes[1, 0]
            if data.min() > 0:
                stats.probplot(data, dist="expon", plot=ax3)
                ax3.set_title(f'{col} vs 지수분포 Q-Q', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '지수분포는 양수 데이터만 가능',
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title(f'{col} vs 지수분포 Q-Q (불가능)', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 4. 정규성 검정 결과
            ax4 = axes[1, 1]
            ax4.axis('off')

            # Shapiro-Wilk 검정
            if len(data) <= 5000:  # 샘플 크기 제한
                shapiro_stat, shapiro_p = stats.shapiro(data)
                shapiro_result = "정규분포" if shapiro_p > 0.05 else "비정규분포"
            else:
                shapiro_stat, shapiro_p = None, None
                shapiro_result = "샘플 크기가 너무 큼"

            # Kolmogorov-Smirnov 검정
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            ks_result = "정규분포" if ks_p > 0.05 else "비정규분포"

            # Anderson-Darling 검정
            ad_stat, ad_critical, ad_sig = stats.anderson(data, dist='norm')
            ad_result = "정규분포" if ad_stat < ad_critical[2] else "비정규분포"  # 5% 유의수준

            test_text = f"""
            정규성 검정 결과:

            Shapiro-Wilk 검정:
            통계량: {shapiro_stat:.4f if shapiro_stat else 'N/A'}
            p-값: {shapiro_p:.4f if shapiro_p else 'N/A'}
            결과: {shapiro_result}

            Kolmogorov-Smirnov 검정:
            통계량: {ks_stat:.4f}
            p-값: {ks_p:.4f}
            결과: {ks_result}

            Anderson-Darling 검정:
            통계량: {ad_stat:.4f}
            임계값(5%): {ad_critical[2]:.4f}
            결과: {ad_result}
            """

            ax4.text(0.1, 0.9, test_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            # 통계 정보 저장
            results["statistics"].setdefault("normality_tests", {})[col] = {
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat) if shapiro_stat else None,
                    "p_value": float(shapiro_p) if shapiro_p else None,
                    "is_normal": shapiro_p > 0.05 if shapiro_p else None
                },
                "kolmogorov_smirnov": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_p > 0.05
                },
                "anderson_darling": {
                    "statistic": float(ad_stat),
                    "critical_value_5pct": float(ad_critical[2]),
                    "is_normal": ad_stat < ad_critical[2]
                }
            }

        else:
            # scipy가 없는 경우 간단한 Q-Q 플롯
            _simple_qq_plot(data, axes, col)

        plt.suptitle(f'{col} Q-Q 플롯 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'statistical_qq_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _simple_qq_plot(data: pd.Series, axes, col: str):
    """간단한 Q-Q 플롯 (scipy 없이)"""

    # 1. 정규분포 Q-Q 플롯 (간단한 버전)
    ax1 = axes[0, 0]
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # 이론적 분위수 (정규분포)
    theoretical_quantiles = np.array([(i - 0.5) / n for i in range(1, n + 1)])

    # 정규분포의 역누적분포함수 근사
    theoretical_normal = []
    for p in theoretical_quantiles:
        if p <= 0.5:
            z = -np.sqrt(-2 * np.log(p))
        else:
            z = np.sqrt(-2 * np.log(1 - p))
        theoretical_normal.append(z)

    theoretical_normal = np.array(theoretical_normal)

    # 데이터 표준화
    standardized_data = (sorted_data - data.mean()) / data.std()

    ax1.scatter(theoretical_normal, standardized_data, alpha=0.6)
    ax1.plot(theoretical_normal, theoretical_normal, 'r-', linewidth=2)
    ax1.set_title(f'{col} vs 정규분포 Q-Q (간단)', fontweight='bold')
    ax1.set_xlabel('이론적 분위수')
    ax1.set_ylabel('표본 분위수')
    ax1.grid(True, alpha=0.3)

    # 나머지 서브플롯들은 빈 상태로 두고 메시지 표시
    for i, ax in enumerate([axes[0, 1], axes[1, 0], axes[1, 1]]):
        ax.text(0.5, 0.5, 'scipy 패키지가 필요합니다',
               transform=ax.transAxes, ha='center', va='center')
        titles = ['균등분포 Q-Q', '지수분포 Q-Q', '정규성 검정']
        ax.set_title(f'{titles[i]} (사용불가)', fontweight='bold')

def _create_residual_plots(df: pd.DataFrame, numeric_columns: List[str],
                          target_column: str, output_dir: str, results: Dict[str, Any]):
    """잔차 플롯 생성 (회귀 분석용)"""

    target_data = df[target_column].dropna()
    if len(target_data) == 0:
        return

    for col in numeric_columns:
        if col == target_column:
            continue

        feature_data = df[col].dropna()

        # 공통 인덱스 찾기
        common_idx = df[col].notna() & df[target_column].notna()
        if common_idx.sum() < 3:
            continue

        x = df.loc[common_idx, col]
        y = df.loc[common_idx, target_column]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 선형 회귀 및 잔차
        ax1 = axes[0, 0]

        # 간단한 선형 회귀
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        residuals = y - y_pred

        ax1.scatter(x, y, alpha=0.6, label='실제 데이터')
        ax1.plot(x, y_pred, 'r-', linewidth=2, label=f'회귀선 (y = {coeffs[0]:.3f}x + {coeffs[1]:.3f})')
        ax1.set_title(f'{col} vs {target_column} 회귀', fontweight='bold')
        ax1.set_xlabel(col)
        ax1.set_ylabel(target_column)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 잔차 vs 예측값
        ax2 = axes[0, 1]
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_title('잔차 vs 예측값', fontweight='bold')
        ax2.set_xlabel('예측값')
        ax2.set_ylabel('잔차')
        ax2.grid(True, alpha=0.3)

        # 3. 잔차 히스토그램
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=20, alpha=0.7, color='lightgreen', density=True)
        ax3.set_title('잔차 분포', fontweight='bold')
        ax3.set_xlabel('잔차')
        ax3.set_ylabel('밀도')
        ax3.grid(True, alpha=0.3)

        # 정규분포 오버레이
        mu_resid, sigma_resid = residuals.mean(), residuals.std()
        x_resid = np.linspace(residuals.min(), residuals.max(), 100)
        y_resid_norm = ((1/(sigma_resid * np.sqrt(2*np.pi))) *
                       np.exp(-0.5*((x_resid - mu_resid)/sigma_resid)**2))
        ax3.plot(x_resid, y_resid_norm, 'r-', linewidth=2, label='정규분포')
        ax3.legend()

        # 4. 회귀 통계
        ax4 = axes[1, 1]
        ax4.axis('off')

        # R² 계산
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 기타 통계
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        stats_text = f"""
        회귀 분석 결과:

        R² (결정계수): {r_squared:.4f}
        RMSE: {rmse:.4f}
        MAE: {mae:.4f}

        회귀 계수:
        기울기: {coeffs[0]:.4f}
        절편: {coeffs[1]:.4f}

        잔차 통계:
        평균: {residuals.mean():.4f}
        표준편차: {residuals.std():.4f}
        최소: {residuals.min():.4f}
        최대: {residuals.max():.4f}
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle(f'{col} → {target_column} 잔차 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'statistical_residual_{col}_{target_column}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

        # 통계 정보 저장
        results["statistics"].setdefault("regression_analysis", {})[f"{col}_to_{target_column}"] = {
            "r_squared": float(r_squared),
            "rmse": float(rmse),
            "mae": float(mae),
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std())
        }

def _create_probability_plots(df: pd.DataFrame, numeric_columns: List[str],
                             output_dir: str, results: Dict[str, Any]):
    """확률 플롯 생성"""

    for col in numeric_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 누적분포함수 (CDF)
        ax1 = axes[0, 0]
        sorted_data = np.sort(data)
        cdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax1.step(sorted_data, cdf_y, where='post', linewidth=2, label='경험적 CDF')

        # 이론적 정규분포 CDF
        x_theory = np.linspace(data.min(), data.max(), 100)
        mean, std = data.mean(), data.std()

        # 정규분포 CDF 근사
        cdf_theory = []
        for x_val in x_theory:
            z = (x_val - mean) / std
            # 정규분포 CDF 근사 (erf 함수 없이)
            if z >= 0:
                cdf_val = 0.5 + 0.5 * (1 - np.exp(-z * z))
            else:
                cdf_val = 0.5 - 0.5 * (1 - np.exp(-z * z))
            cdf_theory.append(cdf_val)

        ax1.plot(x_theory, cdf_theory, 'r-', linewidth=2, label='이론적 정규분포 CDF')
        ax1.set_title(f'{col} 누적분포함수', fontweight='bold')
        ax1.set_xlabel(col)
        ax1.set_ylabel('누적확률')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 확률밀도함수 (PDF)
        ax2 = axes[0, 1]
        ax2.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', label='히스토그램')

        # 이론적 정규분포 PDF
        y_theory = ((1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_theory - mean)/std)**2))
        ax2.plot(x_theory, y_theory, 'r-', linewidth=2, label='이론적 정규분포 PDF')

        ax2.set_title(f'{col} 확률밀도함수', fontweight='bold')
        ax2.set_xlabel(col)
        ax2.set_ylabel('확률밀도')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 분위수 플롯
        ax3 = axes[1, 0]
        percentiles = np.arange(5, 100, 5)
        empirical_quantiles = np.percentile(data, percentiles)

        # 이론적 정규분포 분위수
        theoretical_quantiles = []
        for p in percentiles:
            z_score = _inverse_normal_cdf(p / 100)
            theoretical_quantiles.append(mean + std * z_score)

        ax3.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
        ax3.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
                [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--', linewidth=2)
        ax3.set_title(f'{col} 분위수 비교', fontweight='bold')
        ax3.set_xlabel('이론적 분위수 (정규분포)')
        ax3.set_ylabel('경험적 분위수')
        ax3.grid(True, alpha=0.3)

        # 4. 확률 통계 요약
        ax4 = axes[1, 1]
        ax4.axis('off')

        # 확률 구간 계산
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        p_in_1sigma = np.sum((data >= mean - std) & (data <= mean + std)) / len(data)
        p_in_2sigma = np.sum((data >= mean - 2*std) & (data <= mean + 2*std)) / len(data)

        prob_text = f"""
        확률 분석 요약:

        분위수:
        25% 분위수: {q25:.3f}
        50% 분위수(중앙값): {q50:.3f}
        75% 분위수: {q75:.3f}

        정규분포 적합성:
        1σ 구간 확률: {p_in_1sigma:.3f} (이론값: 0.683)
        2σ 구간 확률: {p_in_2sigma:.3f} (이론값: 0.954)

        극값:
        하위 5%: {np.percentile(data, 5):.3f}
        상위 5%: {np.percentile(data, 95):.3f}
        하위 1%: {np.percentile(data, 1):.3f}
        상위 1%: {np.percentile(data, 99):.3f}
        """

        ax4.text(0.1, 0.9, prob_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

        plt.suptitle(f'{col} 확률 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'statistical_probability_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

        # 통계 정보 저장
        results["statistics"].setdefault("probability_analysis", {})[col] = {
            "percentiles": {
                "5th": float(np.percentile(data, 5)),
                "25th": float(q25),
                "50th": float(q50),
                "75th": float(q75),
                "95th": float(np.percentile(data, 95))
            },
            "sigma_intervals": {
                "1_sigma_probability": float(p_in_1sigma),
                "2_sigma_probability": float(p_in_2sigma)
            }
        }

def _create_confidence_plots(df: pd.DataFrame, numeric_columns: List[str],
                            output_dir: str, results: Dict[str, Any]):
    """신뢰구간 플롯 생성"""

    for col in numeric_columns:
        data = df[col].dropna()
        if len(data) < 3:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 평균의 신뢰구간
        ax1 = axes[0, 0]

        # 부트스트랩 방법으로 신뢰구간 계산
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)
        mean_estimate = np.mean(data)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        ax1.hist(bootstrap_means, bins=50, alpha=0.7, color='lightblue', density=True)
        ax1.axvline(mean_estimate, color='red', linestyle='-', linewidth=2, label=f'표본평균: {mean_estimate:.3f}')
        ax1.axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'95% CI 하한: {ci_lower:.3f}')
        ax1.axvline(ci_upper, color='orange', linestyle='--', linewidth=2, label=f'95% CI 상한: {ci_upper:.3f}')
        ax1.fill_between([ci_lower, ci_upper], [0, 0], [ax1.get_ylim()[1], ax1.get_ylim()[1]],
                        alpha=0.3, color='orange')

        ax1.set_title(f'{col} 평균의 95% 신뢰구간', fontweight='bold')
        ax1.set_xlabel('평균값')
        ax1.set_ylabel('밀도')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 표준편차의 신뢰구간
        ax2 = axes[0, 1]

        bootstrap_stds = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))

        bootstrap_stds = np.array(bootstrap_stds)
        std_estimate = np.std(data, ddof=1)
        std_ci_lower = np.percentile(bootstrap_stds, 2.5)
        std_ci_upper = np.percentile(bootstrap_stds, 97.5)

        ax2.hist(bootstrap_stds, bins=50, alpha=0.7, color='lightgreen', density=True)
        ax2.axvline(std_estimate, color='red', linestyle='-', linewidth=2, label=f'표본표준편차: {std_estimate:.3f}')
        ax2.axvline(std_ci_lower, color='purple', linestyle='--', linewidth=2, label=f'95% CI 하한: {std_ci_lower:.3f}')
        ax2.axvline(std_ci_upper, color='purple', linestyle='--', linewidth=2, label=f'95% CI 상한: {std_ci_upper:.3f}')

        ax2.set_title(f'{col} 표준편차의 95% 신뢰구간', fontweight='bold')
        ax2.set_xlabel('표준편차')
        ax2.set_ylabel('밀도')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 분위수의 신뢰구간
        ax3 = axes[1, 0]

        percentiles_to_estimate = [25, 50, 75]
        percentile_estimates = []
        percentile_cis = []

        for p in percentiles_to_estimate:
            bootstrap_percentiles = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_percentiles.append(np.percentile(bootstrap_sample, p))

            estimate = np.percentile(data, p)
            ci_low = np.percentile(bootstrap_percentiles, 2.5)
            ci_high = np.percentile(bootstrap_percentiles, 97.5)

            percentile_estimates.append(estimate)
            percentile_cis.append((ci_low, ci_high))

        x_pos = range(len(percentiles_to_estimate))
        ax3.errorbar(x_pos, percentile_estimates,
                    yerr=[[est - ci[0] for est, ci in zip(percentile_estimates, percentile_cis)],
                          [ci[1] - est for est, ci in zip(percentile_estimates, percentile_cis)]],
                    fmt='o', capsize=5, capthick=2, markersize=8)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{p}%' for p in percentiles_to_estimate])
        ax3.set_title(f'{col} 분위수의 95% 신뢰구간', fontweight='bold')
        ax3.set_xlabel('분위수')
        ax3.set_ylabel('값')
        ax3.grid(True, alpha=0.3)

        # 4. 신뢰구간 요약
        ax4 = axes[1, 1]
        ax4.axis('off')

        ci_text = f"""
        95% 신뢰구간 요약:

        평균:
        추정값: {mean_estimate:.3f}
        하한: {ci_lower:.3f}
        상한: {ci_upper:.3f}
        폭: {ci_upper - ci_lower:.3f}

        표준편차:
        추정값: {std_estimate:.3f}
        하한: {std_ci_lower:.3f}
        상한: {std_ci_upper:.3f}
        폭: {std_ci_upper - std_ci_lower:.3f}

        분위수:
        25%: {percentile_estimates[0]:.3f} [{percentile_cis[0][0]:.3f}, {percentile_cis[0][1]:.3f}]
        50%: {percentile_estimates[1]:.3f} [{percentile_cis[1][0]:.3f}, {percentile_cis[1][1]:.3f}]
        75%: {percentile_estimates[2]:.3f} [{percentile_cis[2][0]:.3f}, {percentile_cis[2][1]:.3f}]
        """

        ax4.text(0.1, 0.9, ci_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        plt.suptitle(f'{col} 신뢰구간 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'statistical_confidence_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

        # 통계 정보 저장
        results["statistics"].setdefault("confidence_intervals", {})[col] = {
            "mean": {
                "estimate": float(mean_estimate),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_width": float(ci_upper - ci_lower)
            },
            "std": {
                "estimate": float(std_estimate),
                "ci_lower": float(std_ci_lower),
                "ci_upper": float(std_ci_upper),
                "ci_width": float(std_ci_upper - std_ci_lower)
            },
            "percentiles": {
                f"{p}th": {
                    "estimate": float(est),
                    "ci_lower": float(ci[0]),
                    "ci_upper": float(ci[1])
                } for p, est, ci in zip(percentiles_to_estimate, percentile_estimates, percentile_cis)
            }
        }

def _inverse_normal_cdf(p: float) -> float:
    """정규분포의 역누적분포함수 근사 (Box-Muller 변환 없이)"""
    if p <= 0 or p >= 1:
        raise ValueError("확률은 0과 1 사이여야 합니다")

    if p == 0.5:
        return 0.0

    # Beasley-Springer-Moro 알고리즘 간단 버전
    if p < 0.5:
        sign = -1
        p = 1 - p
    else:
        sign = 1

    # 근사 공식
    t = np.sqrt(-2 * np.log(p))
    z = t - ((2.515517 + 0.802853 * t + 0.010328 * t * t) /
             (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))

    return sign * z

def main():
    """
    메인 실행 함수 - 통계 시각화의 진입점
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
        numeric_columns = params.get('numeric_columns')
        target_column = params.get('target_column')
        plot_types = params.get('plot_types', ['distribution', 'qq', 'probability'])
        output_dir = params.get('output_dir', 'visualizations')

        if not numeric_columns:
            # 자동으로 수치형 컬럼 찾기
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise ValueError("수치형 컬럼을 찾을 수 없습니다")

        # 통계 시각화 생성
        result = create_statistical_plots(df, numeric_columns, target_column, plot_types, output_dir)

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