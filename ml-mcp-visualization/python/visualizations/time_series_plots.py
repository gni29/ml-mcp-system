#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Plots Visualization Module
시계열 플롯 시각화 모듈

이 모듈은 시계열 데이터의 포괄적인 시각화를 제공합니다.
주요 기능:
- 기본 시계열 라인 플롯
- 이동평균 및 추세 분석
- 계절성 분해 (Seasonal Decomposition)
- 자기상관함수 (ACF/PACF) 플롯
- 예측 구간이 포함된 시계열 차트
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

def create_time_series_plots(df: pd.DataFrame, date_column: str, value_columns: List[str],
                            plot_types: List[str] = None, rolling_window: int = 30,
                            output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    시계열 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        시계열 데이터프레임
    date_column : str
        날짜/시간 컬럼명
    value_columns : List[str]
        시각화할 값 컬럼들
    plot_types : List[str]
        생성할 플롯 유형들 ['line', 'area', 'seasonal_decompose', 'rolling_stats', 'autocorrelation']
    rolling_window : int
        이동평균 윈도우 크기
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        시계열 시각화 결과
    """

    if plot_types is None:
        plot_types = ['line', 'seasonal_decompose']

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 기본 유효성 검사
    if date_column not in df.columns:
        return {
            "error": f"날짜 컬럼 '{date_column}'을 찾을 수 없습니다",
            "available_columns": df.columns.tolist()
        }

    for col in value_columns:
        if col not in df.columns:
            return {
                "error": f"값 컬럼 '{col}'을 찾을 수 없습니다",
                "available_columns": df.columns.tolist()
            }

    try:
        # 날짜 컬럼 변환
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column], errors='coerce')

        # 날짜 변환 실패 확인
        if df_ts[date_column].isnull().all():
            return {
                "error": f"날짜 컬럼 '{date_column}'을 datetime으로 변환할 수 없습니다",
                "sample_values": df[date_column].head().tolist()
            }

        # 날짜로 정렬
        df_ts = df_ts.sort_values(date_column).reset_index(drop=True)

        results = {
            "success": True,
            "date_column": date_column,
            "value_columns": value_columns,
            "plot_types": plot_types,
            "data_points": len(df_ts),
            "date_range": {
                "start": df_ts[date_column].min().isoformat(),
                "end": df_ts[date_column].max().isoformat(),
                "duration_days": (df_ts[date_column].max() - df_ts[date_column].min()).days
            },
            "generated_files": []
        }

        # 스타일 설정
        sns.set_style("whitegrid")
        plt.style.use('default')

        # 각 플롯 유형별 생성
        for plot_type in plot_types:
            if plot_type == 'line':
                _create_line_plots(df_ts, date_column, value_columns, output_dir, results)
            elif plot_type == 'area':
                _create_area_plots(df_ts, date_column, value_columns, output_dir, results)
            elif plot_type == 'seasonal_decompose':
                _create_seasonal_decomposition(df_ts, date_column, value_columns, output_dir, results)
            elif plot_type == 'rolling_stats':
                _create_rolling_stats(df_ts, date_column, value_columns, rolling_window, output_dir, results)
            elif plot_type == 'autocorrelation':
                _create_autocorrelation_plots(df_ts, date_column, value_columns, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"시계열 시각화 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_line_plots(df: pd.DataFrame, date_column: str, value_columns: List[str],
                      output_dir: str, results: Dict[str, Any]):
    """기본 시계열 라인 플롯 생성"""

    fig, axes = plt.subplots(len(value_columns), 1, figsize=(14, 4*len(value_columns)))
    if len(value_columns) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(value_columns)))

    for i, col in enumerate(value_columns):
        ax = axes[i]

        # 라인 플롯
        ax.plot(df[date_column], df[col], color=colors[i], linewidth=2, alpha=0.8)

        # 축과 제목 설정
        ax.set_title(f'{col} 시계열', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        if i == len(value_columns) - 1:  # 마지막 서브플롯만 x축 레이블
            ax.set_xlabel('날짜')

        # 격자 및 스타일
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # 기본 통계 정보 추가
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'평균: {mean_val:.2f}')
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', alpha=0.5, label=f'+1σ: {mean_val + std_val:.2f}')
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', alpha=0.5, label=f'-1σ: {mean_val - std_val:.2f}')
        ax.legend(fontsize=8)

    plt.suptitle('시계열 라인 플롯', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = Path(output_dir) / 'timeseries_line.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_area_plots(df: pd.DataFrame, date_column: str, value_columns: List[str],
                      output_dir: str, results: Dict[str, Any]):
    """면적 차트 생성"""

    if len(value_columns) == 1:
        # 단일 변수 면적 차트
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(df[date_column], df[value_columns[0]], alpha=0.7, color='skyblue')
        ax.plot(df[date_column], df[value_columns[0]], color='navy', linewidth=2)
        ax.set_title(f'{value_columns[0]} 면적 차트', fontsize=14, fontweight='bold')
        ax.set_ylabel(value_columns[0])
        ax.set_xlabel('날짜')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    else:
        # 다중 변수 스택 면적 차트
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 1. 개별 면적 차트
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_columns)))
        for i, col in enumerate(value_columns):
            ax1.fill_between(df[date_column], df[col], alpha=0.6, color=colors[i], label=col)

        ax1.set_title('개별 변수 면적 차트', fontsize=12, fontweight='bold')
        ax1.set_ylabel('값')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. 스택 면적 차트 (값이 모두 양수인 경우)
        if all(df[col].min() >= 0 for col in value_columns):
            ax2.stackplot(df[date_column], *[df[col] for col in value_columns],
                         labels=value_columns, alpha=0.7, colors=colors)
            ax2.set_title('스택 면적 차트', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left')
        else:
            # 음수 값이 있는 경우 정규화된 면적 차트
            df_normalized = df[value_columns].div(df[value_columns].sum(axis=1), axis=0)
            ax2.stackplot(df[date_column], *[df_normalized[col] for col in value_columns],
                         labels=value_columns, alpha=0.7, colors=colors)
            ax2.set_title('정규화된 스택 면적 차트 (비율)', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left')
            ax2.set_ylabel('비율')

        ax2.set_xlabel('날짜')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = Path(output_dir) / 'timeseries_area.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_seasonal_decomposition(df: pd.DataFrame, date_column: str, value_columns: List[str],
                                  output_dir: str, results: Dict[str, Any]):
    """계절성 분해 플롯 생성"""

    try:
        # statsmodels가 있는 경우 사용
        from statsmodels.tsa.seasonal import seasonal_decompose
        use_statsmodels = True
    except ImportError:
        use_statsmodels = False

    for col in value_columns:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 데이터 준비
        ts_data = df.set_index(date_column)[col].dropna()

        if len(ts_data) < 2:
            continue

        # 원본 시계열
        axes[0].plot(ts_data.index, ts_data.values, color='blue', linewidth=2)
        axes[0].set_title(f'{col} - 원본 시계열', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        if use_statsmodels and len(ts_data) >= 4:
            # statsmodels를 사용한 계절성 분해
            try:
                # 주기 추정 (간단한 방법)
                period = min(12, len(ts_data) // 2) if len(ts_data) >= 24 else len(ts_data) // 2
                if period < 2:
                    period = 2

                decomposition = seasonal_decompose(ts_data, model='additive', period=period, extrapolate_trend='freq')

                # 추세
                axes[1].plot(decomposition.trend.index, decomposition.trend.values, color='red', linewidth=2)
                axes[1].set_title('추세 (Trend)', fontweight='bold')
                axes[1].grid(True, alpha=0.3)

                # 계절성
                axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, color='green', linewidth=2)
                axes[2].set_title('계절성 (Seasonal)', fontweight='bold')
                axes[2].grid(True, alpha=0.3)

                # 잔차
                axes[3].plot(decomposition.resid.index, decomposition.resid.values, color='orange', linewidth=1)
                axes[3].set_title('잔차 (Residual)', fontweight='bold')
                axes[3].grid(True, alpha=0.3)

                # 분해 결과 저장
                results.setdefault("seasonal_decomposition", {})[col] = {
                    "trend_variance": float(decomposition.trend.var()),
                    "seasonal_variance": float(decomposition.seasonal.var()),
                    "residual_variance": float(decomposition.resid.var()),
                    "period_used": period
                }

            except Exception as e:
                results.setdefault("seasonal_decomposition_errors", {})[col] = str(e)
                _simple_decomposition(ts_data, axes[1:], col)
        else:
            # 간단한 분해 방법
            _simple_decomposition(ts_data, axes[1:], col)

        # x축 레이블 회전
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f'{col} 계절성 분해', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'timeseries_decompose_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _simple_decomposition(ts_data: pd.Series, axes: List, col: str):
    """간단한 계절성 분해 (statsmodels 없이)"""

    # 이동평균을 사용한 추세
    window = min(7, len(ts_data) // 4) if len(ts_data) >= 14 else 3
    trend = ts_data.rolling(window=window, center=True).mean()

    axes[0].plot(trend.index, trend.values, color='red', linewidth=2)
    axes[0].set_title('추세 (이동평균)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 계절성 (잔차의 주기적 패턴)
    detrended = ts_data - trend
    seasonal = detrended.groupby(detrended.index.dayofyear % 30).transform('mean')

    axes[1].plot(seasonal.index, seasonal.values, color='green', linewidth=2)
    axes[1].set_title('계절성 (추정)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 잔차
    residual = ts_data - trend - seasonal

    axes[2].plot(residual.index, residual.values, color='orange', linewidth=1)
    axes[2].set_title('잔차', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

def _create_rolling_stats(df: pd.DataFrame, date_column: str, value_columns: List[str],
                         rolling_window: int, output_dir: str, results: Dict[str, Any]):
    """이동 통계 플롯 생성"""

    fig, axes = plt.subplots(len(value_columns), 1, figsize=(14, 4*len(value_columns)))
    if len(value_columns) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(value_columns)))

    for i, col in enumerate(value_columns):
        ax = axes[i]

        # 원본 데이터
        ax.plot(df[date_column], df[col], color=colors[i], alpha=0.5, linewidth=1, label='원본')

        # 이동평균
        rolling_mean = df[col].rolling(window=rolling_window).mean()
        ax.plot(df[date_column], rolling_mean, color='red', linewidth=2, label=f'{rolling_window}일 이동평균')

        # 이동표준편차
        rolling_std = df[col].rolling(window=rolling_window).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std

        ax.fill_between(df[date_column], upper_band, lower_band, alpha=0.2, color='gray', label='±2σ 구간')

        # 볼린저 밴드 스타일
        ax.plot(df[date_column], upper_band, color='orange', linestyle='--', alpha=0.7, label='상단밴드')
        ax.plot(df[date_column], lower_band, color='orange', linestyle='--', alpha=0.7, label='하단밴드')

        ax.set_title(f'{col} 이동 통계 ({rolling_window}일 윈도우)', fontweight='bold')
        ax.set_ylabel(col)
        if i == len(value_columns) - 1:
            ax.set_xlabel('날짜')

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # 통계 정보 저장
        results.setdefault("rolling_statistics", {})[col] = {
            "window_size": rolling_window,
            "final_mean": float(rolling_mean.iloc[-1]) if not rolling_mean.empty else None,
            "final_std": float(rolling_std.iloc[-1]) if not rolling_std.empty else None,
            "trend_direction": "상승" if rolling_mean.iloc[-1] > rolling_mean.iloc[-rolling_window//2] else "하락" if rolling_mean.iloc[-1] < rolling_mean.iloc[-rolling_window//2] else "횡보"
        }

    plt.suptitle('이동 통계 분석', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = Path(output_dir) / 'timeseries_rolling_stats.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def _create_autocorrelation_plots(df: pd.DataFrame, date_column: str, value_columns: List[str],
                                 output_dir: str, results: Dict[str, Any]):
    """자기상관함수 플롯 생성"""

    try:
        # statsmodels가 있는 경우
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        use_statsmodels = True
    except ImportError:
        use_statsmodels = False

    for col in value_columns:
        if use_statsmodels:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # 데이터 준비
            ts_data = df[col].dropna()
            if len(ts_data) < 10:
                continue

            try:
                # ACF 플롯
                plot_acf(ts_data, ax=ax1, lags=min(40, len(ts_data)//4), alpha=0.05)
                ax1.set_title(f'{col} - 자기상관함수 (ACF)', fontweight='bold')

                # PACF 플롯
                plot_pacf(ts_data, ax=ax2, lags=min(20, len(ts_data)//8), alpha=0.05)
                ax2.set_title(f'{col} - 편자기상관함수 (PACF)', fontweight='bold')

                # 상관계수 계산
                acf_values = acf(ts_data, nlags=min(20, len(ts_data)//4))
                pacf_values = pacf(ts_data, nlags=min(10, len(ts_data)//8))

                results.setdefault("autocorrelation", {})[col] = {
                    "lag_1_acf": float(acf_values[1]) if len(acf_values) > 1 else None,
                    "lag_1_pacf": float(pacf_values[1]) if len(pacf_values) > 1 else None,
                    "significant_lags_acf": [int(i) for i, val in enumerate(acf_values) if abs(val) > 0.2 and i > 0],
                    "significant_lags_pacf": [int(i) for i, val in enumerate(pacf_values) if abs(val) > 0.2 and i > 0]
                }

            except Exception as e:
                results.setdefault("autocorrelation_errors", {})[col] = str(e)
                _simple_autocorrelation(ts_data, ax1, ax2, col)

        else:
            # 간단한 자기상관 분석
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ts_data = df[col].dropna()
            _simple_autocorrelation(ts_data, ax1, ax2, col)

        plt.suptitle(f'{col} 자기상관 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_dir) / f'timeseries_autocorr_{col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        results["generated_files"].append(str(output_file))

def _simple_autocorrelation(ts_data: pd.Series, ax1, ax2, col: str):
    """간단한 자기상관 분석 (statsmodels 없이)"""

    max_lags = min(20, len(ts_data) // 4)
    lags = range(1, max_lags + 1)

    # 간단한 자기상관 계산
    autocorr = [ts_data.autocorr(lag=lag) for lag in lags]

    # ACF 플롯
    ax1.bar(lags, autocorr, alpha=0.7, color='skyblue')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='유의수준')
    ax1.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f'{col} - 자기상관함수 (간단 계산)', fontweight='bold')
    ax1.set_xlabel('지연(Lag)')
    ax1.set_ylabel('자기상관계수')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 차분 데이터의 자기상관 (PACF 대신)
    diff_data = ts_data.diff().dropna()
    if len(diff_data) > max_lags:
        diff_autocorr = [diff_data.autocorr(lag=lag) for lag in lags]
        ax2.bar(lags, diff_autocorr, alpha=0.7, color='lightgreen')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
        ax2.set_title(f'{col} - 1차 차분 자기상관', fontweight='bold')
        ax2.set_xlabel('지연(Lag)')
        ax2.set_ylabel('자기상관계수')
        ax2.grid(True, alpha=0.3)

def main():
    """
    메인 실행 함수 - 시계열 시각화의 진입점
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
        date_column = params.get('date_column')
        value_columns = params.get('value_columns')
        plot_types = params.get('plot_types', ['line', 'seasonal_decompose'])
        rolling_window = params.get('rolling_window', 30)
        output_dir = params.get('output_dir', 'visualizations')

        if not date_column:
            raise ValueError("date_column 매개변수가 필요합니다")
        if not value_columns:
            raise ValueError("value_columns 매개변수가 필요합니다")

        # 시계열 시각화 생성
        result = create_time_series_plots(df, date_column, value_columns, plot_types, rolling_window, output_dir)

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