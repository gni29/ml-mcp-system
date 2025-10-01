#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Plots Visualization Module
인터랙티브 플롯 시각화 모듈

이 모듈은 인터랙티브 데이터 시각화를 제공합니다.
주요 기능:
- 인터랙티브 산점도 (Interactive Scatter Plots)
- 인터랙티브 시계열 차트 (Interactive Time Series)
- 인터랙티브 히트맵 (Interactive Heatmaps)
- 3D 시각화 (3D Visualizations)
- 대시보드 스타일 차트 (Dashboard Charts)
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Plotly 가용성 확인
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Bokeh 가용성 확인
try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import HoverTool, ColorBar
    from bokeh.palettes import Viridis256, Set3
    from bokeh.layouts import gridplot
    HAS_BOKEH = True
except ImportError:
    HAS_BOKEH = False

def create_interactive_plots(df: pd.DataFrame,
                           numeric_columns: List[str] = None,
                           categorical_columns: List[str] = None,
                           plot_types: List[str] = None,
                           output_dir: str = 'visualizations') -> Dict[str, Any]:
    """
    인터랙티브 시각화 생성

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    numeric_columns : List[str], optional
        수치형 컬럼들
    categorical_columns : List[str], optional
        범주형 컬럼들
    plot_types : List[str], optional
        생성할 플롯 유형들 ['plotly_scatter', 'plotly_timeseries', 'plotly_3d', 'plotly_heatmap', 'bokeh_scatter']
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict[str, Any]
        인터랙티브 시각화 결과
    """

    if plot_types is None:
        plot_types = ['plotly_scatter', 'plotly_heatmap']

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        results = {
            "success": True,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "plot_types": plot_types,
            "data_points": len(df),
            "generated_files": [],
            "library_availability": {
                "plotly": HAS_PLOTLY,
                "bokeh": HAS_BOKEH
            }
        }

        # 각 플롯 유형별 생성
        for plot_type in plot_types:
            if plot_type.startswith('plotly_') and HAS_PLOTLY:
                if plot_type == 'plotly_scatter':
                    _create_plotly_scatter(df, numeric_columns, categorical_columns, output_dir, results)
                elif plot_type == 'plotly_timeseries':
                    _create_plotly_timeseries(df, numeric_columns, output_dir, results)
                elif plot_type == 'plotly_3d':
                    _create_plotly_3d(df, numeric_columns, categorical_columns, output_dir, results)
                elif plot_type == 'plotly_heatmap':
                    _create_plotly_heatmap(df, numeric_columns, output_dir, results)
                elif plot_type == 'plotly_dashboard':
                    _create_plotly_dashboard(df, numeric_columns, categorical_columns, output_dir, results)

            elif plot_type.startswith('bokeh_') and HAS_BOKEH:
                if plot_type == 'bokeh_scatter':
                    _create_bokeh_scatter(df, numeric_columns, categorical_columns, output_dir, results)
                elif plot_type == 'bokeh_timeseries':
                    _create_bokeh_timeseries(df, numeric_columns, output_dir, results)

            elif plot_type.startswith('matplotlib_interactive'):
                # Matplotlib 기반 인터랙티브 대안
                _create_matplotlib_interactive(df, numeric_columns, categorical_columns, output_dir, results)

        return results

    except Exception as e:
        return {
            "error": f"인터랙티브 시각화 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _create_plotly_scatter(df: pd.DataFrame, numeric_columns: List[str],
                          categorical_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Plotly 인터랙티브 산점도 생성"""

    if len(numeric_columns) < 2:
        return

    # 1. 기본 인터랙티브 산점도
    fig = go.Figure()

    x_col, y_col = numeric_columns[0], numeric_columns[1]
    color_col = categorical_columns[0] if categorical_columns else None

    if color_col:
        # 색상별 그룹
        for category in df[color_col].unique():
            mask = df[color_col] == category
            fig.add_trace(go.Scatter(
                x=df[mask][x_col],
                y=df[mask][y_col],
                mode='markers',
                name=str(category),
                text=[f'{color_col}: {cat}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}'
                      for cat, x, y in zip(df[mask][color_col], df[mask][x_col], df[mask][y_col])],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(size=8, opacity=0.7)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            text=[f'{x_col}: {x:.2f}<br>{y_col}: {y:.2f}'
                  for x, y in zip(df[x_col], df[y_col])],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(size=8, opacity=0.7, color='blue')
        ))

    fig.update_layout(
        title=f'인터랙티브 산점도: {x_col} vs {y_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest',
        width=800,
        height=600
    )

    output_file = Path(output_dir) / 'interactive_scatter_plotly.html'
    fig.write_html(str(output_file))
    results["generated_files"].append(str(output_file))

    # 2. 상관관계 매트릭스 (인터랙티브)
    if len(numeric_columns) >= 3:
        corr_matrix = df[numeric_columns].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig_corr.update_layout(
            title='인터랙티브 상관관계 매트릭스',
            width=600,
            height=600
        )

        output_file_corr = Path(output_dir) / 'interactive_correlation_plotly.html'
        fig_corr.write_html(str(output_file_corr))
        results["generated_files"].append(str(output_file_corr))

def _create_plotly_timeseries(df: pd.DataFrame, numeric_columns: List[str],
                             output_dir: str, results: Dict[str, Any]):
    """Plotly 인터랙티브 시계열 차트 생성"""

    # 날짜 컬럼 찾기
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue

    if not date_columns or not numeric_columns:
        return

    date_col = date_columns[0]
    df_ts = df.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
    df_ts = df_ts.dropna(subset=[date_col]).sort_values(date_col)

    fig = make_subplots(
        rows=len(numeric_columns), cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{col} 시계열' for col in numeric_columns],
        vertical_spacing=0.05
    )

    colors = px.colors.qualitative.Set1

    for i, col in enumerate(numeric_columns):
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_col],
                y=df_ts[col],
                mode='lines+markers',
                name=col,
                line=dict(color=colors[i % len(colors)]),
                hovertemplate=f'{col}: %{{y:.2f}}<br>날짜: %{{x}}<extra></extra>'
            ),
            row=i+1, col=1
        )

        # 이동평균 추가
        if len(df_ts) >= 7:
            rolling_mean = df_ts[col].rolling(window=min(7, len(df_ts)//3)).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_ts[date_col],
                    y=rolling_mean,
                    mode='lines',
                    name=f'{col} 이동평균',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    hovertemplate=f'{col} 이동평균: %{{y:.2f}}<br>날짜: %{{x}}<extra></extra>'
                ),
                row=i+1, col=1
            )

    fig.update_layout(
        title='인터랙티브 시계열 분석',
        height=300 * len(numeric_columns),
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="날짜", row=len(numeric_columns), col=1)

    output_file = Path(output_dir) / 'interactive_timeseries_plotly.html'
    fig.write_html(str(output_file))
    results["generated_files"].append(str(output_file))

def _create_plotly_3d(df: pd.DataFrame, numeric_columns: List[str],
                     categorical_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Plotly 3D 시각화 생성"""

    if len(numeric_columns) < 3:
        return

    x_col, y_col, z_col = numeric_columns[0], numeric_columns[1], numeric_columns[2]
    color_col = categorical_columns[0] if categorical_columns else None

    fig = go.Figure()

    if color_col:
        for category in df[color_col].unique():
            mask = df[color_col] == category
            fig.add_trace(go.Scatter3d(
                x=df[mask][x_col],
                y=df[mask][y_col],
                z=df[mask][z_col],
                mode='markers',
                name=str(category),
                text=[f'{color_col}: {cat}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}'
                      for cat, x, y, z in zip(df[mask][color_col], df[mask][x_col],
                                            df[mask][y_col], df[mask][z_col])],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=6,
                    opacity=0.8
                )
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            text=[f'{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}'
                  for x, y, z in zip(df[x_col], df[y_col], df[z_col])],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                size=6,
                opacity=0.8,
                color=df[z_col],
                colorscale='Viridis',
                colorbar=dict(title=z_col)
            )
        ))

    fig.update_layout(
        title=f'3D 인터랙티브 산점도: {x_col} × {y_col} × {z_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        width=800,
        height=800
    )

    output_file = Path(output_dir) / 'interactive_3d_plotly.html'
    fig.write_html(str(output_file))
    results["generated_files"].append(str(output_file))

def _create_plotly_heatmap(df: pd.DataFrame, numeric_columns: List[str],
                          output_dir: str, results: Dict[str, Any]):
    """Plotly 인터랙티브 히트맵 생성"""

    if len(numeric_columns) < 2:
        return

    # 1. 상관관계 히트맵
    corr_matrix = df[numeric_columns].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='%{x} vs %{y}<br>상관계수: %{z:.3f}<extra></extra>'
    ))

    fig_corr.update_layout(
        title='인터랙티브 상관관계 히트맵',
        width=600,
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )

    output_file_corr = Path(output_dir) / 'interactive_heatmap_correlation_plotly.html'
    fig_corr.write_html(str(output_file_corr))
    results["generated_files"].append(str(output_file_corr))

    # 2. 데이터 값 히트맵 (표준화된)
    if len(df) <= 100:  # 데이터가 너무 크지 않은 경우만
        df_normalized = df[numeric_columns].apply(lambda x: (x - x.mean()) / x.std())

        fig_data = go.Figure(data=go.Heatmap(
            z=df_normalized.T.values,
            x=list(range(len(df_normalized))),
            y=df_normalized.columns,
            colorscale='Viridis',
            hovertemplate='샘플: %{x}<br>변수: %{y}<br>표준화값: %{z:.2f}<extra></extra>'
        ))

        fig_data.update_layout(
            title='표준화된 데이터 히트맵',
            xaxis_title='샘플 인덱스',
            yaxis_title='변수',
            width=max(800, len(df) * 10),
            height=400
        )

        output_file_data = Path(output_dir) / 'interactive_heatmap_data_plotly.html'
        fig_data.write_html(str(output_file_data))
        results["generated_files"].append(str(output_file_data))

def _create_plotly_dashboard(df: pd.DataFrame, numeric_columns: List[str],
                            categorical_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Plotly 대시보드 스타일 차트 생성"""

    if len(numeric_columns) < 2:
        return

    # 2x2 서브플롯 구성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '분포 히스토그램',
            '산점도 매트릭스',
            '박스플롯',
            '시계열/트렌드'
        ],
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )

    # 1. 히스토그램 (첫 번째 수치 컬럼)
    col1 = numeric_columns[0]
    fig.add_trace(
        go.Histogram(x=df[col1], name=col1, nbinsx=30, opacity=0.7),
        row=1, col=1
    )

    # 2. 산점도 (첫 번째와 두 번째 수치 컬럼)
    if len(numeric_columns) >= 2:
        col2 = numeric_columns[1]
        color_col = categorical_columns[0] if categorical_columns else None

        if color_col:
            for category in df[color_col].unique():
                mask = df[color_col] == category
                fig.add_trace(
                    go.Scatter(
                        x=df[mask][col1],
                        y=df[mask][col2],
                        mode='markers',
                        name=str(category),
                        showlegend=True
                    ),
                    row=1, col=2
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[col1],
                    y=df[col2],
                    mode='markers',
                    name=f'{col1} vs {col2}',
                    marker=dict(color='blue', opacity=0.6)
                ),
                row=1, col=2
            )

    # 3. 박스플롯 (범주형이 있는 경우)
    if categorical_columns and len(numeric_columns) >= 1:
        cat_col = categorical_columns[0]
        for category in df[cat_col].unique():
            mask = df[cat_col] == category
            fig.add_trace(
                go.Box(
                    y=df[mask][col1],
                    name=str(category),
                    showlegend=False
                ),
                row=2, col=1
            )
    else:
        fig.add_trace(
            go.Box(y=df[col1], name=col1, showlegend=False),
            row=2, col=1
        )

    # 4. 트렌드 라인 (인덱스별 변화)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df[col1],
            mode='lines+markers',
            name=f'{col1} 트렌드',
            line=dict(width=2),
            showlegend=False
        ),
        row=2, col=2
    )

    # 이동평균 추가
    if len(df) >= 5:
        window = min(10, len(df) // 3)
        rolling_mean = df[col1].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rolling_mean,
                mode='lines',
                name=f'{col1} 이동평균',
                line=dict(dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_layout(
        title='인터랙티브 데이터 대시보드',
        height=800,
        showlegend=True
    )

    output_file = Path(output_dir) / 'interactive_dashboard_plotly.html'
    fig.write_html(str(output_file))
    results["generated_files"].append(str(output_file))

def _create_bokeh_scatter(df: pd.DataFrame, numeric_columns: List[str],
                         categorical_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Bokeh 인터랙티브 산점도 생성"""

    if len(numeric_columns) < 2:
        return

    x_col, y_col = numeric_columns[0], numeric_columns[1]

    # Bokeh 플롯 생성
    p = figure(
        title=f"Bokeh 인터랙티브 산점도: {x_col} vs {y_col}",
        x_axis_label=x_col,
        y_axis_label=y_col,
        width=700,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # 호버 툴 추가
    hover = HoverTool(tooltips=[
        (x_col, f"@{x_col}"),
        (y_col, f"@{y_col}"),
        ("Index", "$index")
    ])
    p.add_tools(hover)

    if categorical_columns:
        # 색상별 그룹
        color_col = categorical_columns[0]
        unique_categories = df[color_col].unique()
        colors = Set3[len(unique_categories)] if len(unique_categories) <= len(Set3) else Viridis256

        for i, category in enumerate(unique_categories):
            mask = df[color_col] == category
            color = colors[i % len(colors)]

            p.circle(
                df[mask][x_col],
                df[mask][y_col],
                size=8,
                color=color,
                alpha=0.7,
                legend_label=str(category)
            )
    else:
        p.circle(df[x_col], df[y_col], size=8, color="blue", alpha=0.7)

    p.legend.click_policy = "hide"

    output_file = Path(output_dir) / 'interactive_scatter_bokeh.html'
    output_file(str(output_file))
    save(p)

    results["generated_files"].append(str(output_file))

def _create_bokeh_timeseries(df: pd.DataFrame, numeric_columns: List[str],
                            output_dir: str, results: Dict[str, Any]):
    """Bokeh 인터랙티브 시계열 차트 생성"""

    # 날짜 컬럼 찾기
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue

    if not date_columns or not numeric_columns:
        return

    date_col = date_columns[0]
    df_ts = df.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
    df_ts = df_ts.dropna(subset=[date_col]).sort_values(date_col)

    # 시계열 플롯들
    plots = []
    colors = ["blue", "red", "green", "orange", "purple"]

    for i, col in enumerate(numeric_columns[:5]):  # 최대 5개
        p = figure(
            title=f"{col} 시계열",
            x_axis_label="날짜",
            y_axis_label=col,
            x_axis_type='datetime',
            width=700,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # 호버 툴
        hover = HoverTool(tooltips=[
            ("날짜", "@x{%F}"),
            (col, "@y"),
        ], formatters={"@x": "datetime"})
        p.add_tools(hover)

        # 라인 플롯
        p.line(df_ts[date_col], df_ts[col],
               line_width=2, color=colors[i % len(colors)], alpha=0.8)
        p.circle(df_ts[date_col], df_ts[col],
                size=4, color=colors[i % len(colors)], alpha=0.6)

        plots.append(p)

    # 그리드 레이아웃으로 배치
    grid = gridplot(plots, ncols=1)

    output_file = Path(output_dir) / 'interactive_timeseries_bokeh.html'
    output_file(str(output_file))
    save(grid)

    results["generated_files"].append(str(output_file))

def _create_matplotlib_interactive(df: pd.DataFrame, numeric_columns: List[str],
                                  categorical_columns: List[str], output_dir: str, results: Dict[str, Any]):
    """Matplotlib 기반 인터랙티브 대안 (정적이지만 상세한 정보 포함)"""

    if len(numeric_columns) < 2:
        return

    # 인터랙티브 대안으로 상세한 정보가 포함된 정적 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x_col, y_col = numeric_columns[0], numeric_columns[1]

    # 1. 상세 산점도
    ax1 = axes[0, 0]
    if categorical_columns:
        color_col = categorical_columns[0]
        for category in df[color_col].unique():
            mask = df[color_col] == category
            ax1.scatter(df[mask][x_col], df[mask][y_col], label=category, alpha=0.7)
        ax1.legend()
    else:
        ax1.scatter(df[x_col], df[y_col], alpha=0.7, c='blue')

    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title(f'{x_col} vs {y_col} (상세 산점도)')
    ax1.grid(True, alpha=0.3)

    # 통계 정보 텍스트 추가
    corr_coef = df[x_col].corr(df[y_col])
    ax1.text(0.05, 0.95, f'상관계수: {corr_coef:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

    # 2. 히스토그램 + 통계 (두 변수)
    ax2 = axes[0, 1]
    ax2.hist([df[x_col], df[y_col]], bins=20, alpha=0.7, label=[x_col, y_col])
    ax2.set_xlabel('값')
    ax2.set_ylabel('빈도')
    ax2.set_title('분포 비교')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 박스플롯 비교
    ax3 = axes[1, 0]
    if categorical_columns:
        cat_col = categorical_columns[0]
        categories = df[cat_col].unique()
        data_by_cat = [df[df[cat_col] == cat][x_col].dropna() for cat in categories]
        box_plot = ax3.boxplot(data_by_cat, labels=categories, patch_artist=True)

        # 색상 적용
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)

        ax3.set_title(f'{cat_col}별 {x_col} 분포')
    else:
        ax3.boxplot([df[x_col], df[y_col]], labels=[x_col, y_col])
        ax3.set_title('박스플롯 비교')

    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. 상세 통계 정보
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 기술통계 계산
    stats_text = f"""
    데이터 요약 통계:

    {x_col}:
    평균: {df[x_col].mean():.3f}
    표준편차: {df[x_col].std():.3f}
    최솟값: {df[x_col].min():.3f}
    최댓값: {df[x_col].max():.3f}
    중앙값: {df[x_col].median():.3f}

    {y_col}:
    평균: {df[y_col].mean():.3f}
    표준편차: {df[y_col].std():.3f}
    최솟값: {df[y_col].min():.3f}
    최댓값: {df[y_col].max():.3f}
    중앙값: {df[y_col].median():.3f}

    관계:
    상관계수: {corr_coef:.3f}
    총 데이터 포인트: {len(df)}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('상세 데이터 분석 (인터랙티브 대안)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = Path(output_dir) / 'interactive_alternative_matplotlib.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    results["generated_files"].append(str(output_file))

def main():
    """
    메인 실행 함수 - 인터랙티브 시각화의 진입점
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
        categorical_columns = params.get('categorical_columns')
        plot_types = params.get('plot_types', ['plotly_scatter', 'plotly_heatmap'])
        output_dir = params.get('output_dir', 'visualizations')

        # 인터랙티브 시각화 생성
        result = create_interactive_plots(df, numeric_columns, categorical_columns, plot_types, output_dir)

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