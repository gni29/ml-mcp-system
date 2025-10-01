#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Dashboard Generator Module
데이터 대시보드 생성 모듈

이 모듈은 종합적인 데이터 분석 대시보드를 생성합니다.
주요 기능:
- 다양한 대시보드 유형 (overview, statistical, exploratory)
- HTML 인터랙티브 대시보드
- 맞춤형 섹션 구성
- 자동 인사이트 생성
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

def create_dashboard(df: pd.DataFrame, dashboard_type: str = 'overview',
                    include_sections: List[str] = None,
                    target_column: Optional[str] = None,
                    output_file: str = 'data_dashboard.html') -> Dict[str, Any]:
    """
    데이터 대시보드 생성

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    dashboard_type : str
        대시보드 유형 ('overview', 'statistical', 'exploratory', 'custom')
    include_sections : List[str]
        포함할 섹션들
    target_column : str, optional
        타겟 변수 (분석 중심)
    output_file : str
        출력 HTML 파일명

    Returns:
    --------
    Dict[str, Any]
        대시보드 생성 결과
    """

    if include_sections is None:
        include_sections = ['summary_stats', 'distributions', 'correlations']

    try:
        # 기본 정보 분석
        data_info = _analyze_basic_info(df)

        results = {
            "success": True,
            "dashboard_type": dashboard_type,
            "sections_included": include_sections,
            "data_info": data_info,
            "output_file": output_file
        }

        # 섹션별 분석 수행
        sections_content = {}

        if 'summary_stats' in include_sections:
            sections_content['summary_stats'] = _generate_summary_stats(df)

        if 'distributions' in include_sections:
            sections_content['distributions'] = _generate_distributions_section(df)

        if 'correlations' in include_sections:
            sections_content['correlations'] = _generate_correlations_section(df)

        if 'missing_data' in include_sections:
            sections_content['missing_data'] = _generate_missing_data_section(df)

        if 'outliers' in include_sections:
            sections_content['outliers'] = _generate_outliers_section(df)

        if 'time_series' in include_sections:
            sections_content['time_series'] = _generate_timeseries_section(df)

        # 타겟 변수 분석 (지정된 경우)
        if target_column and target_column in df.columns:
            sections_content['target_analysis'] = _generate_target_analysis(df, target_column)

        # HTML 대시보드 생성
        html_content = _generate_html_dashboard(data_info, sections_content, dashboard_type, target_column)

        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 결과 통계
        results.update({
            "section_count": len(sections_content),
            "chart_count": sum(section.get('chart_count', 0) for section in sections_content.values()),
            "interactive_elements": sum(section.get('interactive_count', 0) for section in sections_content.values()),
            "insights_generated": sum(len(section.get('insights', [])) for section in sections_content.values())
        })

        return results

    except Exception as e:
        return {
            "error": f"대시보드 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def _analyze_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """기본 데이터 정보 분석"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    return {
        "shape": df.shape,
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "missing_values": df.isnull().sum().sum(),
        "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "duplicate_rows": df.duplicated().sum()
    }

def _generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """기술통계 요약 섹션 생성"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "수치형 컬럼이 없습니다", "chart_count": 0}

    # 기술통계 계산
    desc_stats = df[numeric_cols].describe()

    # 추가 통계 계산
    additional_stats = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            additional_stats[col] = {
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "variance": float(data.var()),
                "coefficient_of_variation": float(data.std() / data.mean()) if data.mean() != 0 else 0
            }

    # 인사이트 생성
    insights = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            cv = abs(additional_stats[col]["coefficient_of_variation"])
            skew = abs(additional_stats[col]["skewness"])

            if cv > 1:
                insights.append(f"{col}: 높은 변동성 (CV={cv:.2f})")
            if skew > 1:
                insights.append(f"{col}: 비대칭 분포 (왜도={skew:.2f})")

    return {
        "descriptive_stats": desc_stats.to_dict(),
        "additional_stats": additional_stats,
        "insights": insights,
        "chart_count": 1,
        "interactive_count": 0
    }

def _generate_distributions_section(df: pd.DataFrame) -> Dict[str, Any]:
    """분포 분석 섹션 생성"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "수치형 컬럼이 없습니다", "chart_count": 0}

    # 분포 특성 분석
    distribution_analysis = {}
    insights = []

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 3:
            # 정규성 검정 (간단한 방법)
            skewness = data.skew()
            kurtosis = data.kurtosis()

            # 분포 타입 추정
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                dist_type = "정규분포에 가까움"
            elif skewness > 1:
                dist_type = "우편향 (긴 꼬리가 오른쪽)"
            elif skewness < -1:
                dist_type = "좌편향 (긴 꼬리가 왼쪽)"
            else:
                dist_type = "비정규분포"

            distribution_analysis[col] = {
                "type": dist_type,
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "outlier_count": _count_outliers(data)
            }

            if distribution_analysis[col]["outlier_count"] > 0:
                insights.append(f"{col}: {distribution_analysis[col]['outlier_count']}개 이상치 탐지")

    return {
        "distribution_analysis": distribution_analysis,
        "insights": insights,
        "chart_count": len(numeric_cols),
        "interactive_count": 1
    }

def _generate_correlations_section(df: pd.DataFrame) -> Dict[str, Any]:
    """상관관계 분석 섹션 생성"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {"error": "상관관계 분석을 위해 최소 2개의 수치형 컬럼이 필요합니다", "chart_count": 0}

    # 상관관계 계산
    corr_matrix = df[numeric_cols].corr()

    # 강한 상관관계 찾기
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": float(corr_val)
                })

    # 인사이트 생성
    insights = []
    if strong_correlations:
        insights.append(f"강한 상관관계 {len(strong_correlations)}쌍 발견")
        top_corr = max(strong_correlations, key=lambda x: abs(x["correlation"]))
        insights.append(f"최고 상관관계: {top_corr['var1']} ↔ {top_corr['var2']} ({top_corr['correlation']:.3f})")

    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "strong_correlations": strong_correlations,
        "insights": insights,
        "chart_count": 1,
        "interactive_count": 1
    }

def _generate_missing_data_section(df: pd.DataFrame) -> Dict[str, Any]:
    """결측값 분석 섹션 생성"""
    missing_info = df.isnull().sum()
    missing_percentage = (missing_info / len(df)) * 100

    missing_analysis = {}
    insights = []

    for col in df.columns:
        if missing_info[col] > 0:
            missing_analysis[col] = {
                "count": int(missing_info[col]),
                "percentage": float(missing_percentage[col])
            }

            if missing_percentage[col] > 50:
                insights.append(f"{col}: 결측값이 50% 이상 ({missing_percentage[col]:.1f}%)")
            elif missing_percentage[col] > 20:
                insights.append(f"{col}: 상당한 결측값 ({missing_percentage[col]:.1f}%)")

    if not missing_analysis:
        insights.append("결측값이 없는 완전한 데이터셋입니다")

    return {
        "missing_analysis": missing_analysis,
        "total_missing": int(missing_info.sum()),
        "insights": insights,
        "chart_count": 1,
        "interactive_count": 0
    }

def _generate_outliers_section(df: pd.DataFrame) -> Dict[str, Any]:
    """이상치 분석 섹션 생성"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "수치형 컬럼이 없습니다", "chart_count": 0}

    outlier_analysis = {}
    insights = []

    for col in numeric_cols:
        data = df[col].dropna()
        outlier_count = _count_outliers(data)
        outlier_percentage = (outlier_count / len(data)) * 100

        outlier_analysis[col] = {
            "count": outlier_count,
            "percentage": float(outlier_percentage)
        }

        if outlier_percentage > 10:
            insights.append(f"{col}: 이상치 비율이 높음 ({outlier_percentage:.1f}%)")

    return {
        "outlier_analysis": outlier_analysis,
        "insights": insights,
        "chart_count": len(numeric_cols),
        "interactive_count": 0
    }

def _generate_timeseries_section(df: pd.DataFrame) -> Dict[str, Any]:
    """시계열 분석 섹션 생성"""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    if not datetime_cols:
        # 날짜 형태의 문자열 컬럼 찾기
        date_like_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                try:
                    pd.to_datetime(df[col].head())
                    date_like_cols.append(col)
                except:
                    pass

        if not date_like_cols:
            return {"error": "시계열 컬럼을 찾을 수 없습니다", "chart_count": 0}

        datetime_cols = date_like_cols

    # 기본 시계열 분석
    timeseries_analysis = {}
    insights = []

    for col in datetime_cols:
        try:
            dates = pd.to_datetime(df[col])
            date_range = dates.max() - dates.min()
            frequency_analysis = dates.dt.freq if hasattr(dates.dt, 'freq') else "불규칙"

            timeseries_analysis[col] = {
                "start_date": dates.min().isoformat(),
                "end_date": dates.max().isoformat(),
                "date_range_days": date_range.days,
                "frequency": str(frequency_analysis)
            }

            insights.append(f"{col}: {date_range.days}일 범위의 시계열 데이터")

        except Exception as e:
            timeseries_analysis[col] = {"error": str(e)}

    return {
        "timeseries_analysis": timeseries_analysis,
        "insights": insights,
        "chart_count": len(datetime_cols),
        "interactive_count": 1
    }

def _generate_target_analysis(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """타겟 변수 분석 섹션 생성"""
    if target_column not in df.columns:
        return {"error": f"타겟 컬럼 '{target_column}'을 찾을 수 없습니다", "chart_count": 0}

    target_data = df[target_column]
    insights = []

    # 타겟 변수 타입 분석
    if pd.api.types.is_numeric_dtype(target_data):
        # 수치형 타겟
        target_analysis = {
            "type": "numeric",
            "stats": {
                "mean": float(target_data.mean()),
                "std": float(target_data.std()),
                "min": float(target_data.min()),
                "max": float(target_data.max()),
                "unique_values": int(target_data.nunique())
            }
        }

        if target_data.nunique() < 10:
            insights.append(f"{target_column}: 분류 문제로 적합 ({target_data.nunique()}개 클래스)")
        else:
            insights.append(f"{target_column}: 회귀 문제로 적합 (연속값)")

    else:
        # 범주형 타겟
        value_counts = target_data.value_counts()
        target_analysis = {
            "type": "categorical",
            "value_counts": value_counts.to_dict(),
            "unique_values": int(target_data.nunique())
        }

        if target_data.nunique() == 2:
            insights.append(f"{target_column}: 이진 분류 문제")
        else:
            insights.append(f"{target_column}: 다중 클래스 분류 문제 ({target_data.nunique()}개 클래스)")

        # 클래스 불균형 체크
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 5:
                insights.append(f"{target_column}: 클래스 불균형 존재 (비율: {imbalance_ratio:.1f}:1)")

    return {
        "target_analysis": target_analysis,
        "insights": insights,
        "chart_count": 2,
        "interactive_count": 1
    }

def _count_outliers(data: pd.Series) -> int:
    """IQR 방법으로 이상치 개수 계산"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return len(data[(data < lower_bound) | (data > upper_bound)])

def _generate_html_dashboard(data_info: Dict[str, Any], sections_content: Dict[str, Any],
                           dashboard_type: str, target_column: Optional[str]) -> str:
    """HTML 대시보드 생성"""

    # HTML 템플릿 시작
    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데이터 분석 대시보드 - {dashboard_type.upper()}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .section h2 {{
            color: #007bff;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .insights {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .insights h3 {{
            margin-top: 0;
            color: #0056b3;
        }}
        .insights ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .table th {{
            background-color: #007bff;
            color: white;
        }}
        .table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 데이터 분석 대시보드</h1>
            <p>대시보드 유형: <strong>{dashboard_type.upper()}</strong></p>
            {f"<p>타겟 변수: <strong>{target_column}</strong></p>" if target_column else ""}
            <p>생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""

    # 데이터 개요 섹션
    html += f"""
        <div class="section">
            <h2>📋 데이터 개요</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{data_info['shape'][0]:,}</div>
                    <div>총 행 수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data_info['shape'][1]}</div>
                    <div>총 컬럼 수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data_info['numeric_columns']}</div>
                    <div>수치형 컬럼</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data_info['categorical_columns']}</div>
                    <div>범주형 컬럼</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data_info['missing_percentage']:.1f}%</div>
                    <div>결측값 비율</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data_info['memory_usage_mb']:.1f} MB</div>
                    <div>메모리 사용량</div>
                </div>
            </div>
        </div>
"""

    # 각 섹션 내용 추가
    for section_name, section_data in sections_content.items():
        if 'error' in section_data:
            continue

        section_title = {
            'summary_stats': '📈 기술통계 요약',
            'distributions': '📊 분포 분석',
            'correlations': '🔗 상관관계 분석',
            'missing_data': '❓ 결측값 분석',
            'outliers': '🎯 이상치 분석',
            'time_series': '⏰ 시계열 분석',
            'target_analysis': '🎯 타겟 변수 분석'
        }.get(section_name, section_name.replace('_', ' ').title())

        html += f"""
        <div class="section">
            <h2>{section_title}</h2>
"""

        # 인사이트 추가
        if 'insights' in section_data and section_data['insights']:
            html += f"""
            <div class="insights">
                <h3>💡 주요 인사이트</h3>
                <ul>
"""
            for insight in section_data['insights']:
                html += f"                    <li>{insight}</li>\n"
            html += """
                </ul>
            </div>
"""

        # 섹션별 상세 내용 추가
        if section_name == 'summary_stats' and 'descriptive_stats' in section_data:
            html += _add_summary_stats_table(section_data['descriptive_stats'])

        elif section_name == 'correlations' and 'strong_correlations' in section_data:
            html += _add_correlations_table(section_data['strong_correlations'])

        elif section_name == 'missing_data' and 'missing_analysis' in section_data:
            html += _add_missing_data_table(section_data['missing_analysis'])

        html += """
        </div>
"""

    # HTML 마무리
    html += """
    </div>
</body>
</html>
"""

    return html

def _add_summary_stats_table(desc_stats: Dict[str, Any]) -> str:
    """기술통계 테이블 추가"""
    if not desc_stats:
        return ""

    html = """
            <table class="table">
                <thead>
                    <tr>
                        <th>변수</th>
                        <th>개수</th>
                        <th>평균</th>
                        <th>표준편차</th>
                        <th>최솟값</th>
                        <th>25%</th>
                        <th>50%</th>
                        <th>75%</th>
                        <th>최댓값</th>
                    </tr>
                </thead>
                <tbody>
"""

    # 첫 번째 컬럼의 통계를 기준으로 행 생성
    first_col = list(desc_stats.keys())[0]
    stat_names = list(desc_stats[first_col].keys())

    for stat_name in stat_names:
        if stat_name == 'count':
            continue

        html += f"                    <tr>\n"
        html += f"                        <td><strong>{stat_name}</strong></td>\n"

        for col_name in desc_stats.keys():
            if stat_name in desc_stats[col_name]:
                value = desc_stats[col_name][stat_name]
                html += f"                        <td>{value:.2f}</td>\n"
            else:
                html += f"                        <td>-</td>\n"

        html += f"                    </tr>\n"

    html += """
                </tbody>
            </table>
"""
    return html

def _add_correlations_table(strong_correlations: List[Dict[str, Any]]) -> str:
    """강한 상관관계 테이블 추가"""
    if not strong_correlations:
        return "<p>강한 상관관계(|r| > 0.7)가 발견되지 않았습니다.</p>"

    html = """
            <table class="table">
                <thead>
                    <tr>
                        <th>변수 1</th>
                        <th>변수 2</th>
                        <th>상관계수</th>
                        <th>강도</th>
                    </tr>
                </thead>
                <tbody>
"""

    for corr in strong_correlations[:10]:  # 상위 10개만 표시
        html += f"""
                    <tr>
                        <td>{corr['var1']}</td>
                        <td>{corr['var2']}</td>
                        <td>{corr['correlation']:.3f}</td>
                        <td>{"매우 강함" if abs(corr['correlation']) >= 0.9 else "강함"}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
"""
    return html

def _add_missing_data_table(missing_analysis: Dict[str, Any]) -> str:
    """결측값 테이블 추가"""
    if not missing_analysis:
        return "<p>결측값이 없습니다.</p>"

    html = """
            <table class="table">
                <thead>
                    <tr>
                        <th>변수</th>
                        <th>결측값 개수</th>
                        <th>결측값 비율</th>
                    </tr>
                </thead>
                <tbody>
"""

    # 결측값 비율로 정렬
    sorted_missing = sorted(missing_analysis.items(), key=lambda x: x[1]['percentage'], reverse=True)

    for col_name, missing_info in sorted_missing:
        html += f"""
                    <tr>
                        <td>{col_name}</td>
                        <td>{missing_info['count']}</td>
                        <td>{missing_info['percentage']:.1f}%</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
"""
    return html

def main():
    """
    메인 실행 함수 - 대시보드 생성의 진입점
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
        dashboard_type = params.get('dashboard_type', 'overview')
        include_sections = params.get('include_sections', ['summary_stats', 'distributions', 'correlations'])
        target_column = params.get('target_column')
        output_file = params.get('output_file', 'data_dashboard.html')

        # 대시보드 생성
        result = create_dashboard(df, dashboard_type, include_sections, target_column, output_file)

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