#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive Statistics Analysis for Lightweight Analysis MCP
경량 분석 MCP용 기술통계 분석 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params, get_numeric_columns
except ImportError:
    # Fallback implementations
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

    def get_numeric_columns(df: pd.DataFrame, min_unique: int = 2) -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if df[col].nunique() >= min_unique]


def calculate_comprehensive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics
    포괄적인 기술통계 계산
    """
    # Select columns to analyze
    if columns is None:
        columns = get_numeric_columns(df)
    else:
        # Validate provided columns
        available_numeric = get_numeric_columns(df)
        columns = [col for col in columns if col in available_numeric]

    if not columns:
        return {
            "error": "분석 가능한 수치형 컬럼이 없습니다",
            "available_columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()
        }

    subset_df = df[columns]
    results = {}

    # Basic descriptive statistics
    results["basic_stats"] = subset_df.describe().to_dict()

    # Advanced statistics
    advanced_stats = {}
    for col in columns:
        clean_data = subset_df[col].dropna()
        if len(clean_data) > 1:
            advanced_stats[col] = {
                "variance": float(clean_data.var()),
                "std_dev": float(clean_data.std()),
                "skewness": float(stats.skew(clean_data)),
                "kurtosis": float(stats.kurtosis(clean_data)),
                "coefficient_of_variation": float(clean_data.std() / clean_data.mean()) if clean_data.mean() != 0 else None,
                "range": float(clean_data.max() - clean_data.min()),
                "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
                "mad": float(stats.median_abs_deviation(clean_data)),  # Median Absolute Deviation
                "valid_count": len(clean_data),
                "missing_count": int(subset_df[col].isnull().sum())
            }
        else:
            advanced_stats[col] = {
                "error": "계산에 충분한 데이터가 없습니다",
                "valid_count": len(clean_data),
                "missing_count": int(subset_df[col].isnull().sum())
            }

    results["advanced_stats"] = advanced_stats

    # Distribution analysis
    distribution_analysis = {}
    for col in columns:
        clean_data = subset_df[col].dropna()
        if len(clean_data) > 3:
            distribution_analysis[col] = analyze_distribution(clean_data)

    results["distribution_analysis"] = distribution_analysis

    # Summary statistics across all columns
    summary_stats = calculate_summary_across_columns(subset_df)
    results["summary_stats"] = summary_stats

    # High variation variables
    high_variation_vars = identify_high_variation_variables(advanced_stats)
    results["high_variation_vars"] = high_variation_vars

    return results


def analyze_distribution(data: pd.Series) -> Dict[str, Any]:
    """
    Analyze distribution characteristics
    분포 특성 분석
    """
    analysis = {}

    # Shape characteristics
    skewness = stats.skew(data)
    kurtosis_val = stats.kurtosis(data)

    analysis["shape"] = {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_val),
        "distribution_type": classify_distribution_type(skewness, kurtosis_val)
    }

    # Quartile analysis
    quartiles = data.quantile([0.25, 0.5, 0.75])
    analysis["quartiles"] = {
        "q1": float(quartiles[0.25]),
        "median": float(quartiles[0.5]),
        "q3": float(quartiles[0.75]),
        "iqr": float(quartiles[0.75] - quartiles[0.25])
    }

    # Outlier detection
    q1, q3 = quartiles[0.25], quartiles[0.75]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    analysis["outliers"] = {
        "count": len(outliers),
        "percentage": round((len(outliers) / len(data)) * 100, 2),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }

    # Normality tests (for larger samples)
    if len(data) >= 20:
        try:
            # Shapiro-Wilk test (for smaller samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                analysis["normality_tests"] = {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    }
                }
            else:
                # For larger samples, use Anderson-Darling
                anderson_result = stats.anderson(data, dist='norm')
                analysis["normality_tests"] = {
                    "anderson_darling": {
                        "statistic": float(anderson_result.statistic),
                        "critical_values": anderson_result.critical_values.tolist(),
                        "is_normal": anderson_result.statistic < anderson_result.critical_values[2]  # 5% significance level
                    }
                }
        except Exception:
            analysis["normality_tests"] = {"error": "정규성 검정 실행 불가"}

    return analysis


def classify_distribution_type(skewness: float, kurtosis: float) -> str:
    """
    Classify distribution type based on skewness and kurtosis
    왜도와 첨도를 기반으로 분포 유형 분류
    """
    # Skewness classification
    if abs(skewness) < 0.5:
        skew_type = "symmetric"
    elif skewness > 0.5:
        skew_type = "right_skewed"
    else:
        skew_type = "left_skewed"

    # Kurtosis classification
    if abs(kurtosis) < 0.5:
        kurt_type = "mesokurtic"  # normal kurtosis
    elif kurtosis > 0.5:
        kurt_type = "leptokurtic"  # heavy tails
    else:
        kurt_type = "platykurtic"  # light tails

    return f"{skew_type}_{kurt_type}"


def calculate_summary_across_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics across all columns
    모든 컬럼에 대한 요약 통계 계산
    """
    summary = {}

    # Mean statistics
    means = df.mean()
    summary["mean"] = {
        "min": float(means.min()),
        "max": float(means.max()),
        "average": float(means.mean()),
        "std": float(means.std())
    }

    # Standard deviation statistics
    stds = df.std()
    summary["std"] = {
        "min": float(stds.min()),
        "max": float(stds.max()),
        "average": float(stds.mean()),
        "range": float(stds.max() - stds.min())
    }

    # Missing data summary
    missing_counts = df.isnull().sum()
    summary["missing_data"] = {
        "total_missing": int(missing_counts.sum()),
        "columns_with_missing": int((missing_counts > 0).sum()),
        "max_missing_in_column": int(missing_counts.max()),
        "avg_missing_per_column": float(missing_counts.mean())
    }

    return summary


def identify_high_variation_variables(advanced_stats: Dict[str, Any], cv_threshold: float = 1.0) -> List[str]:
    """
    Identify variables with high coefficient of variation
    높은 변동계수를 가진 변수 식별
    """
    high_variation = []

    for col, stats_dict in advanced_stats.items():
        if isinstance(stats_dict, dict) and "coefficient_of_variation" in stats_dict:
            cv = stats_dict["coefficient_of_variation"]
            if cv is not None and cv > cv_threshold:
                high_variation.append(col)

    return high_variation


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['file_path'])

        # Load data
        df = load_data(options['file_path'])

        # Get basic data info
        data_info = get_data_info(df)

        # Perform descriptive statistics analysis
        columns = options.get('columns', None)
        analysis_results = calculate_comprehensive_stats(df, columns)

        if "error" in analysis_results:
            error_result = {
                "success": False,
                "error": analysis_results["error"],
                "analysis_type": "descriptive_statistics",
                **{k: v for k, v in analysis_results.items() if k != "error"}
            }
            output_results(error_result)
            return

        # Create final result
        analyzed_columns = columns if columns else get_numeric_columns(df)
        result = create_analysis_result(
            analysis_type="descriptive_statistics",
            data_info=data_info,
            results=analysis_results,
            summary=f"기술통계 분석 완료 - {len(analyzed_columns)}개 수치형 변수 분석"
        )

        # Output results
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "descriptive_statistics"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()