#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Data Analysis Script for Lightweight Analysis MCP
경량 분석 MCP용 기본 데이터 분석 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # Fallback implementation if shared utils not available
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from file"""
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic data information"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        """Create standardized result"""
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        """Output results as JSON"""
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        """Validate required parameters"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def perform_basic_analysis(df: pd.DataFrame, include_distribution: bool = True) -> Dict[str, Any]:
    """
    Perform basic data analysis
    기본 데이터 분석 수행
    """
    results = {}

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        results["numeric_summary"] = numeric_stats.to_dict()

        # Additional statistics
        results["numeric_additional"] = {
            "skewness": df[numeric_cols].skew().to_dict(),
            "kurtosis": df[numeric_cols].kurtosis().to_dict(),
            "variance": df[numeric_cols].var().to_dict()
        }

    # Basic info for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_info = {}
        for col in categorical_cols:
            categorical_info[col] = {
                "unique_count": int(df[col].nunique()),
                "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "most_frequent_count": int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0,
                "null_count": int(df[col].isnull().sum())
            }
        results["categorical_summary"] = categorical_info

    # Missing data summary
    missing_data = df.isnull().sum()
    results["missing_summary"] = {
        "total_missing": int(missing_data.sum()),
        "missing_percentage": round((missing_data.sum() / (len(df) * len(df.columns))) * 100, 2),
        "columns_with_missing": missing_data[missing_data > 0].to_dict(),
        "complete_rows": int(len(df) - df.isnull().any(axis=1).sum())
    }

    # Distribution analysis if requested
    if include_distribution and len(numeric_cols) > 0:
        distribution_analysis = {}
        for col in numeric_cols:
            if df[col].notna().sum() > 1:  # Need at least 2 non-null values
                distribution_analysis[col] = {
                    "quartiles": {
                        "q1": float(df[col].quantile(0.25)),
                        "q2": float(df[col].quantile(0.5)),  # median
                        "q3": float(df[col].quantile(0.75))
                    },
                    "outliers_iqr": detect_outliers_iqr(df[col]),
                    "distribution_shape": classify_distribution(df[col])
                }
        results["distribution_analysis"] = distribution_analysis

    return results


def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    """
    Detect outliers using IQR method
    IQR 방법으로 이상치 탐지
    """
    clean_series = series.dropna()
    if len(clean_series) < 4:
        return {"outlier_count": 0, "outlier_percentage": 0}

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]

    return {
        "outlier_count": len(outliers),
        "outlier_percentage": round((len(outliers) / len(clean_series)) * 100, 2),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }


def classify_distribution(series: pd.Series) -> str:
    """
    Classify distribution shape
    분포 형태 분류
    """
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return "insufficient_data"

    skewness = clean_series.skew()

    if abs(skewness) < 0.5:
        return "approximately_normal"
    elif skewness > 0.5:
        return "right_skewed"
    elif skewness < -0.5:
        return "left_skewed"
    else:
        return "unknown"


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

        # Perform analysis
        include_distribution = options.get('include_distribution', True)
        analysis_results = perform_basic_analysis(df, include_distribution)

        # Create final result
        result = create_analysis_result(
            analysis_type="basic_analysis",
            data_info=data_info,
            results=analysis_results,
            summary=f"기본 데이터 분석 완료 - {df.shape[0]}행 × {df.shape[1]}열"
        )

        # Output results
        output_results(result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "basic_analysis"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()