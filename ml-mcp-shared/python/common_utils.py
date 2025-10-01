#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for ML MCP Python modules
ML MCP 파이썬 모듈용 공통 유틸리티
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats
    다양한 파일 형식에서 데이터 로드
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    try:
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    except Exception as e:
        raise Exception(f"데이터 로드 실패: {str(e)}")


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset
    데이터셋의 기본 정보 추출
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def create_analysis_result(analysis_type: str, data_info: Dict[str, Any],
                          results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
    """
    Create standardized analysis result structure
    표준화된 분석 결과 구조 생성
    """
    return {
        "analysis_type": analysis_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_info": data_info,
        "summary": summary or f"{analysis_type} 분석 완료",
        **results
    }


def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON, handling numpy types
    numpy 타입을 처리하여 안전하게 JSON 직렬화
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def output_results(results: Dict[str, Any]):
    """
    Output results in the standard format expected by the MCP system
    MCP 시스템에서 예상하는 표준 형식으로 결과 출력
    """
    try:
        # Convert any non-serializable objects
        serializable_results = json.loads(
            json.dumps(results, default=safe_json_serialize, ensure_ascii=False)
        )

        # Output as JSON
        print(json.dumps(serializable_results, ensure_ascii=False, indent=2))

    except Exception as e:
        # Fallback error response
        error_result = {
            "error": True,
            "message": f"결과 출력 실패: {str(e)}",
            "analysis_type": results.get("analysis_type", "unknown")
        }
        print(json.dumps(error_result, ensure_ascii=False))


def validate_required_params(params: Dict[str, Any], required: List[str]):
    """
    Validate that required parameters are present
    필수 매개변수가 있는지 검증
    """
    missing = [param for param in required if param not in params or params[param] is None]
    if missing:
        raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def get_numeric_columns(df: pd.DataFrame, min_unique: int = 2) -> List[str]:
    """
    Get numeric columns with sufficient unique values for analysis
    분석에 충분한 고유값을 가진 수치형 컬럼 반환
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric_cols if df[col].nunique() >= min_unique]


def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> List[str]:
    """
    Get categorical columns with reasonable number of categories
    적절한 범주 수를 가진 범주형 컬럼 반환
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    return [col for col in categorical_cols if df[col].nunique() <= max_unique]


if __name__ == "__main__":
    # Test the utilities
    print("ML MCP 공통 유틸리티 테스트")

    # Create sample data for testing
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.1, 20.2, 30.3, 40.4, 50.5],
        'C': ['cat', 'dog', 'cat', 'bird', 'dog'],
        'D': pd.date_range('2023-01-01', periods=5)
    })

    info = get_data_info(sample_data)
    result = create_analysis_result("test_analysis", info, {"test": "success"})

    print("테스트 완료!")
    output_results(result)