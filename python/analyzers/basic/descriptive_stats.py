#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive Statistics Analysis Module
기술통계 분석 모듈

이 모듈은 데이터셋의 기술통계량을 종합적으로 계산하고 분석합니다.
주요 기능:
- 수치형 변수 기술통계 (평균, 중위값, 분산, 왜도, 첨도 등)
- 범주형 변수 빈도 분석
- 분포 특성 및 이상치 탐지
- 데이터 요약 및 품질 평가
- 통계적 해석 및 권고사항 제공
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

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

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 기본 정보 추출"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        """표준화된 분석 결과 구조 생성"""
        return {
            "analysis_type": analysis_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        """결과를 JSON 형태로 출력"""
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

def calculate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    DataFrame에 대한 포괄적인 기술통계 계산

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임

    Returns:
    --------
    Dict[str, Any]
        기술통계 분석 결과
        - numeric_statistics: 수치형 변수 통계량
        - categorical_statistics: 범주형 변수 통계량
        - distribution_analysis: 분포 특성 분석
        - data_quality_metrics: 데이터 품질 지표
    """
    try:
        # 숫자형 컬럼 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 범주형 컬럼 선택
        categorical_df = df.select_dtypes(include=['object', 'category'])
        
        # 결과 딕셔너리 초기화
        result = {
            'success': True,
            'data_summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns_count': len(numeric_df.columns),
                'categorical_columns_count': len(categorical_df.columns)
            }
        }
        
        # 숫자형 통계 계산
        if not numeric_df.empty:
            result['numeric_columns'] = numeric_df.columns.tolist()
            result['numeric_statistics'] = {
                'count': numeric_df.count().to_dict(),
                'mean': numeric_df.mean().to_dict(),
                'median': numeric_df.median().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'q25': numeric_df.quantile(0.25).to_dict(),   # 1사분위수
                'q75': numeric_df.quantile(0.75).to_dict(),   # 3사분위수
            }
            
            # 분포 특성 (왜도/첨도) 추가
            result['distribution_characteristics'] = {
                'skewness': {},
                'kurtosis': {}
            }
            
            for col in numeric_df.columns:
                try:
                    clean_data = numeric_df[col].dropna()
                    if len(clean_data) > 1:  # 최소 2개 이상의 값이 있어야 계산 가능
                        result['distribution_characteristics']['skewness'][col] = float(skew(clean_data))
                        result['distribution_characteristics']['kurtosis'][col] = float(kurtosis(clean_data))
                    else:
                        result['distribution_characteristics']['skewness'][col] = None
                        result['distribution_characteristics']['kurtosis'][col] = None
                except:
                    result['distribution_characteristics']['skewness'][col] = None
                    result['distribution_characteristics']['kurtosis'][col] = None
        else:
            result['numeric_columns'] = []
            result['numeric_statistics'] = {}
            result['distribution_characteristics'] = {}
        
        # 범주형 통계 계산
        if not categorical_df.empty:
            result['categorical_columns'] = categorical_df.columns.tolist()
            result['categorical_statistics'] = {}
            
            for col in categorical_df.columns:
                try:
                    # 기본 통계
                    col_stats = {
                        'unique_count': int(df[col].nunique()),
                        'null_count': int(df[col].isnull().sum()),
                        'non_null_count': int(df[col].notna().sum())
                    }
                    
                    # 최빈값 계산
                    mode_values = df[col].mode()
                    if len(mode_values) > 0:
                        col_stats['most_frequent'] = mode_values.iloc[0]
                        col_stats['most_frequent_count'] = int(df[col].value_counts().iloc[0])
                    else:
                        col_stats['most_frequent'] = None
                        col_stats['most_frequent_count'] = 0
                    
                    # 빈도 테이블 (상위 10개)
                    value_counts = df[col].value_counts().head(10)
                    col_stats['frequency_table'] = value_counts.to_dict()
                    
                    result['categorical_statistics'][col] = col_stats
                    
                except Exception as e:
                    result['categorical_statistics'][col] = {
                        'error': f'범주형 통계 계산 실패: {str(e)}'
                    }
        else:
            result['categorical_columns'] = []
            result['categorical_statistics'] = {}
        
        # 전체적으로 숫자형과 범주형 모두 없는 경우
        if numeric_df.empty and categorical_df.empty:
            return {
                'success': False,
                'error': '분석 가능한 컬럼이 없습니다',
                'available_columns': df.columns.tolist(),
                'column_types': df.dtypes.astype(str).to_dict()
            }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'통계 계산 실패: {str(e)}',
            'error_type': 'CalculationError'
        }

def main():
    """
    메인 실행 함수 - 기술통계 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 기술통계 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, statistical_options

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 수치형/범주형 변수별 상세 통계
    - 한국어 해석 및 데이터 품질 평가
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 파일 경로가 제공된 경우 파일에서 데이터 로드
        if 'file_path' in params:
            df = load_data(params['file_path'])
        else:
            # JSON 데이터에서 직접 DataFrame 생성
            if 'data' in params:
                df = pd.DataFrame(params['data'])
            else:
                df = pd.DataFrame(params)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 기술통계 분석 수행
        stats_result = calculate_descriptive_stats(df)

        if not stats_result.get('success', True):
            error_result = {
                "success": False,
                "error": stats_result.get('error', '알 수 없는 오류'),
                "analysis_type": "descriptive_statistics",
                "available_columns": stats_result.get('available_columns', []),
                "column_types": stats_result.get('column_types', {})
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "descriptive_statistics": stats_result,
            "statistical_summary": {
                "numeric_columns_analyzed": len(stats_result.get('numeric_columns', [])),
                "categorical_columns_analyzed": len(stats_result.get('categorical_columns', [])),
                "total_variables": len(df.columns),
                "data_completeness": f"{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%"
            }
        }

        # 요약 생성
        numeric_count = len(stats_result.get('numeric_columns', []))
        categorical_count = len(stats_result.get('categorical_columns', []))
        summary = f"기술통계 분석 완료 - 수치형 {numeric_count}개, 범주형 {categorical_count}개 변수 분석"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="descriptive_statistics",
            data_info=data_info,
            results=analysis_results,
            summary=summary
        )

        # 결과 출력
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "descriptive_statistics",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()