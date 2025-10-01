#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Analysis and Outlier Detection Module
분포 분석 및 이상치 탐지 모듈

이 모듈은 데이터셋의 수치형 변수들의 분포 특성을 분석하고 이상치를 탐지합니다.
주요 기능:
- 분포 형태 분석 (정규성, 왜도, 첨도)
- 통계적 정규성 검정 (Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov)
- 이상치 탐지 (IQR, Z-score, Modified Z-score)
- 분포 요약 통계 및 시각화 권고
- 데이터 변환 제안
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest, skew, kurtosis
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
        def comprehensive_json_serializer(obj):
            """포괄적인 JSON 직렬화 함수"""
            if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Index):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            elif hasattr(obj, '__dict__') and not isinstance(obj, type):
                return str(obj)
            else:
                return str(obj)

        try:
            print(json.dumps(results, ensure_ascii=False, indent=2, default=comprehensive_json_serializer))
        except Exception as e:
            # 최후의 수단: 모든 것을 문자열로 변환
            error_output = {
                "success": False,
                "error": f"JSON 직렬화 실패: {str(e)}",
                "error_type": "SerializationError"
            }
            print(json.dumps(error_output, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

def analyze_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    수치형 변수들의 포괄적인 분포 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임

    Returns:
    --------
    Dict[str, Any]
        분포 분석 결과
        - distribution_statistics: 각 변수별 분포 통계량
        - normality_tests: 정규성 검정 결과
        - outlier_analysis: 이상치 탐지 결과
        - distribution_insights: 분포 해석 및 권고사항
    """
    try:
        # 숫자형 컬럼만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {
                'success': False,
                'error': '숫자형 컬럼이 없습니다',
                'available_columns': df.columns.tolist()
            }
        
        result = {
            'success': True,
            'analyzed_columns': numeric_df.columns.tolist(),
            'normality_tests': test_normality(numeric_df),
            'distribution_shapes': analyze_distribution_shape(numeric_df),
            'outlier_detection': detect_outliers(numeric_df),
            'histogram_info': calculate_histogram_info(numeric_df)
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'분포 분석 실패: {str(e)}',
            'error_type': 'DistributionAnalysisError'
        }

def test_normality(numeric_df):
    """
    정규성 검정 수행
    
    Args:
        numeric_df (pd.DataFrame): 숫자형 데이터프레임
        
    Returns:
        dict: 정규성 검정 결과
    """
    results = {}
    
    for col in numeric_df.columns:
        # 결측치 제거
        clean_data = numeric_df[col].dropna()
        
        if len(clean_data) < 3:
            results[col] = {
                'error': '데이터가 부족합니다 (최소 3개 필요)',
                'sample_size': len(clean_data)
            }
            continue
            
        col_results = {
            'sample_size': len(clean_data),
            'tests': {}
        }
        
        # 1. Shapiro-Wilk 검정 (표본 크기 5000개 제한)
        if len(clean_data) <= 5000:
            try:
                stat, p_value = shapiro(clean_data)
                col_results['tests']['shapiro_wilk'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
                }
            except Exception:
                col_results['tests']['shapiro_wilk'] = {'error': '검정 실패'}
        
        # 2. D'Agostino-Pearson 검정 (최소 20개 필요)
        if len(clean_data) >= 20:
            try:
                stat, p_value = normaltest(clean_data)
                col_results['tests']['dagostino_pearson'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
                }
            except Exception:
                col_results['tests']['dagostino_pearson'] = {'error': '검정 실패'}
        
        # 3. Kolmogorov-Smirnov 검정 (대표본용)
        if len(clean_data) >= 50:
            try:
                # 표준정규분포와 비교
                normalized_data = (clean_data - clean_data.mean()) / clean_data.std()
                stat, p_value = kstest(normalized_data, 'norm')
                col_results['tests']['kolmogorov_smirnov'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
                }
            except Exception:
                col_results['tests']['kolmogorov_smirnov'] = {'error': '검정 실패'}
        
        # 4. 전체 결론
        normal_tests = [test['is_normal'] for test in col_results['tests'].values() 
                       if 'is_normal' in test]
        
        if normal_tests:
            normal_count = sum(normal_tests)
            total_tests = len(normal_tests)
            
            col_results['overall_conclusion'] = {
                'is_normal': normal_count >= total_tests / 2,
                'confidence': 'High' if normal_count == total_tests or normal_count == 0 else 'Medium',
                'test_agreement': f'{normal_count}/{total_tests} tests indicate normality'
            }
        else:
            col_results['overall_conclusion'] = {
                'is_normal': None,
                'confidence': 'Unknown',
                'test_agreement': 'No valid tests performed'
            }
        
        results[col] = col_results
    
    return results

def analyze_distribution_shape(numeric_df):
    """
    분포 형태 분석 (왜도, 첨도, 모양 특성)
    
    Args:
        numeric_df (pd.DataFrame): 숫자형 데이터프레임
        
    Returns:
        dict: 분포 형태 분석 결과
    """
    results = {}
    
    for col in numeric_df.columns:
        clean_data = numeric_df[col].dropna()
        
        if len(clean_data) < 3:
            results[col] = {
                'error': '데이터가 부족합니다',
                'sample_size': len(clean_data)
            }
            continue
        
        try:
            # 왜도와 첨도 계산
            skew_val = skew(clean_data)
            kurt_val = kurtosis(clean_data)  # scipy는 excess kurtosis (정규분포=0)
            
            # 왜도 해석
            if abs(skew_val) < 0.5:
                skew_interpretation = "대칭적 (거의 정규분포)"
            elif 0.5 <= skew_val < 1.0:
                skew_interpretation = "약간 우편향 (오른쪽 꼬리가 길음)"
            elif skew_val >= 1.0:
                skew_interpretation = "강하게 우편향 (심한 오른쪽 꼬리)"
            elif -1.0 < skew_val <= -0.5:
                skew_interpretation = "약간 좌편향 (왼쪽 꼬리가 길음)"
            else:
                skew_interpretation = "강하게 좌편향 (심한 왼쪽 꼬리)"
            
            # 첨도 해석 (scipy kurtosis: 정규분포 = 0)
            if kurt_val > 1:
                kurt_interpretation = f"뾰족한 분포 (정규분포보다 {kurt_val:.1f} 더 뾰족)"
            elif kurt_val < -1:
                kurt_interpretation = f"평평한 분포 (정규분포보다 {abs(kurt_val):.1f} 더 평평)"
            else:
                kurt_interpretation = "정규분포와 비슷한 첨도"
            
            # 분포 모양 분류
            if abs(skew_val) < 0.5 and abs(kurt_val) < 1:
                shape_category = "Normal-like"
            elif skew_val > 1:
                shape_category = "Right-skewed"
            elif skew_val < -1:
                shape_category = "Left-skewed"
            elif kurt_val > 2:
                shape_category = "Heavy-tailed"
            elif kurt_val < -1:
                shape_category = "Light-tailed"
            else:
                shape_category = "Moderately non-normal"
            
            # 분포의 범위와 집중도
            data_range = float(clean_data.max() - clean_data.min())
            cv = float(clean_data.std() / clean_data.mean()) if clean_data.mean() != 0 else None
            
            results[col] = {
                'sample_size': len(clean_data),
                'skewness': {
                    'value': float(skew_val),
                    'interpretation': skew_interpretation
                },
                'kurtosis': {
                    'value': float(kurt_val),
                    'interpretation': kurt_interpretation
                },
                'shape_category': shape_category,
                'spread_measures': {
                    'range': data_range,
                    'coefficient_of_variation': cv
                },
                'percentiles': {
                    'p1': float(clean_data.quantile(0.01)),
                    'p5': float(clean_data.quantile(0.05)),
                    'p95': float(clean_data.quantile(0.95)),
                    'p99': float(clean_data.quantile(0.99))
                }
            }
            
        except Exception as e:
            results[col] = {
                'error': f'분포 형태 분석 실패: {str(e)}',
                'sample_size': len(clean_data)
            }
    
    return results

def detect_outliers(numeric_df):
    """
    다양한 방법으로 이상치 탐지
    
    Args:
        numeric_df (pd.DataFrame): 숫자형 데이터프레임
        
    Returns:
        dict: 이상치 탐지 결과
    """
    results = {}
    
    for col in numeric_df.columns:
        clean_data = numeric_df[col].dropna()
        
        if len(clean_data) < 4:
            results[col] = {
                'error': '이상치 탐지를 위한 데이터가 부족합니다',
                'sample_size': len(clean_data)
            }
            continue
        
        try:
            col_results = {
                'sample_size': len(clean_data),
                'methods': {}
            }
            
            # 1. IQR 방법
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # IQR이 0이 아닌 경우만
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = (clean_data < lower_bound) | (clean_data > upper_bound)
                
                col_results['methods']['iqr'] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(iqr_outliers.sum()),
                    'outlier_percentage': float(iqr_outliers.sum() / len(clean_data) * 100),
                    'outlier_indices': clean_data[iqr_outliers].index.tolist()
                }
            
            # 2. Z-score 방법
            if clean_data.std() > 0:  # 표준편차가 0이 아닌 경우만
                z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
                zscore_outliers = z_scores > 3
                
                col_results['methods']['zscore'] = {
                    'threshold': 3.0,
                    'outlier_count': int(zscore_outliers.sum()),
                    'outlier_percentage': float(zscore_outliers.sum() / len(clean_data) * 100),
                    'outlier_indices': clean_data[zscore_outliers].index.tolist(),
                    'max_zscore': float(z_scores.max())
                }
            
            # 3. Modified Z-score 방법 (중앙값 기반)
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            
            if mad > 0:  # MAD가 0이 아닌 경우만
                modified_z_scores = 0.6745 * (clean_data - median) / mad
                modified_zscore_outliers = np.abs(modified_z_scores) > 3.5
                
                col_results['methods']['modified_zscore'] = {
                    'threshold': 3.5,
                    'outlier_count': int(modified_zscore_outliers.sum()),
                    'outlier_percentage': float(modified_zscore_outliers.sum() / len(clean_data) * 100),
                    'outlier_indices': clean_data[modified_zscore_outliers].index.tolist()
                }
            
            # 4. 종합 결론 (각 방법의 합의)
            outlier_methods = [method for method in col_results['methods'].values() 
                             if 'outlier_count' in method]
            
            if outlier_methods:
                total_outliers = sum(method['outlier_count'] for method in outlier_methods)
                avg_outlier_percentage = np.mean([method['outlier_percentage'] for method in outlier_methods])
                
                # 모든 방법에서 탐지된 공통 이상치 찾기
                if len(outlier_methods) >= 2:
                    common_outliers = set(outlier_methods[0]['outlier_indices'])
                    for method in outlier_methods[1:]:
                        common_outliers &= set(method['outlier_indices'])
                    
                    col_results['consensus'] = {
                        'common_outliers': list(common_outliers),
                        'common_outlier_count': len(common_outliers),
                        'average_outlier_percentage': float(avg_outlier_percentage),
                        'severity': 'High' if avg_outlier_percentage > 10 else 'Medium' if avg_outlier_percentage > 5 else 'Low'
                    }
            
            results[col] = col_results
            
        except Exception as e:
            results[col] = {
                'error': f'이상치 탐지 실패: {str(e)}',
                'sample_size': len(clean_data)
            }
    
    return results

def calculate_histogram_info(numeric_df):
    """
    히스토그램 생성을 위한 정보 계산
    
    Args:
        numeric_df (pd.DataFrame): 숫자형 데이터프레임
        
    Returns:
        dict: 히스토그램 정보
    """
    results = {}
    
    for col in numeric_df.columns:
        clean_data = numeric_df[col].dropna()
        
        if len(clean_data) < 2:
            results[col] = {
                'error': '히스토그램 생성을 위한 데이터가 부족합니다',
                'sample_size': len(clean_data)
            }
            continue
        
        try:
            # 최적 bin 수 계산 (여러 방법)
            n = len(clean_data)
            
            # Sturges' rule
            bins_sturges = int(np.log2(n) + 1)
            
            # Square-root choice
            bins_sqrt = int(np.sqrt(n))
            
            # Freedman-Diaconis rule
            if n > 1:
                q75, q25 = np.percentile(clean_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    h = 2 * iqr / (n ** (1/3))  # bin width
                    bins_fd = int((clean_data.max() - clean_data.min()) / h) if h > 0 else bins_sturges
                else:
                    bins_fd = bins_sturges
            else:
                bins_fd = bins_sturges
            
            # 최종 bin 수 선택 (중간값)
            bin_options = [bins_sturges, bins_sqrt, bins_fd]
            optimal_bins = int(np.median([b for b in bin_options if 5 <= b <= 50]))  # 5-50 범위로 제한
            
            # 히스토그램 계산
            hist, bin_edges = np.histogram(clean_data, bins=optimal_bins)
            
            # bin 중심점 계산
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            results[col] = {
                'sample_size': len(clean_data),
                'optimal_bins': optimal_bins,
                'bin_calculation_methods': {
                    'sturges': bins_sturges,
                    'sqrt': bins_sqrt,
                    'freedman_diaconis': bins_fd
                },
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'bin_centers': bin_centers.tolist(),
                    'bin_width': float(bin_edges[1] - bin_edges[0])
                },
                'data_range': {
                    'min': float(clean_data.min()),
                    'max': float(clean_data.max()),
                    'range': float(clean_data.max() - clean_data.min())
                }
            }
            
        except Exception as e:
            results[col] = {
                'error': f'히스토그램 정보 계산 실패: {str(e)}',
                'sample_size': len(clean_data)
            }
    
    return results

def main():
    """
    메인 실행 함수 - 분포 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 분포 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, analysis_options

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 권고사항
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

        # 분포 분석 수행
        distribution_result = analyze_distribution(df)

        if not distribution_result.get('success', True):
            error_result = {
                "success": False,
                "error": distribution_result.get('error', '분포 분석 실패'),
                "analysis_type": "distribution_analysis",
                "available_columns": distribution_result.get('available_columns', [])
            }
            output_results(error_result)
            return

        # 분석 결과 통합 - 안전한 직렬화를 위해 단계별 처리
        def safe_extract_value(obj, path, default=None):
            """안전하게 중첩 딕셔너리에서 값 추출"""
            try:
                current = obj
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return default
                return current
            except:
                return default

        analysis_results = {
            "distribution_analysis": distribution_result,
            "analysis_summary": {
                "analyzed_columns_count": len(distribution_result.get('analyzed_columns', [])),
                "normality_results": {},
                "outlier_summary": {}
            }
        }

        # 정규성 결과 안전하게 추출
        normality_tests = distribution_result.get('normality_tests', {})
        if isinstance(normality_tests, dict):
            for col, result in normality_tests.items():
                if isinstance(result, dict):
                    is_normal = safe_extract_value(result, ['overall_conclusion', 'is_normal'], None)
                    analysis_results["analysis_summary"]["normality_results"][col] = is_normal

        # 이상치 요약 안전하게 추출
        outlier_detection = distribution_result.get('outlier_detection', {})
        if isinstance(outlier_detection, dict):
            for col, result in outlier_detection.items():
                if isinstance(result, dict):
                    severity = safe_extract_value(result, ['consensus', 'severity'], 'Unknown')
                    analysis_results["analysis_summary"]["outlier_summary"][col] = severity

        # 요약 생성
        analyzed_count = len(distribution_result.get('analyzed_columns', []))
        normal_cols = sum(1 for result in distribution_result.get('normality_tests', {}).values()
                         if isinstance(result, dict) and
                         result.get('overall_conclusion', {}).get('is_normal', False))
        summary = f"분포 분석 완료 - {analyzed_count}개 수치형 변수 분석, {normal_cols}개 정규분포 변수 확인"

        # 표준화된 결과 생성 - 단순화하여 순환 참조 방지
        final_result = {
            "analysis_type": "distribution_analysis",
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": data_info,
            "summary": summary,
            "distribution_analysis": distribution_result,
            "analysis_summary": analysis_results["analysis_summary"]
        }

        # 결과 출력
        output_results(final_result)

    except Exception as e:
        import traceback
        error_result = {
            "success": False,
            "error": str(e),
            "error_details": traceback.format_exc(),
            "analysis_type": "distribution_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()