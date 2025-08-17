#!/usr/bin/env python3
"""
CSV 파일 로딩 스크립트
JavaScript에서 호출되어 CSV 파일을 로드하고 기본 정보를 반환합니다.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

def load_csv(file_path, **kwargs):
    """
    CSV 파일을 로드하고 기본 정보를 반환
    
    Args:
        file_path (str): CSV 파일 경로
        **kwargs: pandas.read_csv에 전달할 추가 옵션
        
    Returns:
        dict: 로딩 결과 및 데이터 정보
    """
    try:
        # 파일 존재 확인
        if not Path(file_path).exists():
            return {
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {file_path}',
                'error_type': 'FileNotFound'
            }
        
        # 기본 옵션 설정
        default_options = {
            'encoding': 'utf-8',
            'low_memory': False,
            'dtype_backend': 'numpy_nullable'
        }
        
        # 사용자 옵션과 병합
        options = {**default_options, **kwargs}
        
        # UTF-8로 먼저 시도, 실패하면 다른 인코딩 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                options['encoding'] = encoding
                df = pd.read_csv(file_path, **options)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # 마지막 인코딩에서도 실패
                    return {
                        'success': False,
                        'error': f'CSV 파일 읽기 실패: {str(e)}',
                        'error_type': 'ReadError'
                    }
        
        if df is None:
            return {
                'success': False,
                'error': '지원되는 인코딩으로 파일을 읽을 수 없습니다',
                'error_type': 'EncodingError'
            }
        
        # 데이터 정보 수집
        info = {
            'success': True,
            'file_path': file_path,
            'encoding': used_encoding,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'has_duplicates': df.duplicated().any(),
            'duplicate_count': df.duplicated().sum()
        }
        
        # 결측치 정보
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(df)) * 100, 2)
                }
        info['missing_values'] = missing_info
        
        # 각 컬럼별 기본 통계
        column_stats = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'null_count': int(df[col].isnull().sum())
            }
            
            # 숫자형 컬럼 통계
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None
                })
            
            # 문자열 컬럼 통계
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                try:
                    col_info.update({
                        'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                        'avg_length': float(df[col].astype(str).str.len().mean()) if not df[col].isnull().all() else None
                    })
                except:
                    pass
            
            # 날짜/시간 컬럼 감지 및 통계
            if df[col].dtype == 'object':
                # 날짜 형식 자동 감지 시도
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if date_series.notna().sum() > len(df) * 0.5:  # 50% 이상이 유효한 날짜
                        col_info['potential_datetime'] = True
                        col_info['date_range'] = {
                            'start': str(date_series.min()) if date_series.notna().any() else None,
                            'end': str(date_series.max()) if date_series.notna().any() else None
                        }
                except:
                    pass
            
            column_stats[col] = col_info
        
        info['column_stats'] = column_stats
        
        # 데이터 미리보기 (처음 5행)
        try:
            preview = df.head().to_dict('records')
            # NaN 값을 None으로 변환 (JSON 직렬화 가능)
            for row in preview:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row[key] = float(value) if not np.isnan(value) else None
            info['preview'] = preview
        except Exception:
            info['preview'] = []
        
        # 데이터 품질 점수 계산
        quality_score = calculate_data_quality(df)
        info['quality_score'] = quality_score
        
        # 추천 전처리 단계
        info['recommendations'] = generate_recommendations(df, column_stats, missing_info)
        
        return info
        
    except Exception as e:
        return {
            'success': False,
            'error': f'예상치 못한 오류: {str(e)}',
            'error_type': 'UnexpectedError'
        }

def calculate_data_quality(df):
    """
    데이터 품질 점수를 계산합니다 (0-100점)
    
    Args:
        df (pd.DataFrame): 데이터프레임
        
    Returns:
        dict: 품질 점수 정보
    """
    try:
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # 기본 점수 (100점에서 시작)
        score = 100
        
        # 결측치 비율에 따른 감점 (최대 30점)
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        score -= min(missing_ratio * 30, 30)
        
        # 중복 행 비율에 따른 감점 (최대 20점)
        duplicate_ratio = duplicate_rows / df.shape[0] if df.shape[0] > 0 else 0
        score -= min(duplicate_ratio * 20, 20)
        
        # 데이터 타입 일관성 확인 (최대 15점 감점)
        inconsistent_types = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # 숫자로 변환 가능한지 확인
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    # 날짜로 변환 가능한지 확인
                    try:
                        pd.to_datetime(df[col], errors='raise')
                    except:
                        inconsistent_types += 1
        
        type_consistency_ratio = inconsistent_types / len(df.columns) if len(df.columns) > 0 else 0
        score -= min(type_consistency_ratio * 15, 15)
        
        # 최소값 보장
        score = max(score, 0)
        
        return {
            'overall_score': round(score, 1),
            'missing_ratio': round(missing_ratio * 100, 2),
            'duplicate_ratio': round(duplicate_ratio * 100, 2),
            'type_consistency': round((1 - type_consistency_ratio) * 100, 2),
            'recommendations': get_quality_recommendations(score, missing_ratio, duplicate_ratio)
        }
        
    except Exception:
        return {
            'overall_score': 0,
            'missing_ratio': 0,
            'duplicate_ratio': 0,
            'type_consistency': 0,
            'recommendations': ['데이터 품질 계산 중 오류 발생']
        }

def get_quality_recommendations(score, missing_ratio, duplicate_ratio):
    """품질 점수에 따른 추천사항 생성"""
    recommendations = []
    
    if score < 60:
        recommendations.append("전반적인 데이터 품질이 낮습니다. 전처리가 필요합니다.")
    
    if missing_ratio > 0.1:
        recommendations.append("결측치가 많습니다. 결측치 처리 전략을 수립하세요.")
    
    if duplicate_ratio > 0.05:
        recommendations.append("중복 행이 많습니다. 중복 제거를 고려하세요.")
    
    if score >= 80:
        recommendations.append("데이터 품질이 양호합니다.")
    
    return recommendations

def generate_recommendations(df, column_stats, missing_info):
    """
    데이터에 대한 전처리 추천사항을 생성합니다
    
    Args:
        df (pd.DataFrame): 데이터프레임
        column_stats (dict): 컬럼 통계 정보
        missing_info (dict): 결측치 정보
        
    Returns:
        list: 추천사항 목록
    """
    recommendations = []
    
    # 결측치 처리 추천
    if missing_info:
        high_missing_cols = [col for col, info in missing_info.items() if info['percentage'] > 50]
        if high_missing_cols:
            recommendations.append(f"결측치가 50% 이상인 컬럼 제거 고려: {', '.join(high_missing_cols)}")
        
        medium_missing_cols = [col for col, info in missing_info.items() if 10 < info['percentage'] <= 50]
        if medium_missing_cols:
            recommendations.append(f"결측치 대체 방법 적용 필요: {', '.join(medium_missing_cols)}")
    
    # 중복 행 처리
    if df.duplicated().any():
        recommendations.append("중복 행이 발견되었습니다. 중복 제거를 고려하세요.")
    
    # 데이터 타입 최적화
    object_cols = [col for col, stats in column_stats.items() if stats['dtype'] == 'object']
    if object_cols:
        recommendations.append("문자열 컬럼의 데이터 타입 최적화를 고려하세요.")
    
    # 범주형 변수 감지
    categorical_candidates = [
        col for col, stats in column_stats.items() 
        if stats['unique_count'] < 20 and stats['unique_count'] < len(df) * 0.05
    ]
    if categorical_candidates:
        recommendations.append(f"범주형 변수로 변환 고려: {', '.join(categorical_candidates)}")
    
    # 이상치 감지 제안
    numeric_cols = [col for col, stats in column_stats.items() if 'mean' in stats]
    if numeric_cols:
        recommendations.append("숫자형 컬럼에 대한 이상치 분석을 수행하세요.")
    
    # 날짜/시간 컬럼 처리
    datetime_candidates = [
        col for col, stats in column_stats.items() 
        if stats.get('potential_datetime', False)
    ]
    if datetime_candidates:
        recommendations.append(f"날짜/시간 형식으로 변환 고려: {', '.join(datetime_candidates)}")
    
    return recommendations

def main():
    """
    메인 함수 - 명령행에서 호출될 때 실행
    """
    if len(sys.argv) < 2:
        result = {
            'success': False,
            'error': '사용법: python load_csv.py <file_path> [options]',
            'error_type': 'InvalidArguments'
        }
    else:
        file_path = sys.argv[1]
        
        # 추가 옵션 파싱 (JSON 형식)
        options = {}
        if len(sys.argv) > 2:
            try:
                options = json.loads(sys.argv[2])
            except json.JSONDecodeError:
                pass
        
        result = load_csv(file_path, **options)
    
    # 결과를 JSON으로 출력
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()