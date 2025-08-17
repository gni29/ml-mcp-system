import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import sys
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_descriptive_stats(df):
    """
    DataFrame에 대한 기술통계 계산
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        
    Returns:
        dict: 통계 분석 결과
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
    메인 함수 - JavaScript에서 DataFrame 데이터 받기
    """
    try:
        # 표준 입력에서 DataFrame 데이터 받기
        input_data = sys.stdin.read()
        
        # JSON을 DataFrame으로 변환
        data_dict = json.loads(input_data)
        df = pd.DataFrame(data_dict)
        
        # 통계 분석 수행
        result = calculate_descriptive_stats(df)
        
        # 결과를 JSON으로 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': 'AnalysisError'
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()