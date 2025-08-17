#!/usr/bin/env python3
"""
파일명: example_script.py
목적: 이 스크립트의 목적을 간단히 설명
JavaScript에서 호출되어 특정 작업을 수행하고 결과를 JSON으로 반환합니다.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

def main_function(data_or_path, **kwargs):
    """
    메인 처리 함수
    
    Args:
        data_or_path: 데이터 또는 파일 경로
        **kwargs: 추가 옵션들
        
    Returns:
        dict: 처리 결과 및 메타데이터
    """
    try:
        # 1. 입력 데이터 처리
        if isinstance(data_or_path, str):
            # 파일 경로인 경우
            if not Path(data_or_path).exists():
                return {
                    'success': False,
                    'error': f'파일을 찾을 수 없습니다: {data_or_path}',
                    'error_type': 'FileNotFound'
                }
            
            # 파일 로드 (확장자에 따라 다르게 처리)
            file_ext = Path(data_or_path).suffix.lower()
            if file_ext == '.csv':
                df = pd.read_csv(data_or_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(data_or_path)
            elif file_ext == '.json':
                df = pd.read_json(data_or_path)
            else:
                return {
                    'success': False,
                    'error': f'지원하지 않는 파일 형식: {file_ext}',
                    'error_type': 'UnsupportedFormat'
                }
        else:
            # DataFrame이 직접 전달된 경우
            df = data_or_path
        
        # 2. 데이터 검증
        if df.empty:
            return {
                'success': False,
                'error': '빈 데이터셋입니다',
                'error_type': 'EmptyData'
            }
        
        # 3. 실제 처리 로직 수행
        result = perform_analysis(df, **kwargs)
        
        # 4. 결과 반환
        return {
            'success': True,
            'data_shape': df.shape,
            'processing_time': result.get('processing_time', 0),
            'result': result,
            'metadata': {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'처리 중 오류 발생: {str(e)}',
            'error_type': 'ProcessingError'
        }

def perform_analysis(df, **kwargs):
    """
    실제 분석 수행 함수
    
    Args:
        df (pd.DataFrame): 분석할 데이터
        **kwargs: 분석 옵션들
        
    Returns:
        dict: 분석 결과
    """
    import time
    start_time = time.time()
    
    try:
        # 여기에 실제 분석 로직 구현
        # 예시: 기술통계 계산
        stats = df.describe()
        
        # JSON 직렬화 가능하도록 변환
        stats_dict = {}
        for col in stats.columns:
            stats_dict[col] = {
                stat: float(value) if not pd.isna(value) else None
                for stat, value in stats[col].items()
            }
        
        processing_time = time.time() - start_time
        
        return {
            'analysis_type': 'descriptive_statistics',
            'results': stats_dict,
            'processing_time': round(processing_time, 3),
            'summary': f'{df.shape[0]}행 {df.shape[1]}열 데이터 분석 완료'
        }
        
    except Exception as e:
        raise Exception(f'분석 실패: {str(e)}')

def helper_function_1(data, param):
    """
    보조 함수 1
    
    Args:
        data: 입력 데이터
        param: 매개변수
        
    Returns:
        처리된 결과
    """
    # 보조 기능 구현
    pass

def helper_function_2(data, options={}):
    """
    보조 함수 2
    
    Args:
        data: 입력 데이터
        options: 옵션 딕셔너리
        
    Returns:
        처리된 결과
    """
    # 보조 기능 구현
    pass

def validate_input(data, required_columns=None):
    """
    입력 데이터 검증 함수
    
    Args:
        data: 검증할 데이터
        required_columns: 필수 컬럼 리스트
        
    Returns:
        bool: 검증 결과
        
    Raises:
        ValueError: 검증 실패 시
    """
    if data is None or data.empty:
        raise ValueError("데이터가 비어있습니다")
    
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    return True

def export_results(results, output_path=None, format='json'):
    """
    결과 내보내기 함수
    
    Args:
        results: 내보낼 결과 데이터
        output_path: 출력 파일 경로
        format: 출력 형식 ('json', 'csv', 'excel')
        
    Returns:
        bool: 내보내기 성공 여부
    """
    try:
        if output_path is None:
            # 기본 파일명 생성
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results_{timestamp}.{format}"
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif format == 'csv' and 'result' in results:
            # DataFrame으로 변환 가능한 경우
            if isinstance(results['result'], dict):
                df = pd.DataFrame(results['result'])
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return True
        
    except Exception as e:
        print(f"결과 내보내기 실패: {e}")
        return False

def main():
    """
    메인 함수 - 명령행에서 호출될 때 실행
    
    사용법:
        python script.py <input_file> [options_json]
        python script.py data.csv '{"param1": "value1", "param2": 123}'
    """
    try:
        # 1. 명령행 인수 확인
        if len(sys.argv) < 2:
            result = {
                'success': False,
                'error': '사용법: python script.py <input_file> [options]',
                'error_type': 'InvalidArguments',
                'usage': {
                    'description': '이 스크립트의 사용법',
                    'examples': [
                        'python script.py data.csv',
                        'python script.py data.csv \'{"option1": "value1"}\'',
                        'python script.py data.xlsx \'{"sheet_name": "Sheet1"}\''
                    ]
                }
            }
        else:
            # 2. 입력 파일 경로
            input_file = sys.argv[1]
            
            # 3. 추가 옵션 파싱 (JSON 형식)
            options = {}
            if len(sys.argv) > 2:
                try:
                    options = json.loads(sys.argv[2])
                except json.JSONDecodeError as e:
                    result = {
                        'success': False,
                        'error': f'옵션 JSON 파싱 실패: {str(e)}',
                        'error_type': 'InvalidJSON'
                    }
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return
            
            # 4. 메인 함수 실행
            result = main_function(input_file, **options)
            
            # 5. 결과 내보내기 옵션 처리
            if result['success'] and options.get('save_results', False):
                output_path = options.get('output_path')
                output_format = options.get('output_format', 'json')
                export_results(result, output_path, output_format)
        
        # 6. 결과를 JSON으로 출력 (JavaScript에서 파싱)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except KeyboardInterrupt:
        result = {
            'success': False,
            'error': '사용자에 의해 중단됨',
            'error_type': 'UserInterrupt'
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        result = {
            'success': False,
            'error': f'예상치 못한 오류: {str(e)}',
            'error_type': 'UnexpectedError'
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()