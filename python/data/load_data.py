#!/usr/bin/env python3
"""
통합 데이터 로더
다양한 형식의 데이터 파일을 자동으로 인식하고 로드합니다.
JavaScript에서 호출되어 결과를 JSON으로 반환합니다.
"""

import sys
import json
import time
from pathlib import Path
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 필수 라이브러리
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Error: pandas가 설치되지 않았습니다.", file=sys.stderr)
    sys.exit(1)

# 선택적 라이브러리들
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import xlrd
    HAS_XLRD = True
except ImportError:
    HAS_XLRD = False

try:
    import pyarrow
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import feather
    HAS_FEATHER = True
except ImportError:
    HAS_FEATHER = False

# 지원되는 파일 형식과 필요한 라이브러리
SUPPORTED_FORMATS = {
    '.csv': {
        'type': 'csv',
        'required_libs': ['pandas'],
        'available': HAS_PANDAS
    },
    '.tsv': {
        'type': 'csv',
        'required_libs': ['pandas'],
        'available': HAS_PANDAS
    },
    '.xlsx': {
        'type': 'excel',
        'required_libs': ['pandas', 'openpyxl'],
        'available': HAS_PANDAS and HAS_OPENPYXL
    },
    '.xls': {
        'type': 'excel',
        'required_libs': ['pandas', 'xlrd'],
        'available': HAS_PANDAS and HAS_XLRD
    },
    '.json': {
        'type': 'json',
        'required_libs': ['pandas'],
        'available': HAS_PANDAS
    },
    '.parquet': {
        'type': 'parquet',
        'required_libs': ['pandas', 'pyarrow'],
        'available': HAS_PANDAS and HAS_PARQUET
    },
    '.feather': {
        'type': 'feather',
        'required_libs': ['pandas', 'feather'],
        'available': HAS_PANDAS and HAS_FEATHER
    }
}


def get_file_type(file_path):
    """
    파일 확장자를 기반으로 파일 타입 결정
    
    Args:
        file_path (str): 파일 경로
        
    Returns:
        dict: 파일 타입 정보
    """
    extension = Path(file_path).suffix.lower()
    
    if extension not in SUPPORTED_FORMATS:
        return {
            'type': None,
            'supported': False,
            'error': f'지원하지 않는 파일 형식: {extension}'
        }
    
    format_info = SUPPORTED_FORMATS[extension]
    
    return {
        'type': format_info['type'],
        'extension': extension,
        'supported': True,
        'available': format_info['available'],
        'required_libs': format_info['required_libs']
    }


def load_csv_data(file_path, **kwargs):
    """CSV/TSV 파일 로드"""
    # 기본 옵션
    default_options = {
        'encoding': 'utf-8',
        'low_memory': False
    }
    
    # TSV 파일인 경우 구분자 설정
    if Path(file_path).suffix.lower() == '.tsv':
        default_options['sep'] = '\t'
    
    # 사용자 옵션과 병합
    options = {**default_options, **kwargs}
    
    # 인코딩 자동 감지
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']
    
    for encoding in encodings:
        try:
            options['encoding'] = encoding
            df = pd.read_csv(file_path, **options)
            return df, {'encoding_used': encoding}
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]:  # 마지막 인코딩에서도 실패
                raise Exception(f'CSV 파일 읽기 실패: {str(e)}')
    
    raise Exception('지원되는 인코딩으로 파일을 읽을 수 없습니다')


def load_excel_data(file_path, **kwargs):
    """Excel 파일 로드"""
    file_ext = Path(file_path).suffix.lower()
    
    # 엔진 선택
    if file_ext == '.xlsx' and HAS_OPENPYXL:
        engine = 'openpyxl'
    elif file_ext == '.xls' and HAS_XLRD:
        engine = 'xlrd'
    else:
        available_engines = []
        if HAS_OPENPYXL:
            available_engines.append('openpyxl (.xlsx)')
        if HAS_XLRD:
            available_engines.append('xlrd (.xls)')
        
        raise Exception(f'Excel 파일 읽기 엔진이 없습니다. 설치 가능한 엔진: {", ".join(available_engines)}')
    
    # 기본 옵션
    default_options = {
        'engine': engine
    }
    
    # 사용자 옵션과 병합
    options = {**default_options, **kwargs}
    
    try:
        # 먼저 Excel 파일의 시트 정보 확인
        excel_file = pd.ExcelFile(file_path, engine=engine)
        sheet_names = excel_file.sheet_names
        
        # 시트가 지정되지 않았으면 첫 번째 시트 사용
        if 'sheet_name' not in options:
            options['sheet_name'] = sheet_names[0]
        
        df = pd.read_excel(file_path, **options)
        
        return df, {
            'engine_used': engine,
            'sheet_names': sheet_names,
            'current_sheet': options['sheet_name']
        }
        
    except Exception as e:
        raise Exception(f'Excel 파일 읽기 실패: {str(e)}')


def load_json_data(file_path, **kwargs):
    """JSON 파일 로드"""
    try:
        # 먼저 일반적인 pandas.read_json 시도
        try:
            df = pd.read_json(file_path, **kwargs)
            return df, {'method': 'pd.read_json'}
        except:
            # 실패하면 수동으로 JSON 로드 후 normalize
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # 배열인 경우
                df = pd.json_normalize(data)
                method = 'pd.json_normalize (array)'
            elif isinstance(data, dict):
                # 단일 객체인 경우
                df = pd.json_normalize([data])
                method = 'pd.json_normalize (object)'
            else:
                # 단일 값인 경우
                df = pd.DataFrame({'value': [data]})
                method = 'pd.DataFrame (single value)'
            
            return df, {'method': method, 'original_type': type(data).__name__}
            
    except Exception as e:
        raise Exception(f'JSON 파일 읽기 실패: {str(e)}')


def load_parquet_data(file_path, **kwargs):
    """Parquet 파일 로드"""
    try:
        df = pd.read_parquet(file_path, **kwargs)
        return df, {'engine': 'pyarrow'}
    except Exception as e:
        raise Exception(f'Parquet 파일 읽기 실패: {str(e)}')


def load_feather_data(file_path, **kwargs):
    """Feather 파일 로드"""
    try:
        df = pd.read_feather(file_path, **kwargs)
        return df, {}
    except Exception as e:
        raise Exception(f'Feather 파일 읽기 실패: {str(e)}')


def analyze_dataframe(df):
    """DataFrame 기본 분석"""
    try:
        # 기본 정보
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'has_duplicates': bool(df.duplicated().any()),
            'duplicate_count': int(df.duplicated().sum())
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
        
        # 데이터 미리보기 (처음 5행)
        preview = df.head().to_dict('records')
        # NaN 값을 None으로 변환 (JSON 직렬화 가능)
        for row in preview:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value):
                        row[key] = None
                    else:
                        row[key] = float(value)
        info['preview'] = preview
        
        # 기본 통계 (숫자형 컬럼만)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats = df[numeric_columns].describe()
            info['basic_stats'] = {}
            for col in numeric_columns:
                info['basic_stats'][col] = {
                    'count': int(stats.loc['count', col]),
                    'mean': float(stats.loc['mean', col]) if not pd.isna(stats.loc['mean', col]) else None,
                    'std': float(stats.loc['std', col]) if not pd.isna(stats.loc['std', col]) else None,
                    'min': float(stats.loc['min', col]) if not pd.isna(stats.loc['min', col]) else None,
                    'max': float(stats.loc['max', col]) if not pd.isna(stats.loc['max', col]) else None
                }
        
        return info
        
    except Exception as e:
        return {'error': f'DataFrame 분석 실패: {str(e)}'}


def load_data(file_path, **kwargs):
    """
    메인 데이터 로딩 함수
    
    Args:
        file_path (str): 파일 경로
        **kwargs: 파일 형식별 추가 옵션
        
    Returns:
        dict: 로딩 결과
    """
    start_time = time.time()
    
    try:
        # 1. 파일 존재 확인
        if not Path(file_path).exists():
            return {
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {file_path}',
                'error_type': 'FileNotFound'
            }
        
        # 2. 파일 타입 확인
        file_info = get_file_type(file_path)
        
        if not file_info['supported']:
            return {
                'success': False,
                'error': file_info['error'],
                'error_type': 'UnsupportedFormat',
                'supported_formats': list(SUPPORTED_FORMATS.keys())
            }
        
        if not file_info['available']:
            return {
                'success': False,
                'error': f'{file_info["type"]} 파일 지원을 위한 라이브러리가 설치되지 않았습니다',
                'error_type': 'MissingDependency',
                'required_libraries': file_info['required_libs']
            }
        
        # 3. 파일 로드
        file_type = file_info['type']
        loaders = {
            'csv': load_csv_data,
            'excel': load_excel_data,
            'json': load_json_data,
            'parquet': load_parquet_data,
            'feather': load_feather_data
        }
        
        df, load_info = loaders[file_type](file_path, **kwargs)
        
        # 4. DataFrame 분석
        analysis = analyze_dataframe(df)
        
        # 5. 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 6. 결과 반환
        result = {
            'success': True,
            'file_info': {
                'path': file_path,
                'size_bytes': Path(file_path).stat().st_size,
                'type': file_type,
                'extension': file_info['extension']
            },
            'load_info': load_info,
            'data_analysis': analysis,
            'processing_time': round(processing_time, 3)
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'데이터 로딩 실패: {str(e)}',
            'error_type': 'LoadingError'
        }


def main():
    """
    메인 함수 - 명령행에서 호출될 때 실행
    
    사용법:
        python load_data.py <파일경로> [옵션]
    """
    try:
        # 명령행 인수 확인
        if len(sys.argv) < 2:
            result = {
                'success': False,
                'error': '사용법: python load_data.py <파일경로> [옵션]',
                'error_type': 'InvalidArguments',
                'usage': {
                    'description': '다양한 형식의 데이터 파일을 자동으로 로드합니다',
                    'examples': [
                        'python load_data.py data.csv',
                        'python load_data.py data.xlsx',
                        'python load_data.py data.json',
                        'python load_data.py data.csv \'{"sep": ";", "encoding": "cp949"}\'',
                        'python load_data.py data.xlsx \'{"sheet_name": "Sheet2"}\''
                    ],
                    'supported_formats': list(SUPPORTED_FORMATS.keys()),
                    'available_formats': [ext for ext, info in SUPPORTED_FORMATS.items() if info['available']]
                }
            }
        else:
            # 입력 파일 경로
            file_path = sys.argv[1]
            
            # 추가 옵션 파싱 (JSON 형식)
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
            
            # 메인 로딩 수행
            result = load_data(file_path, **options)
        
        # 결과를 JSON으로 출력 (JavaScript에서 파싱)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        
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