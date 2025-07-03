// tools/ml/data-loader.js에서 미완성 부분들을 완성

  async loadExcel(filePath, options = {}) {
    const {
      sheetName = null,
      header = true,
      skipRows = 0,
      maxRows = null,
      columns = null,
      engine = 'openpyxl'
    } = options;

    try {
      // Python을 통한 Excel 파일 로드
      const pythonCode = `
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Excel 파일 로드
try:
    # 시트 이름 확인
    excel_file = pd.ExcelFile('${filePath}')
    available_sheets = excel_file.sheet_names
    
    # 로드할 시트 결정
    sheet_to_load = ${sheetName ? `'${sheetName}'` : 'available_sheets[0]'}
    
    # 옵션 설정
    read_options = {
        'sheet_name': sheet_to_load,
        'header': ${header ? '0' : 'None'},
        'skiprows': ${skipRows},
        'engine': '${engine}'
    }
    
    # 최대 행 수 제한
    if ${maxRows}:
        read_options['nrows'] = ${maxRows}
    
    # 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['usecols'] = ${columns ? JSON.stringify(columns) : 'None'}
    
    # 데이터 로드
    df = pd.read_excel('${filePath}', **read_options)
    
    # 헤더가 없는 경우 컬럼명 생성
    if not ${header}:
        df.columns = [f'column_{i}' for i in range(len(df.columns))]
    
    # 데이터 타입 정보
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # 기본 통계
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    basic_stats = {}
    for col in numeric_columns:
        basic_stats[col] = {
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'count': int(df[col].count())
        }
    
    # 누락값 정보
    missing_info = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # 결과 구성
    result = {
        'data': df.replace({np.nan: None}).to_dict('records'),
        'headers': df.columns.tolist(),
        'rowCount': len(df),
        'columnCount': len(df.columns),
        'filePath': '${filePath}',
        'fileType': 'excel',
        'sheetName': sheet_to_load,
        'availableSheets': available_sheets,
        'dtypes': dtypes,
        'statistics': {
            'basic_stats': basic_stats,
            'missing_values': missing_info,
            'total_missing': sum(missing_info.values()),
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100
        },
        'metadata': {
            'mode': '${mode}',
            'used_key': used_key,
            'file_structure': file_structure,
            'available_keys': list(file_structure.keys())
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
    raise
`;

      if (this.pythonExecutor) {
        const result = await this.pythonExecutor.execute(pythonCode);
        return JSON.parse(result.output);
      } else {
        throw new Error('HDF5 파일 로드를 위해서는 Python 실행기가 필요합니다. PythonExecutor를 초기화해주세요.');
      }
    } catch (error) {
      throw new Error(`HDF5 파일 로드 실패: ${error.message}`);
    }
  }

  // PythonExecutor 설정 메서드 추가
  setPythonExecutor(pythonExecutor) {
    this.pythonExecutor = pythonExecutor;
  }

  // 파일 형식별 지원 여부 확인
  async checkFormatSupport(filePath) {
    const extension = path.extname(filePath).toLowerCase();
    const support = {
      supported: false,
      requiresPython: false,
      missingLibraries: []
    };

    switch (extension) {
      case '.csv':
      case '.json':
      case '.txt':
        support.supported = true;
        break;
      
      case '.xlsx':
      case '.xls':
        support.supported = true;
        support.requiresPython = true;
        if (!this.pythonExecutor) {
          support.missingLibraries.push('PythonExecutor');
        }
        break;
      
      case '.parquet':
        support.supported = true;
        support.requiresPython = true;
        if (!this.pythonExecutor) {
          support.missingLibraries.push('PythonExecutor');
        }
        break;
      
      case '.h5':
        support.supported = true;
        support.requiresPython = true;
        if (!this.pythonExecutor) {
          support.missingLibraries.push('PythonExecutor');
        }
        break;
      
      default:
        support.supported = false;
    }

    return support;
  }

  // 파일 형식별 로드 옵션 가져오기
  getFormatOptions(extension) {
    const options = {
      '.csv': {
        encoding: 'utf8',
        separator: ',',
        header: true,
        skipRows: 0,
        maxRows: null,
        columns: null,
        dtypes: null
      },
      '.xlsx': {
        sheetName: null,
        header: true,
        skipRows: 0,
        maxRows: null,
        columns: null,
        engine: 'openpyxl'
      },
      '.json': {
        encoding: 'utf8',
        flatten: false,
        arrayPath: null,
        maxDepth: 10
      },
      '.txt': {
        encoding: 'utf8',
        delimiter: '\n',
        skipEmpty: true,
        maxLines: null
      },
      '.parquet': {
        columns: null,
        engine: 'pyarrow',
        maxRows: null,
        filters: null
      },
      '.h5': {
        key: null,
        columns: null,
        start: null,
        stop: null,
        where: null,
        mode: 'r'
      }
    };

    return options[extension] || {};
  }

  // 자동 형식 감지 및 로드
  async autoLoadData(filePath, options = {}) {
    const extension = path.extname(filePath).toLowerCase();
    const support = await this.checkFormatSupport(filePath);
    
    if (!support.supported) {
      throw new Error(`지원하지 않는 파일 형식: ${extension}`);
    }
    
    if (support.requiresPython && !this.pythonExecutor) {
      throw new Error(`${extension} 파일을 로드하려면 PythonExecutor가 필요합니다.`);
    }
    
    const defaultOptions = this.getFormatOptions(extension);
    const mergedOptions = { ...defaultOptions, ...options };
    
    return await this.loadData(filePath, mergedOptions);
  }stats,
            'missing_values': missing_info,
            'total_missing': sum(missing_info.values()),
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100
        },
        'metadata': {
            'engine': '${engine}',
            'hasHeader': ${header},
            'skippedRows': ${skipRows}
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
    raise
`;

      // PythonExecutor가 있다면 사용, 없다면 에러 메시지
      if (this.pythonExecutor) {
        const result = await this.pythonExecutor.execute(pythonCode);
        return JSON.parse(result.output);
      } else {
        throw new Error('Excel 파일 로드를 위해서는 Python 실행기가 필요합니다. PythonExecutor를 초기화해주세요.');
      }
    } catch (error) {
      throw new Error(`Excel 파일 로드 실패: ${error.message}`);
    }
  }

  async loadParquet(filePath, options = {}) {
    const {
      columns = null,
      engine = 'pyarrow',
      maxRows = null,
      filters = null
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import json
import numpy as np
from pathlib import Path

try:
    # Parquet 파일 로드
    read_options = {
        'engine': '${engine}'
    }
    
    # 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['columns'] = ${columns ? JSON.stringify(columns) : 'None'}
    
    # 행 수 제한
    if ${maxRows}:
        # PyArrow 엔진의 경우 다른 방식 사용
        if '${engine}' == 'pyarrow':
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile('${filePath}')
            df = parquet_file.read(columns=${columns ? JSON.stringify(columns) : 'None'}).to_pandas()
            df = df.head(${maxRows})
        else:
            df = pd.read_parquet('${filePath}', **read_options)
            df = df.head(${maxRows})
    else:
        df = pd.read_parquet('${filePath}', **read_options)
    
    # 데이터 타입 정보
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # 기본 통계
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    basic_stats = {}
    for col in numeric_columns:
        basic_stats[col] = {
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'count': int(df[col].count())
        }
    
    # 누락값 정보
    missing_info = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # 파일 메타데이터 (PyArrow 사용 시)
    file_metadata = {}
    if '${engine}' == 'pyarrow':
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile('${filePath}')
            file_metadata = {
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'created_by': parquet_file.metadata.created_by,
                'version': parquet_file.metadata.version
            }
        except:
            pass
    
    # 결과 구성
    result = {
        'data': df.replace({np.nan: None}).to_dict('records'),
        'headers': df.columns.tolist(),
        'rowCount': len(df),
        'columnCount': len(df.columns),
        'filePath': '${filePath}',
        'fileType': 'parquet',
        'dtypes': dtypes,
        'statistics': {
            'basic_stats': basic_stats,
            'missing_values': missing_info,
            'total_missing': sum(missing_info.values()),
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100
        },
        'metadata': {
            'engine': '${engine}',
            'file_metadata': file_metadata
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
    raise
`;

      if (this.pythonExecutor) {
        const result = await this.pythonExecutor.execute(pythonCode);
        return JSON.parse(result.output);
      } else {
        throw new Error('Parquet 파일 로드를 위해서는 Python 실행기가 필요합니다. PythonExecutor를 초기화해주세요.');
      }
    } catch (error) {
      throw new Error(`Parquet 파일 로드 실패: ${error.message}`);
    }
  }

  async loadHDF5(filePath, options = {}) {
    const {
      key = null,
      columns = null,
      start = null,
      stop = null,
      where = null,
      mode = 'r'
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import json
import numpy as np
import h5py
from pathlib import Path

try:
    # HDF5 파일 구조 탐색
    def explore_hdf5_structure(file_path, max_depth=3, current_depth=0):
        structure = {}
        if current_depth >= max_depth:
            return structure
        
        with h5py.File(file_path, 'r') as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {
                        'type': 'dataset',
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size': obj.size
                    }
                elif isinstance(obj, h5py.Group):
                    structure[name] = {
                        'type': 'group',
                        'keys': list(obj.keys())
                    }
            
            f.visititems(visit_func)
        
        return structure
    
    # 파일 구조 탐색
    file_structure = explore_hdf5_structure('${filePath}')
    
    # 데이터 로드
    df = None
    used_key = None
    
    if ${key ? `'${key}'` : 'None'}:
        # 특정 키로 데이터 로드
        used_key = '${key}'
        read_options = {'key': '${key}', 'mode': '${mode}'}
        
        # 추가 옵션 설정
        if ${columns ? JSON.stringify(columns) : 'None'}:
            read_options['columns'] = ${columns ? JSON.stringify(columns) : 'None'}
        if ${start !== null ? start : 'None'} is not None:
            read_options['start'] = ${start}
        if ${stop !== null ? stop : 'None'} is not None:
            read_options['stop'] = ${stop}
        if ${where ? `'${where}'` : 'None'}:
            read_options['where'] = '${where}'
        
        df = pd.read_hdf('${filePath}', **read_options)
    else:
        # 기본 키 찾기
        try:
            with pd.HDFStore('${filePath}', mode='r') as store:
                keys = store.keys()
                if keys:
                    used_key = keys[0]
                    df = store[used_key]
                else:
                    raise ValueError("HDF5 파일에서 유효한 키를 찾을 수 없습니다.")
        except Exception as e:
            # pandas HDFStore로 안되면 h5py로 시도
            with h5py.File('${filePath}', 'r') as f:
                # 첫 번째 데이터셋 찾기
                dataset_keys = [k for k, v in file_structure.items() if v.get('type') == 'dataset']
                if dataset_keys:
                    used_key = dataset_keys[0]
                    # 간단한 데이터셋이면 pandas DataFrame으로 변환
                    data = f[used_key][:]
                    if len(data.shape) == 1:
                        df = pd.DataFrame({'data': data})
                    elif len(data.shape) == 2:
                        df = pd.DataFrame(data)
                    else:
                        raise ValueError(f"지원하지 않는 데이터 형태: {data.shape}")
                else:
                    raise ValueError("HDF5 파일에서 데이터셋을 찾을 수 없습니다.")
    
    if df is None:
        raise ValueError("데이터를 로드할 수 없습니다.")
    
    # 데이터 타입 정보
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # 기본 통계
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    basic_stats = {}
    for col in numeric_columns:
        basic_stats[col] = {
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'count': int(df[col].count())
        }
    
    # 누락값 정보
    missing_info = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # 결과 구성
    result = {
        'data': df.replace({np.nan: None}).to_dict('records'),
        'headers': df.columns.tolist(),
        'rowCount': len(df),
        'columnCount': len(df.columns),
        'filePath': '${filePath}',
        'fileType': 'hdf5',
        'dtypes': dtypes,
        'statistics': {
            'basic_stats': basic_
