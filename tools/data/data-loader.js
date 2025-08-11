// tools/data/data-loader.js
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import fs from 'fs/promises';
import path from 'path';

export class DataLoader {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.supportedFormats = ['csv', 'xlsx', 'json', 'parquet', 'hdf5', 'txt', 'pickle'];
    this.loadedData = new Map();
    this.dataCache = new Map();
    this.maxCacheSize = 100 * 1024 * 1024; // 100MB
    this.currentCacheSize = 0;
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('DataLoader 초기화 완료');
    } catch (error) {
      this.logger.error('DataLoader 초기화 실패:', error);
      throw error;
    }
  }

  async loadData(filePath, options = {}) {
    try {
      this.logger.info(`데이터 로딩 시작: ${filePath}`);
      
      // 파일 존재 확인
      await this.validateFile(filePath);
      
      // 캐시 확인
      const cacheKey = this.generateCacheKey(filePath, options);
      if (this.dataCache.has(cacheKey)) {
        this.logger.debug('캐시에서 데이터 로드');
        return this.dataCache.get(cacheKey);
      }

      // 파일 형식 감지
      const fileFormat = this.detectFileFormat(filePath);
      
      // 형식별 로딩
      let result;
      switch (fileFormat) {
        case 'csv':
          result = await this.loadCSV(filePath, options);
          break;
        case 'xlsx':
          result = await this.loadExcel(filePath, options);
          break;
        case 'json':
          result = await this.loadJSON(filePath, options);
          break;
        case 'parquet':
          result = await this.loadParquet(filePath, options);
          break;
        case 'hdf5':
          result = await this.loadHDF5(filePath, options);
          break;
        case 'txt':
          result = await this.loadText(filePath, options);
          break;
        case 'pickle':
          result = await this.loadPickle(filePath, options);
          break;
        default:
          throw new Error(`지원하지 않는 파일 형식: ${fileFormat}`);
      }

      // 결과 후처리
      result = await this.postProcessData(result, filePath, options);
      
      // 캐시에 저장
      this.addToCache(cacheKey, result);
      
      this.logger.info('데이터 로딩 완료', {
        format: fileFormat,
        shape: result.data_info?.shape,
        columns: result.data_info?.columns?.length
      });

      return result;

    } catch (error) {
      this.logger.error('데이터 로딩 실패:', error);
      throw error;
    }
  }

  async validateFile(filePath) {
    try {
      await fs.access(filePath);
      const stats = await fs.stat(filePath);
      
      if (!stats.isFile()) {
        throw new Error('경로가 파일이 아닙니다.');
      }
      
      if (stats.size === 0) {
        throw new Error('파일이 비어있습니다.');
      }
      
      // 파일 크기 제한 (1GB)
      const maxFileSize = 1024 * 1024 * 1024;
      if (stats.size > maxFileSize) {
        this.logger.warn(`큰 파일 감지: ${(stats.size / 1024 / 1024).toFixed(2)}MB`);
      }
      
      return true;
    } catch (error) {
      throw new Error(`파일 접근 실패: ${error.message}`);
    }
  }

  detectFileFormat(filePath) {
    const extension = path.extname(filePath).toLowerCase();
    const formatMap = {
      '.csv': 'csv',
      '.xlsx': 'xlsx',
      '.xls': 'xlsx',
      '.json': 'json',
      '.parquet': 'parquet',
      '.h5': 'hdf5',
      '.hdf5': 'hdf5',
      '.txt': 'txt',
      '.tsv': 'csv',
      '.pkl': 'pickle',
      '.pickle': 'pickle'
    };
    
    return formatMap[extension] || 'csv'; // 기본값은 CSV
  }

  async loadCSV(filePath, options = {}) {
    const {
      delimiter = ',',
      header = true,
      skipRows = 0,
      maxRows = null,
      encoding = 'utf-8',
      columns = null,
      dtypes = null,
      naValues = ['', 'NULL', 'null', 'NaN', 'nan', 'N/A']
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import numpy as np
import json
from pathlib import Path

try:
    # CSV 로딩 옵션 설정
    read_options = {
        'filepath_or_buffer': '${filePath}',
        'sep': '${delimiter}',
        'header': ${header ? '0' : 'None'},
        'skiprows': ${skipRows},
        'encoding': '${encoding}',
        'na_values': ${JSON.stringify(naValues)}
    }
    
    # 최대 행 수 제한
    if ${maxRows}:
        read_options['nrows'] = ${maxRows}
    
    # 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['usecols'] = ${JSON.stringify(columns)}
    
    # 데이터 타입 지정
    if ${dtypes ? JSON.stringify(dtypes) : 'None'}:
        read_options['dtype'] = ${JSON.stringify(dtypes)}
    
    # CSV 로드
    df = pd.read_csv(**read_options)
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
    }
    
    # 기본 통계
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    # 범주형 변수 통계
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    # 결과 구성
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'csv',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': read_options
        }
    }
    
    # 임시 데이터 저장 (분석용)
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000 // 2분
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('CSV 파일 로드 실패:', error);
      throw new Error(`CSV 로딩 실패: ${error.message}`);
    }
  }

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
      const pythonCode = `
import pandas as pd
import json
import numpy as np
from pathlib import Path

try:
    # Excel 파일 로드
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
        read_options['usecols'] = ${JSON.stringify(columns)}
    
    # Excel 로드
    df = pd.read_excel('${filePath}', **read_options)
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else [],
        'available_sheets': available_sheets,
        'loaded_sheet': sheet_to_load
    }
    
    # 통계 계산 (CSV와 동일한 로직)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    # 범주형 변수 통계
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    # 결과 구성
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'excel',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': read_options
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('Excel 파일 로드 실패:', error);
      throw new Error(`Excel 로딩 실패: ${error.message}`);
    }
  }

  async loadJSON(filePath, options = {}) {
    const {
      orient = 'records',
      lines = false,
      encoding = 'utf-8',
      normalize = false,
      maxLevel = null
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import json
import numpy as np

try:
    # JSON 로딩 옵션
    read_options = {
        'orient': '${orient}',
        'lines': ${lines},
        'encoding': '${encoding}'
    }
    
    # JSON 로드
    if ${lines}:
        df = pd.read_json('${filePath}', lines=True, encoding='${encoding}')
    else:
        df = pd.read_json('${filePath}', orient='${orient}', encoding='${encoding}')
    
    # 정규화 (필요시)
    if ${normalize} and '${orient}' == 'records':
        try:
            # 중첩된 JSON 구조 평평하게 만들기
            df = pd.json_normalize(df.to_dict('records')${maxLevel ? `, max_level=${maxLevel}` : ''})
        except Exception as norm_error:
            # 정규화 실패 시 원본 데이터 사용
            pass
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
    }
    
    # 컬럼 타입 분류
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 통계 계산
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    # 범주형 통계
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    # 결과 구성
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'json',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': read_options
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('JSON 파일 로드 실패:', error);
      throw new Error(`JSON 로딩 실패: ${error.message}`);
    }
  }

  async loadParquet(filePath, options = {}) {
    const {
      columns = null,
      filters = null,
      engine = 'pyarrow'
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import json
import numpy as np

try:
    # Parquet 로딩 옵션
    read_options = {
        'engine': '${engine}'
    }
    
    # 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['columns'] = ${JSON.stringify(columns)}
    
    # 필터 적용 (pyarrow만 지원)
    if ${filters ? JSON.stringify(filters) : 'None'} and '${engine}' == 'pyarrow':
        read_options['filters'] = ${JSON.stringify(filters)}
    
    # Parquet 로드
    df = pd.read_parquet('${filePath}', **read_options)
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
    }
    
    # 컬럼 타입 분류 및 통계 (이전과 동일한 로직)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'parquet',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': read_options
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}',
        'suggestion': 'PythonExecutor를 초기화해주세요.'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('Parquet 파일 로드 실패:', error);
      throw new Error(`Parquet 로딩 실패: ${error.message}`);
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
            read_options['columns'] = ${JSON.stringify(columns)}
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
            # HDFStore로 실패하면 h5py로 시도
            with h5py.File('${filePath}', 'r') as f:
                # 첫 번째 데이터셋 찾기
                dataset_names = []
                def find_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        dataset_names.append(name)
                f.visititems(find_datasets)
                
                if dataset_names:
                    used_key = dataset_names[0]
                    dataset = f[used_key]
                    df = pd.DataFrame(dataset[:])
                else:
                    raise ValueError("HDF5 파일에서 데이터셋을 찾을 수 없습니다.")
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else [],
        'hdf5_structure': file_structure,
        'used_key': used_key
    }
    
    # 통계 계산
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'hdf5',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': read_options if '${key}' else {}
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('HDF5 파일 로드 실패:', error);
      throw new Error(`HDF5 로딩 실패: ${error.message}`);
    }
  }

  async loadText(filePath, options = {}) {
    const {
      delimiter = null,
      encoding = 'utf-8',
      skipLines = 0,
      maxLines = null,
      parseNumbers = true
    } = options;

    try {
      const pythonCode = `
import pandas as pd
import json
import numpy as np
from pathlib import Path
import re

try:
    # 텍스트 파일 읽기
    with open('${filePath}', 'r', encoding='${encoding}') as f:
        lines = f.readlines()
    
    # 스키핑과 라인 제한
    if ${skipLines} > 0:
        lines = lines[${skipLines}:]
    
    if ${maxLines}:
        lines = lines[:${maxLines}]
    
    # 구분자가 지정된 경우 CSV로 처리
    if ${delimiter ? `'${delimiter}'` : 'None'}:
        # 임시 파일로 저장 후 CSV로 로드
        temp_path = './temp/temp_text.csv'
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        df = pd.read_csv(temp_path, sep='${delimiter}')
    else:
        # 일반 텍스트 처리
        # 숫자 패턴 찾기
        if ${parseNumbers}:
            processed_lines = []
            for line in lines:
                # 숫자와 공백으로만 이루어진 라인 찾기
                numbers = re.findall(r'-?\\d+\\.?\\d*', line.strip())
                if numbers:
                    processed_lines.append(numbers)
                else:
                    # 텍스트 라인은 그대로 유지
                    processed_lines.append([line.strip()])
            
            # 최대 컬럼 수 찾기
            max_cols = max(len(row) for row in processed_lines) if processed_lines else 1
            
            # 모든 행을 같은 길이로 만들기
            for row in processed_lines:
                while len(row) < max_cols:
                    row.append('')
            
            # DataFrame 생성
            columns = [f'col_{i}' for i in range(max_cols)]
            df = pd.DataFrame(processed_lines, columns=columns)
            
            # 숫자 컬럼 변환 시도
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        else:
            # 텍스트 라인을 그대로 DataFrame으로
            df = pd.DataFrame({'text': [line.strip() for line in lines]})
    
    # 기본 정보 수집
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else [],
        'total_lines': len(lines),
        'processed_lines': len(df)
    }
    
    # 통계 계산
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': []
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'text',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2),
            'load_options': {
                'delimiter': ${delimiter ? `'${delimiter}'` : 'None'},
                'encoding': '${encoding}',
                'parse_numbers': ${parseNumbers}
            }
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('텍스트 파일 로드 실패:', error);
      throw new Error(`텍스트 로딩 실패: ${error.message}`);
    }
  }

  async loadPickle(filePath, options = {}) {
    try {
      const pythonCode = `
import pandas as pd
import pickle
import json
import numpy as np

try:
    # Pickle 파일 로드
    with open('${filePath}', 'rb') as f:
        data = pickle.load(f)
    
    # 데이터 타입에 따른 처리
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # 딕셔너리를 DataFrame으로 변환 시도
        try:
            df = pd.DataFrame(data)
        except:
            # 변환 실패시 딕셔너리 정보만 제공
            result = {
                'success': True,
                'data_type': 'dictionary',
                'keys': list(data.keys()),
                'data_info': {
                    'type': 'dict',
                    'keys_count': len(data.keys()),
                    'sample_keys': list(data.keys())[:10]
                },
                'file_info': {
                    'path': '${filePath}',
                    'format': 'pickle'
                }
            }
            print(json.dumps(result, ensure_ascii=False, default=str))
            exit()
    elif isinstance(data, (list, tuple)):
        # 리스트/튜플을 DataFrame으로 변환 시도
        try:
            df = pd.DataFrame(data)
        except:
            # 변환 실패시 리스트 정보만 제공
            result = {
                'success': True,
                'data_type': 'list',
                'length': len(data),
                'data_info': {
                    'type': 'list',
                    'length': len(data),
                    'sample_items': data[:5] if len(data) > 0 else []
                },
                'file_info': {
                    'path': '${filePath}',
                    'format': 'pickle'
                }
            }
            print(json.dumps(result, ensure_ascii=False, default=str))
            exit()
    else:
        # 기타 객체 타입
        result = {
            'success': True,
            'data_type': str(type(data)),
            'data_info': {
                'type': str(type(data)),
                'str_representation': str(data)[:500]
            },
            'file_info': {
                'path': '${filePath}',
                'format': 'pickle'
            }
        }
        print(json.dumps(result, ensure_ascii=False, default=str))
        exit()
    
    # DataFrame 처리
    data_info = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
    }
    
    # 통계 계산
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    statistics = {}
    if numeric_columns:
        desc_stats = df[numeric_columns].describe()
        for col in numeric_columns:
            statistics[col] = {
                'count': int(desc_stats.loc['count', col]),
                'mean': float(desc_stats.loc['mean', col]),
                'std': float(desc_stats.loc['std', col]),
                'min': float(desc_stats.loc['min', col]),
                'q25': float(desc_stats.loc['25%', col]),
                'median': float(desc_stats.loc['50%', col]),
                'q75': float(desc_stats.loc['75%', col]),
                'max': float(desc_stats.loc['max', col]),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
    
    categorical_stats = {}
    for col in categorical_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'top_values': value_counts.head(10).to_dict()
            }
    
    result = {
        'success': True,
        'data_info': data_info,
        'statistics': statistics,
        'categorical_stats': categorical_stats,
        'column_types': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'datetime': datetime_columns
        },
        'file_info': {
            'path': '${filePath}',
            'format': 'pickle',
            'size_mb': round(data_info['memory_usage'] / 1024 / 1024, 2)
        }
    }
    
    # 임시 데이터 저장
    temp_file = './temp/loaded_data.csv'
    df.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__,
        'file_path': '${filePath}'
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

      const executionResult = await this.pythonExecutor.execute(pythonCode, {
        timeout: 120000
      });

      if (executionResult.success) {
        const result = JSON.parse(executionResult.output);
        if (result.success) {
          return result;
        } else {
          throw new Error(result.error);
        }
      } else {
        throw new Error(executionResult.error);
      }

    } catch (error) {
      this.logger.error('Pickle 파일 로드 실패:', error);
      throw new Error(`Pickle 로딩 실패: ${error.message}`);
    }
  }

  // 데이터 후처리 메서드들
  async postProcessData(result, filePath, options) {
    try {
      // 메타데이터 추가
      result.metadata = {
        loadedAt: new Date().toISOString(),
        filePath: filePath,
        loadOptions: options,
        dataId: this.generateDataId(filePath)
      };

      // 데이터 품질 평가
      result.quality_assessment = await this.assessDataQuality(result);

      // 권장사항 생성
      result.recommendations = this.generateRecommendations(result);

      return result;

    } catch (error) {
      this.logger.error('데이터 후처리 실패:', error);
      return result; // 후처리 실패해도 원본 결과는 반환
    }
  }

  async assessDataQuality(result) {
    const assessment = {
      overall_score: 0,
      completeness: 0,
      consistency: 0,
      validity: 0,
      issues: [],
      strengths: []
    };

    try {
      const { data_info, statistics } = result;
      
      // 완전성 평가 (결측값 비율)
      const totalCells = data_info.shape[0] * data_info.shape[1];
      const nullCells = Object.values(data_info.null_counts || {})
        .reduce((sum, count) => sum + count, 0);
      assessment.completeness = Math.max(0, 1 - (nullCells / totalCells));

      // 일관성 평가 (데이터 타입 일관성)
      const numericColumns = result.column_types?.numeric?.length || 0;
      const totalColumns = data_info.columns.length;
      assessment.consistency = totalColumns > 0 ? 
        (numericColumns + result.column_types?.datetime?.length || 0) / totalColumns : 0;

      // 유효성 평가 (이상값 검출)
      let validityScore = 1.0;
      if (statistics) {
        for (const [column, stats] of Object.entries(statistics)) {
          if (stats.std === 0) {
            assessment.issues.push(`${column}: 모든 값이 동일합니다.`);
            validityScore -= 0.1;
          }
          if (stats.missing_percentage > 50) {
            assessment.issues.push(`${column}: 결측값이 50% 이상입니다.`);
            validityScore -= 0.2;
          }
        }
      }
      assessment.validity = Math.max(0, validityScore);

      // 전체 점수 계산
      assessment.overall_score = (
        assessment.completeness * 0.4 +
        assessment.consistency * 0.3 +
        assessment.validity * 0.3
      );

      // 강점 식별
      if (assessment.completeness > 0.9) {
        assessment.strengths.push('데이터가 거의 완전합니다.');
      }
      if (totalColumns > 10) {
        assessment.strengths.push('풍부한 특성을 가지고 있습니다.');
      }
      if (data_info.shape[0] > 1000) {
        assessment.strengths.push('충분한 양의 데이터가 있습니다.');
      }

    } catch (error) {
      this.logger.error('데이터 품질 평가 실패:', error);
    }

    return assessment;
  }

  generateRecommendations(result) {
    const recommendations = [];

    try {
      const { data_info, statistics, quality_assessment } = result;

      // 데이터 품질 기반 권장사항
      if (quality_assessment?.completeness < 0.8) {
        recommendations.push({
          type: 'data_cleaning',
          priority: 'high',
          message: '결측값 처리가 필요합니다.',
          action: 'missing_value_imputation'
        });
      }

      // 데이터 크기 기반 권장사항
      if (data_info.shape[0] < 100) {
        recommendations.push({
          type: 'sample_size',
          priority: 'medium',
          message: '데이터 샘플이 부족할 수 있습니다.',
          action: 'collect_more_data'
        });
      }

      // 특성 수 기반 권장사항
      if (data_info.shape[1] > 50) {
        recommendations.push({
          type: 'dimensionality',
          priority: 'medium',
          message: '차원 축소를 고려해보세요.',
          action: 'feature_selection_or_pca'
        });
      }

      // 데이터 타입 기반 권장사항
      const numericRatio = (result.column_types?.numeric?.length || 0) / data_info.shape[1];
      if (numericRatio < 0.3) {
        recommendations.push({
          type: 'feature_engineering',
          priority: 'medium',
          message: '범주형 변수 인코딩이 필요할 수 있습니다.',
          action: 'categorical_encoding'
        });
      }

      // 통계 기반 권장사항
      if (statistics) {
        for (const [column, stats] of Object.entries(statistics)) {
          if (Math.abs(stats.mean) > 3 * stats.std && stats.std > 0) {
            recommendations.push({
              type: 'outlier_detection',
              priority: 'medium',
              message: `${column} 컬럼에 이상값이 있을 수 있습니다.`,
              action: 'outlier_analysis'
            });
            break; // 한 번만 추가
          }
        }
      }

    } catch (error) {
      this.logger.error('권장사항 생성 실패:', error);
    }

    return recommendations;
  }

  // 캐시 관리 메서드들
  generateCacheKey(filePath, options) {
    const optionsStr = JSON.stringify(options);
    return `${filePath}_${Buffer.from(optionsStr).toString('base64')}`;
  }

  addToCache(key, data) {
    const dataSize = JSON.stringify(data).length;
    
    // 캐시 크기 확인
    if (this.currentCacheSize + dataSize > this.maxCacheSize) {
      this.clearOldCache();
    }
    
    this.dataCache.set(key, {
      data,
      timestamp: Date.now(),
      size: dataSize
    });
    
    this.currentCacheSize += dataSize;
  }

  clearOldCache() {
    const entries = Array.from(this.dataCache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    
    // 오래된 것부터 절반 삭제
    const toDelete = Math.floor(entries.length / 2);
    for (let i = 0; i < toDelete; i++) {
      const [key, value] = entries[i];
      this.currentCacheSize -= value.size;
      this.dataCache.delete(key);
    }
    
    this.logger.debug(`캐시 정리 완료: ${toDelete}개 항목 삭제`);
  }

  // 유틸리티 메서드들
  generateDataId(filePath) {
    return Buffer.from(filePath + Date.now()).toString('base64').substring(0, 16);
  }

  getSupportedFormats() {
    return [...this.supportedFormats];
  }

  getCacheInfo() {
    return {
      cacheSize: this.dataCache.size,
      currentCacheSizeMB: Math.round(this.currentCacheSize / 1024 / 1024 * 100) / 100,
      maxCacheSizeMB: this.maxCacheSize / 1024 / 1024
    };
  }

  // 데이터 변환 메서드들
  async validateData(data, validationRules = {}) {
    // 데이터 검증 로직 구현
    const validation = {
      isValid: true,
      errors: [],
      warnings: []
    };

    // 기본 검증 규칙들...
    
    return validation;
  }

  async transformData(data, transformOptions = {}) {
    // 데이터 변환 로직 구현
    try {
      const transformedData = { ...data };
      
      // 변환 작업들...
      
      return transformedData;
    } catch (error) {
      this.logger.error('데이터 변환 실패:', error);
      throw error;
    }
  }

  // 전처리 메서드들 (pipeline-manager에서 사용)
  async handleMissingValues(data, options = {}) {
    const { strategy = 'drop', fillValue = null } = options;
    
    const pythonCode = `
import pandas as pd
import numpy as np
import json

try:
    # 임시 파일에서 데이터 로드
    df = pd.read_csv('./temp/loaded_data.csv')
    
    strategy = '${strategy}'
    
    if strategy == 'drop':
        # 결측값이 있는 행 삭제
        df_cleaned = df.dropna()
    elif strategy == 'fill_mean':
        # 수치형 컬럼은 평균으로, 범주형은 최빈값으로
        df_cleaned = df.copy()
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df[col].mean(), inplace=True)
            else:
                df_cleaned[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    elif strategy == 'fill_value':
        # 지정된 값으로 채우기
        fill_val = ${fillValue ? `'${fillValue}'` : 'None'}
        df_cleaned = df.fillna(fill_val)
    else:
        df_cleaned = df.copy()
    
    # 결과 정보
    result = {
        'success': True,
        'original_shape': list(df.shape),
        'cleaned_shape': list(df_cleaned.shape),
        'rows_removed': df.shape[0] - df_cleaned.shape[0],
        'strategy_used': strategy,
        'missing_values_before': df.isnull().sum().sum(),
        'missing_values_after': df_cleaned.isnull().sum().sum()
    }
    
    # 정리된 데이터 저장
    temp_file = './temp/cleaned_data.csv'
    df_cleaned.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

    const executionResult = await this.pythonExecutor.execute(pythonCode);
    
    if (executionResult.success) {
      const result = JSON.parse(executionResult.output);
      return result;
    } else {
      throw new Error(executionResult.error);
    }
  }

  async normalizeData(data, options = {}) {
    const { method = 'standard', columns = null } = options;
    
    const pythonCode = `
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

try:
    # 데이터 로드
    df = pd.read_csv('./temp/loaded_data.csv')
    
    # 정규화할 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        cols_to_normalize = ${JSON.stringify(columns)}
    else:
        # 수치형 컬럼만 선택
        cols_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    scaler_info = {}
    
    if '${method}' == 'standard':
        scaler = StandardScaler()
    elif '${method}' == 'minmax':
        scaler = MinMaxScaler()
    elif '${method}' == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"지원하지 않는 정규화 방법: ${method}")
    
    # 정규화 적용
    if cols_to_normalize:
        df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        
        # 스케일러 정보 저장
        for i, col in enumerate(cols_to_normalize):
            if hasattr(scaler, 'mean_'):
                scaler_info[col] = {
                    'mean': float(scaler.mean_[i]),
                    'scale': float(scaler.scale_[i])
                }
            elif hasattr(scaler, 'data_min_'):
                scaler_info[col] = {
                    'min': float(scaler.data_min_[i]),
                    'scale': float(scaler.scale_[i])
                }
    
    result = {
        'success': True,
        'method': '${method}',
        'normalized_columns': cols_to_normalize,
        'scaler_info': scaler_info,
        'shape': list(df_normalized.shape)
    }
    
    # 정규화된 데이터 저장
    temp_file = './temp/normalized_data.csv'
    df_normalized.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

    const executionResult = await this.pythonExecutor.execute(pythonCode);
    
    if (executionResult.success) {
      const result = JSON.parse(executionResult.output);
      return result;
    } else {
      throw new Error(executionResult.error);
    }
  }

  async encodeCategorical(data, options = {}) {
    const { method = 'onehot', columns = null, dropFirst = true } = options;
    
    const pythonCode = `
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    # 데이터 로드
    df = pd.read_csv('./temp/loaded_data.csv')
    
    # 인코딩할 컬럼 선택
    if ${columns ? JSON.stringify(columns) : 'None'}:
        cols_to_encode = ${JSON.stringify(columns)}
    else:
        # 범주형 컬럼만 선택
        cols_to_encode = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_encoded = df.copy()
    encoding_info = {}
    
    for col in cols_to_encode:
        if col in df.columns:
            if '${method}' == 'label':
                # 라벨 인코딩
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                encoding_info[col] = {
                    'method': 'label',
                    'classes': le.classes_.tolist()
                }
            elif '${method}' == 'onehot':
                # 원핫 인코딩
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=${dropFirst})
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                encoding_info[col] = {
                    'method': 'onehot',
                    'new_columns': dummies.columns.tolist(),
                    'dropped_first': ${dropFirst}
                }
    
    result = {
        'success': True,
        'method': '${method}',
        'encoded_columns': cols_to_encode,
        'encoding_info': encoding_info,
        'original_shape': list(df.shape),
        'encoded_shape': list(df_encoded.shape)
    }
    
    # 인코딩된 데이터 저장
    temp_file = './temp/encoded_data.csv'
    df_encoded.to_csv(temp_file, index=False)
    result['temp_file'] = temp_file
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_result, ensure_ascii=False))
`;

    const executionResult = await this.pythonExecutor.execute(pythonCode);
    
    if (executionResult.success) {
      const result = JSON.parse(executionResult.output);
      return result;
    } else {
      throw new Error(executionResult.error);
    }
  }

  async scaleFeatures(data, options = {}) {
    // normalizeData와 동일한 기능이지만 다른 이름으로 호출
    return await this.normalizeData(data, options);
  }

  async comprehensivePreprocessing(data, options = {}) {
    const {
      handleMissing = true,
      missingStrategy = 'drop',
      normalizeFeatures = true,
      normalizationMethod = 'standard',
      encodeCategorical = true,
      encodingMethod = 'onehot',
      removeOutliers = false,
      outlierMethod = 'iqr'
    } = options;

    try {
      let currentData = data;
      const steps = [];

      // 1. 결측값 처리
      if (handleMissing) {
        const missingResult = await this.handleMissingValues(currentData, {
          strategy: missingStrategy
        });
        steps.push({
          step: 'missing_values',
          result: missingResult
        });
      }

      // 2. 이상치 제거 (구현 필요시)
      if (removeOutliers) {
        // 이상치 제거 로직 추가 가능
      }

      // 3. 범주형 인코딩
      if (encodeCategorical) {
        const encodingResult = await this.encodeCategorical(currentData, {
          method: encodingMethod
        });
        steps.push({
          step: 'categorical_encoding',
          result: encodingResult
        });
      }

      // 4. 특성 정규화
      if (normalizeFeatures) {
        const normalizationResult = await this.normalizeData(currentData, {
          method: normalizationMethod
        });
        steps.push({
          step: 'feature_normalization',
          result: normalizationResult
        });
      }

      return {
        success: true,
        preprocessing_steps: steps,
        summary: `총 ${steps.length}개의 전처리 단계가 완료되었습니다.`,
        final_temp_file: steps[steps.length - 1]?.result?.temp_file
      };

    } catch (error) {
      this.logger.error('종합 전처리 실패:', error);
      throw error;
    }
  }

  // 정리 작업
  async cleanup() {
    try {
      // 캐시 정리
      this.dataCache.clear();
      this.currentCacheSize = 0;
      
      // Python 실행기 정리
      if (this.pythonExecutor && typeof this.pythonExecutor.cleanup === 'function') {
        await this.pythonExecutor.cleanup();
      }
      
      this.logger.info('DataLoader 정리 완료');
    } catch (error) {
      this.logger.error('DataLoader 정리 실패:', error);
    }
  }

  // 헬스 체크
  async healthCheck() {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      cache_info: this.getCacheInfo(),
      supported_formats: this.getSupportedFormats()
    };

    try {
      // Python 실행기 상태 확인
      if (this.pythonExecutor && typeof this.pythonExecutor.healthCheck === 'function') {
        const pythonHealth = await this.pythonExecutor.healthCheck();
        health.python_executor = pythonHealth;
        
        if (pythonHealth.status !== 'healthy') {
          health.status = 'degraded';
        }
      }
    } catch (error) {
      health.status = 'error';
      health.error = error.message;
    }

    return health;
  }
}