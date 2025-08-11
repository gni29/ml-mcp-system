// tools/data/data-loader.js - 완전한 데이터 로딩 시스템
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import fs from 'fs/promises';
import path from 'path';

export class DataLoader {
  constructor() {
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.pythonExecutor = null;
    this.supportedFormats = ['csv', 'xlsx', 'xls', 'json', 'txt', 'parquet', 'h5'];
    this.maxFileSize = 500 * 1024 * 1024; // 500MB
    this.loadHistory = [];
  }

  async initialize() {
    try {
      await this.configLoader.initialize();
      this.logger.info('DataLoader 초기화 완료');
    } catch (error) {
      this.logger.error('DataLoader 초기화 실패:', error);
      throw error;
    }
  }

  setPythonExecutor(pythonExecutor) {
    this.pythonExecutor = pythonExecutor;
    this.logger.info('PythonExecutor 설정 완료');
  }

  async loadData(filePath, options = {}) {
    const startTime = Date.now();
    
    try {
      this.logger.info(`데이터 로딩 시작: ${filePath}`);

      // 파일 존재 및 크기 확인
      await this.validateFile(filePath);

      // 파일 형식 감지
      const extension = path.extname(filePath).toLowerCase();
      const format = this.detectFileFormat(extension);

      // 형식별 로딩
      let result;
      switch (format) {
        case 'csv':
          result = await this.loadCSV(filePath, options);
          break;
        case 'json':
          result = await this.loadJSON(filePath, options);
          break;
        case 'excel':
          result = await this.loadExcel(filePath, options);
          break;
        case 'text':
          result = await this.loadText(filePath, options);
          break;
        case 'parquet':
          result = await this.loadParquet(filePath, options);
          break;
        case 'hdf5':
          result = await this.loadHDF5(filePath, options);
          break;
        default:
          throw new Error(`지원하지 않는 파일 형식: ${extension}`);
      }

      // 로딩 결과 후처리
      const finalResult = await this.postProcessResult(result, options);
      
      // 로딩 히스토리 기록
      this.recordLoadHistory(filePath, format, Date.now() - startTime, true);
      
      this.logger.info(`데이터 로딩 완료: ${filePath} (${Date.now() - startTime}ms)`);
      return finalResult;

    } catch (error) {
      this.recordLoadHistory(filePath, 'unknown', Date.now() - startTime, false, error);
      this.logger.error(`데이터 로딩 실패: ${filePath}`, error);
      throw error;
    }
  }

  async validateFile(filePath) {
    try {
      const stats = await fs.stat(filePath);
      
      if (!stats.isFile()) {
        throw new Error('지정된 경로가 파일이 아닙니다.');
      }

      if (stats.size > this.maxFileSize) {
        throw new Error(`파일이 너무 큽니다: ${this.formatFileSize(stats.size)} (최대: ${this.formatFileSize(this.maxFileSize)})`);
      }

      if (stats.size === 0) {
        throw new Error('파일이 비어있습니다.');
      }

    } catch (error) {
      if (error.code === 'ENOENT') {
        throw new Error(`파일을 찾을 수 없습니다: ${filePath}`);
      } else if (error.code === 'EACCES') {
        throw new Error(`파일에 접근할 수 없습니다: ${filePath}`);
      } else {
        throw error;
      }
    }
  }

  detectFileFormat(extension) {
    const formatMap = {
      '.csv': 'csv',
      '.tsv': 'csv',
      '.xlsx': 'excel',
      '.xls': 'excel',
      '.json': 'json',
      '.txt': 'text',
      '.parquet': 'parquet',
      '.h5': 'hdf5',
      '.hdf5': 'hdf5'
    };
    
    return formatMap[extension] || 'unknown';
  }

  async loadCSV(filePath, options = {}) {
    const {
      encoding = 'utf8',
      separator = 'auto',
      header = true,
      skipRows = 0,
      maxRows = null,
      columns = null,
      dtypes = null
    } = options;

    try {
      // 파일 내용 읽기
      const content = await fs.readFile(filePath, encoding);
      const lines = content.split('\n').filter(line => line.trim());
      
      if (lines.length === 0) {
        throw new Error('CSV 파일이 비어있습니다.');
      }

      // 구분자 자동 감지
      const delimiter = separator === 'auto' ? this.detectCSVDelimiter(lines[0]) : separator;
      
      // 헤더 처리
      let headerRow = header ? lines[skipRows] : null;
      let dataStartIndex = header ? skipRows + 1 : skipRows;
      
      const headers = headerRow ? 
        headerRow.split(delimiter).map(h => h.trim().replace(/"/g, '')) :
        null;

      // 데이터 행 처리
      const dataLines = lines.slice(dataStartIndex);
      const maxDataRows = maxRows ? Math.min(maxRows, dataLines.length) : dataLines.length;
      
      const data = [];
      const parseErrors = [];

      for (let i = 0; i < maxDataRows; i++) {
        try {
          const values = this.parseCSVRow(dataLines[i], delimiter);
          
          // 컬럼 선택
          if (columns && headers) {
            const selectedValues = {};
            columns.forEach(col => {
              const colIndex = headers.indexOf(col);
              if (colIndex !== -1) {
                selectedValues[col] = values[colIndex];
              }
            });
            data.push(selectedValues);
          } else if (headers) {
            const rowData = {};
            headers.forEach((header, index) => {
              rowData[header] = values[index] || null;
            });
            data.push(rowData);
          } else {
            data.push(values);
          }
        } catch (error) {
          parseErrors.push({ row: i + dataStartIndex + 1, error: error.message });
        }
      }

      // 데이터 타입 추론 및 변환
      let processedData = data;
      if (dtypes || options.inferTypes !== false) {
        processedData = this.inferAndConvertTypes(data, dtypes);
      }

      // 기본 통계 계산
      const statistics = this.calculateBasicStatistics(processedData, headers);

      return {
        data: processedData,
        headers: headers || (data.length > 0 ? Object.keys(data[0]) : []),
        rowCount: data.length,
        columnCount: headers ? headers.length : (data.length > 0 ? data[0].length : 0),
        filePath,
        fileType: 'csv',
        delimiter,
        encoding,
        statistics,
        parseErrors,
        metadata: {
          hasHeader: header,
          skippedRows: skipRows,
          totalLines: lines.length,
          originalSize: content.length
        }
      };

    } catch (error) {
      throw new Error(`CSV 파일 로드 실패: ${error.message}`);
    }
  }

  async loadJSON(filePath, options = {}) {
    const {
      encoding = 'utf8',
      flatten = false,
      arrayPath = null,
      maxDepth = 10
    } = options;

    try {
      const content = await fs.readFile(filePath, encoding);
      let data = JSON.parse(content);

      // 배열 경로 처리
      if (arrayPath) {
        const pathParts = arrayPath.split('.');
        for (const part of pathParts) {
          if (data && typeof data === 'object') {
            data = data[part];
          } else {
            throw new Error(`배열 경로를 찾을 수 없습니다: ${arrayPath}`);
          }
        }
      }

      // 데이터 구조 분석
      const isArray = Array.isArray(data);
      let processedData = data;
      let headers = [];

      if (isArray && data.length > 0) {
        // 배열의 첫 번째 객체에서 헤더 추출
        if (typeof data[0] === 'object' && data[0] !== null) {
          headers = Object.keys(data[0]);
          
          // 플래튼 처리
          if (flatten) {
            processedData = data.map(item => this.flattenObject(item, maxDepth));
            headers = processedData.length > 0 ? Object.keys(processedData[0]) : [];
          }
        }
      } else if (typeof data === 'object' && data !== null) {
        // 객체인 경우 배열로 변환
        if (flatten) {
          processedData = [this.flattenObject(data, maxDepth)];
          headers = Object.keys(processedData[0]);
        } else {
          processedData = [data];
          headers = Object.keys(data);
        }
      }

      // 통계 계산
      const statistics = this.calculateBasicStatistics(processedData, headers);

      return {
        data: processedData,
        headers,
        rowCount: Array.isArray(processedData) ? processedData.length : 1,
        columnCount: headers.length,
        filePath,
        fileType: 'json',
        encoding,
        statistics,
        metadata: {
          originalType: isArray ? 'array' : 'object',
          flattened: flatten,
          arrayPath,
          maxDepth: flatten ? maxDepth : null,
          originalSize: content.length
        }
      };

    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(`JSON 파일 형식이 유효하지 않습니다: ${error.message}`);
      }
      throw new Error(`JSON 파일 로드 실패: ${error.message}`);
    }
  }

  async loadExcel(filePath, options = {}) {
    if (!this.pythonExecutor) {
      throw new Error('Excel 파일 로드를 위해서는 PythonExecutor가 필요합니다.');
    }

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

try:
    # Excel 파일 및 시트 정보 확인
    excel_file = pd.ExcelFile('${filePath}')
    available_sheets = excel_file.sheet_names
    
    # 로드할 시트 결정
    sheet_to_load = ${sheetName ? `'${sheetName}'` : 'available_sheets[0]'}
    
    if sheet_to_load not in available_sheets:
        raise ValueError(f"시트 '{sheet_to_load}'를 찾을 수 없습니다. 사용 가능한 시트: {available_sheets}")
    
    # 읽기 옵션 설정
    read_options = {
        'sheet_name': sheet_to_load,
        'header': ${header ? '0' : 'None'},
        'skiprows': ${skipRows},
        'engine': '${engine}'
    }
    
    if ${maxRows}:
        read_options['nrows'] = ${maxRows}
    
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['usecols'] = ${JSON.stringify(columns)}
    
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
    
    # 결과 반환
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
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
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

      const result = await this.pythonExecutor.execute(pythonCode);
      const parsedResult = JSON.parse(result.output);
      
      if (parsedResult.error) {
        throw new Error(parsedResult.error);
      }
      
      return parsedResult;

    } catch (error) {
      throw new Error(`Excel 파일 로드 실패: ${error.message}`);
    }
  }

  async loadText(filePath, options = {}) {
    const {
      encoding = 'utf8',
      delimiter = '\n',
      skipEmpty = true,
      maxLines = null
    } = options;

    try {
      const content = await fs.readFile(filePath, encoding);
      let lines = content.split(delimiter);
      
      if (skipEmpty) {
        lines = lines.filter(line => line.trim() !== '');
      }

      if (maxLines) {
        lines = lines.slice(0, maxLines);
      }

      const data = lines.map((line, index) => ({
        line_number: index + 1,
        content: line,
        length: line.length,
        word_count: line.split(/\s+/).filter(word => word.length > 0).length
      }));

      return {
        data,
        headers: ['line_number', 'content', 'length', 'word_count'],
        rowCount: data.length,
        columnCount: 4,
        filePath,
        fileType: 'text',
        encoding,
        statistics: {
          total_lines: data.length,
          total_characters: content.length,
          average_line_length: data.length > 0 ? content.length / data.length : 0,
          total_words: data.reduce((sum, line) => sum + line.word_count, 0)
        },
        metadata: {
          delimiter,
          skipEmpty,
          originalSize: content.length
        }
      };

    } catch (error) {
      throw new Error(`텍스트 파일 로드 실패: ${error.message}`);
    }
  }

  async loadParquet(filePath, options = {}) {
    if (!this.pythonExecutor) {
      throw new Error('Parquet 파일 로드를 위해서는 PythonExecutor가 필요합니다.');
    }

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

try:
    # Parquet 파일 로드
    read_options = {'engine': '${engine}'}
    
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['columns'] = ${JSON.stringify(columns)}
    
    df = pd.read_parquet('${filePath}', **read_options)
    
    # 행 수 제한
    if ${maxRows}:
        df = df.head(${maxRows})
    
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
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        },
        'metadata': {
            'engine': '${engine}',
            'selectedColumns': ${columns ? JSON.stringify(columns) : 'None'}
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

      const result = await this.pythonExecutor.execute(pythonCode);
      const parsedResult = JSON.parse(result.output);
      
      if (parsedResult.error) {
        throw new Error(parsedResult.error);
      }
      
      return parsedResult;

    } catch (error) {
      throw new Error(`Parquet 파일 로드 실패: ${error.message}`);
    }
  }

  async loadHDF5(filePath, options = {}) {
    if (!this.pythonExecutor) {
      throw new Error('HDF5 파일 로드를 위해서는 PythonExecutor가 필요합니다.');
    }

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

try:
    # HDF5 파일 구조 확인
    with h5py.File('${filePath}', '${mode}') as h5file:
        def get_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                return {
                    'type': 'dataset',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype)
                }
            elif isinstance(obj, h5py.Group):
                return {
                    'type': 'group',
                    'keys': list(obj.keys())
                }
        
        file_structure = {}
        h5file.visititems(lambda name, obj: file_structure.update({name: get_structure(name, obj)}))
    
    # 키 결정
    if ${key ? `'${key}'` : 'None'}:
        used_key = '${key}'
    else:
        # pandas가 저장한 기본 키들 시도
        possible_keys = ['data', 'df', 'table', list(file_structure.keys())[0] if file_structure else None]
        used_key = None
        for pk in possible_keys:
            if pk and pk in file_structure:
                used_key = pk
                break
        
        if not used_key:
            raise ValueError(f"키를 지정해주세요. 사용 가능한 키: {list(file_structure.keys())}")
    
    # 데이터 로드
    read_options = {'key': used_key, 'mode': '${mode}'}
    
    if ${columns ? JSON.stringify(columns) : 'None'}:
        read_options['columns'] = ${JSON.stringify(columns)}
    
    if ${start}:
        read_options['start'] = ${start}
    
    if ${stop}:
        read_options['stop'] = ${stop}
    
    if '${where}':
        read_options['where'] = '${where}'
    
    df = pd.read_hdf('${filePath}', **read_options)
    
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
    
    result = {
        'data': df.replace({np.nan: None}).to_dict('records'),
        'headers': df.columns.tolist(),
        'rowCount': len(df),
        'columnCount': len(df.columns),
        'filePath': '${filePath}',
        'fileType': 'hdf5',
        'dtypes': dtypes,
        'statistics': {
            'basic_stats': basic_stats,
            'missing_values': missing_info,
            'total_missing': sum(missing_info.values()),
            'missing_percentage': (sum(missing_info.values()) / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
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

      const result = await this.pythonExecutor.execute(pythonCode);
      const parsedResult = JSON.parse(result.output);
      
      if (parsedResult.error) {
        throw new Error(parsedResult.error);
      }
      
      return parsedResult;

    } catch (error) {
      throw new Error(`HDF5 파일 로드 실패: ${error.message}`);
    }
  }

  // 유틸리티 메서드들
  detectCSVDelimiter(line) {
    const delimiters = [',', ';', '\t', '|'];
    const counts = delimiters.map(delim => ({
      delimiter: delim,
      count: (line.match(new RegExp('\\' + delim, 'g')) || []).length
    }));
    
    counts.sort((a, b) => b.count - a.count);
    return counts[0].count > 0 ? counts[0].delimiter : ',';
  }

  parseCSVRow(row, delimiter) {
    const values = [];
    let currentValue = '';
    let inQuotes = false;
    
    for (let i = 0; i < row.length; i++) {
      const char = row[i];
      
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === delimiter && !inQuotes) {
        values.push(currentValue.trim());
        currentValue = '';
      } else {
        currentValue += char;
      }
    }
    
    values.push(currentValue.trim());
    return values;
  }

  inferAndConvertTypes(data, explicitTypes = null) {
    if (!data || data.length === 0) return data;
    
    const headers = Object.keys(data[0]);
    const typeMap = {};
    
    // 타입 추론
    headers.forEach(header => {
      if (explicitTypes && explicitTypes[header]) {
        typeMap[header] = explicitTypes[header];
      } else {
        typeMap[header] = this.inferColumnType(data, header);
      }
    });
    
    // 타입 변환
    return data.map(row => {
      const convertedRow = {};
      headers.forEach(header => {
        const value = row[header];
        convertedRow[header] = this.convertValue(value, typeMap[header]);
      });
      return convertedRow;
    });
  }

  // inferColumnType 메서드 계속
  inferColumnType(data, columnName) {
    const sampleSize = Math.min(100, data.length);
    const values = data.slice(0, sampleSize).map(row => row[columnName]).filter(val => val !== null && val !== '' && val !== undefined);
    
    if (values.length === 0) return 'string';
    
    let numericCount = 0;
    let integerCount = 0;
    let dateCount = 0;
    let booleanCount = 0;
    
    values.forEach(value => {
      const strValue = String(value).trim();
      
      // 숫자 타입 체크
      if (!isNaN(strValue) && strValue !== '') {
        numericCount++;
        if (Number.isInteger(parseFloat(strValue))) {
          integerCount++;
        }
      }
      
      // 날짜 타입 체크
      if (this.isDateString(strValue)) {
        dateCount++;
      }
      
      // 불린 타입 체크
      if (['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'].includes(strValue.toLowerCase())) {
        booleanCount++;
      }
    });
    
    const threshold = values.length * 0.8;
    
    if (booleanCount >= threshold) return 'boolean';
    if (dateCount >= threshold) return 'datetime';
    if (integerCount >= threshold) return 'integer';
    if (numericCount >= threshold) return 'float';
    
    return 'string';
  }

  convertValue(value, targetType) {
    if (value === null || value === '' || value === undefined) {
      return null;
    }
    
    const strValue = String(value).trim();
    
    try {
      switch (targetType) {
        case 'integer':
          return parseInt(strValue, 10);
        
        case 'float':
          return parseFloat(strValue);
        
        case 'boolean':
          const lowerValue = strValue.toLowerCase();
          if (['true', '1', 'yes', 'y'].includes(lowerValue)) return true;
          if (['false', '0', 'no', 'n'].includes(lowerValue)) return false;
          return Boolean(strValue);
        
        case 'datetime':
          return new Date(strValue).toISOString();
        
        default:
          return strValue;
      }
    } catch (error) {
      return strValue; // 변환 실패 시 원본 문자열 반환
    }
  }

  isDateString(value) {
    if (!value || typeof value !== 'string') return false;
    
    const datePatterns = [
      /^\d{4}-\d{2}-\d{2}$/,           // YYYY-MM-DD
      /^\d{2}\/\d{2}\/\d{4}$/,         // MM/DD/YYYY
      /^\d{2}-\d{2}-\d{4}$/,           // MM-DD-YYYY
      /^\d{4}\/\d{2}\/\d{2}$/,         // YYYY/MM/DD
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/ // ISO format
    ];
    
    return datePatterns.some(pattern => pattern.test(value)) && !isNaN(Date.parse(value));
  }

  flattenObject(obj, maxDepth = 10, currentDepth = 0, prefix = '') {
    if (currentDepth >= maxDepth) return obj;
    
    const flattened = {};
    
    for (const [key, value] of Object.entries(obj)) {
      const newKey = prefix ? `${prefix}.${key}` : key;
      
      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        Object.assign(flattened, this.flattenObject(value, maxDepth, currentDepth + 1, newKey));
      } else {
        flattened[newKey] = value;
      }
    }
    
    return flattened;
  }

  calculateBasicStatistics(data, headers) {
    if (!data || data.length === 0) {
      return { basic_stats: {}, missing_values: {}, summary: {} };
    }

    const statistics = {
      basic_stats: {},
      missing_values: {},
      summary: {
        row_count: data.length,
        column_count: headers ? headers.length : 0,
        total_cells: data.length * (headers ? headers.length : 0),
        non_null_cells: 0,
        data_types: {}
      }
    };

    if (!headers || headers.length === 0) return statistics;

    headers.forEach(column => {
      const values = data.map(row => row[column]).filter(val => val !== null && val !== '' && val !== undefined);
      const nonNullCount = values.length;
      const nullCount = data.length - nonNullCount;

      // 누락값 통계
      statistics.missing_values[column] = {
        count: nullCount,
        percentage: (nullCount / data.length) * 100
      };

      // 데이터 타입 추론
      const inferredType = this.inferColumnType(data, column);
      statistics.summary.data_types[column] = inferredType;

      // 기본 통계 (숫자형 컬럼만)
      if (['integer', 'float'].includes(inferredType) && values.length > 0) {
        const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
        
        if (numericValues.length > 0) {
          numericValues.sort((a, b) => a - b);
          
          const sum = numericValues.reduce((a, b) => a + b, 0);
          const mean = sum / numericValues.length;
          const variance = numericValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / numericValues.length;
          
          statistics.basic_stats[column] = {
            count: numericValues.length,
            mean: mean,
            std: Math.sqrt(variance),
            min: numericValues[0],
            max: numericValues[numericValues.length - 1],
            q25: this.percentile(numericValues, 0.25),
            q50: this.percentile(numericValues, 0.5),
            q75: this.percentile(numericValues, 0.75),
            sum: sum
          };
        }
      }

      // 문자열 통계
      if (inferredType === 'string' && values.length > 0) {
        const lengths = values.map(v => String(v).length);
        const uniqueValues = new Set(values);
        
        statistics.basic_stats[column] = {
          count: values.length,
          unique_count: uniqueValues.size,
          max_length: Math.max(...lengths),
          min_length: Math.min(...lengths),
          avg_length: lengths.reduce((a, b) => a + b, 0) / lengths.length
        };
      }

      statistics.summary.non_null_cells += nonNullCount;
    });

    return statistics;
  }

  percentile(sortedArray, p) {
    if (sortedArray.length === 0) return 0;
    
    const index = (sortedArray.length - 1) * p;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
      return sortedArray[lower];
    }
    
    return sortedArray[lower] * (upper - index) + sortedArray[upper] * (index - lower);
  }

  async postProcessResult(result, options) {
    // 데이터 후처리 옵션 적용
    let processedResult = { ...result };

    // 샘플링
    if (options.sampleSize && result.data.length > options.sampleSize) {
      const sampledData = this.sampleData(result.data, options.sampleSize, options.sampleMethod || 'random');
      processedResult.data = sampledData;
      processedResult.rowCount = sampledData.length;
      processedResult.metadata = {
        ...processedResult.metadata,
        sampled: true,
        originalRowCount: result.data.length,
        sampleSize: options.sampleSize,
        sampleMethod: options.sampleMethod || 'random'
      };
    }

    // 데이터 품질 정보 추가
    if (options.includeQuality !== false) {
      processedResult.quality = this.assessDataQuality(processedResult);
    }

    // 프로파일링 정보 추가
    if (options.includeProfile) {
      processedResult.profile = await this.generateDataProfile(processedResult);
    }

    return processedResult;
  }

  sampleData(data, sampleSize, method = 'random') {
    if (data.length <= sampleSize) return data;

    switch (method) {
      case 'random':
        const shuffled = [...data].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, sampleSize);
      
      case 'systematic':
        const step = Math.floor(data.length / sampleSize);
        return data.filter((_, index) => index % step === 0).slice(0, sampleSize);
      
      case 'stratified':
        // 간단한 계층 샘플링 (첫 번째 컬럼 기준)
        if (data.length === 0) return [];
        const firstColumn = Object.keys(data[0])[0];
        const groups = this.groupBy(data, firstColumn);
        const sampledGroups = Object.values(groups).map(group => 
          group.slice(0, Math.max(1, Math.floor(sampleSize / Object.keys(groups).length)))
        );
        return sampledGroups.flat().slice(0, sampleSize);
      
      default:
        return data.slice(0, sampleSize);
    }
  }

  groupBy(data, key) {
    return data.reduce((groups, item) => {
      const group = item[key] || 'undefined';
      groups[group] = groups[group] || [];
      groups[group].push(item);
      return groups;
    }, {});
  }

  assessDataQuality(result) {
    const quality = {
      overall_score: 100,
      issues: [],
      recommendations: [],
      metrics: {}
    };

    if (!result.data || result.data.length === 0) {
      quality.overall_score = 0;
      quality.issues.push('데이터가 없습니다.');
      return quality;
    }

    // 누락값 품질 평가
    if (result.statistics && result.statistics.missing_values) {
      const missingStats = Object.values(result.statistics.missing_values);
      const avgMissingPercent = missingStats.reduce((sum, stat) => sum + stat.percentage, 0) / missingStats.length;
      
      quality.metrics.missing_data_percentage = avgMissingPercent;
      
      if (avgMissingPercent > 50) {
        quality.overall_score -= 30;
        quality.issues.push('높은 누락값 비율 (50% 이상)');
        quality.recommendations.push('데이터 수집 방법을 검토하거나 누락값 처리 전략을 수립하세요.');
      } else if (avgMissingPercent > 20) {
        quality.overall_score -= 15;
        quality.issues.push('상당한 누락값 (20% 이상)');
        quality.recommendations.push('누락값 처리를 고려하세요.');
      }
    }

    // 데이터 일관성 평가
    const consistency = this.checkDataConsistency(result.data);
    quality.metrics.consistency_score = consistency.score;
    
    if (consistency.score < 0.8) {
      quality.overall_score -= 20;
      quality.issues.push('데이터 일관성 문제');
      quality.recommendations.push('데이터 형식을 표준화하세요.');
    }

    // 중복 데이터 평가
    const duplicates = this.findDuplicateRows(result.data);
    const duplicatePercent = (duplicates.length / result.data.length) * 100;
    quality.metrics.duplicate_percentage = duplicatePercent;
    
    if (duplicatePercent > 10) {
      quality.overall_score -= 15;
      quality.issues.push(`중복 행 발견 (${duplicatePercent.toFixed(1)}%)`);
      quality.recommendations.push('중복 제거를 고려하세요.');
    }

    quality.overall_score = Math.max(0, quality.overall_score);
    return quality;
  }

  checkDataConsistency(data) {
    if (data.length === 0) return { score: 1, issues: [] };

    const headers = Object.keys(data[0]);
    const issues = [];
    let totalScore = 0;

    headers.forEach(header => {
      const values = data.map(row => row[header]);
      const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
      
      if (nonNullValues.length === 0) return;

      // 데이터 타입 일관성 확인
      const typeConsistency = this.checkTypeConsistency(nonNullValues);
      totalScore += typeConsistency.score;
      
      if (typeConsistency.score < 0.8) {
        issues.push(`컬럼 '${header}': 타입 불일치`);
      }
    });

    return {
      score: totalScore / headers.length,
      issues
    };
  }

  checkTypeConsistency(values) {
    const types = values.map(value => typeof value);
    const typeCount = types.reduce((acc, type) => {
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    const dominantType = Object.keys(typeCount).reduce((a, b) => 
      typeCount[a] > typeCount[b] ? a : b
    );

    const consistency = typeCount[dominantType] / values.length;
    
    return {
      score: consistency,
      dominantType,
      typeDistribution: typeCount
    };
  }

  findDuplicateRows(data) {
    const seen = new Set();
    const duplicates = [];

    data.forEach((row, index) => {
      const rowKey = JSON.stringify(row);
      if (seen.has(rowKey)) {
        duplicates.push({ index, row });
      } else {
        seen.add(rowKey);
      }
    });

    return duplicates;
  }

  async generateDataProfile(result) {
    const profile = {
      overview: {
        shape: [result.rowCount, result.columnCount],
        memory_usage: this.estimateMemoryUsage(result.data),
        data_types: result.statistics?.summary?.data_types || {}
      },
      columns: {},
      correlations: {},
      patterns: {}
    };

    // 컬럼별 상세 프로파일
    if (result.headers) {
      for (const column of result.headers) {
        profile.columns[column] = await this.generateColumnProfile(result.data, column);
      }
    }

    // 상관관계 분석 (숫자형 컬럼만)
    const numericColumns = Object.entries(profile.overview.data_types)
      .filter(([_, type]) => ['integer', 'float'].includes(type))
      .map(([col, _]) => col);

    if (numericColumns.length > 1) {
      profile.correlations = this.calculateCorrelations(result.data, numericColumns);
    }

    return profile;
  }

  async generateColumnProfile(data, columnName) {
    const values = data.map(row => row[columnName]);
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    
    const profile = {
      count: values.length,
      non_null_count: nonNullValues.length,
      null_count: values.length - nonNullValues.length,
      null_percentage: ((values.length - nonNullValues.length) / values.length) * 100,
      unique_count: new Set(nonNullValues).size,
      data_type: this.inferColumnType(data, columnName)
    };

    // 타입별 상세 정보
    if (['integer', 'float'].includes(profile.data_type)) {
      const numericValues = nonNullValues.map(v => parseFloat(v)).filter(v => !isNaN(v));
      if (numericValues.length > 0) {
        profile.numeric_profile = this.generateNumericProfile(numericValues);
      }
    } else if (profile.data_type === 'string') {
      profile.text_profile = this.generateTextProfile(nonNullValues);
    }

    // 빈도 분석
    profile.value_counts = this.getValueCounts(nonNullValues, 10);

    return profile;
  }

  generateNumericProfile(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: mean,
      median: this.percentile(sorted, 0.5),
      std: Math.sqrt(variance),
      variance: variance,
      sum: sum,
      quantiles: {
        q25: this.percentile(sorted, 0.25),
        q50: this.percentile(sorted, 0.5),
        q75: this.percentile(sorted, 0.75)
      },
      outliers: this.detectOutliers(sorted)
    };
  }

  generateTextProfile(values) {
    const lengths = values.map(v => String(v).length);
    const words = values.flatMap(v => String(v).split(/\s+/));
    
    return {
      min_length: Math.min(...lengths),
      max_length: Math.max(...lengths),
      avg_length: lengths.reduce((a, b) => a + b, 0) / lengths.length,
      total_characters: lengths.reduce((a, b) => a + b, 0),
      word_count: words.length,
      avg_words: words.length / values.length,
      common_patterns: this.findCommonPatterns(values)
    };
  }

  detectOutliers(sortedValues) {
    if (sortedValues.length < 4) return [];

    const q1 = this.percentile(sortedValues, 0.25);
    const q3 = this.percentile(sortedValues, 0.75);
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    return sortedValues.filter(val => val < lowerBound || val > upperBound);
  }

  findCommonPatterns(values) {
    const patterns = {};
    
    values.forEach(value => {
      const str = String(value);
      
      // 길이 패턴
      const lengthPattern = `length_${str.length}`;
      patterns[lengthPattern] = (patterns[lengthPattern] || 0) + 1;
      
      // 첫 글자 패턴
      if (str.length > 0) {
        const firstChar = str[0].toLowerCase();
        const firstPattern = `starts_with_${firstChar}`;
        patterns[firstPattern] = (patterns[firstPattern] || 0) + 1;
      }
    });

    return Object.entries(patterns)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([pattern, count]) => ({ pattern, count, percentage: (count / values.length) * 100 }));
  }

  getValueCounts(values, limit = 10) {
    const counts = values.reduce((acc, val) => {
      acc[val] = (acc[val] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(counts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, limit)
      .map(([value, count]) => ({ 
        value, 
        count, 
        percentage: (count / values.length) * 100 
      }));
  }

  calculateCorrelations(data, numericColumns) {
    const correlations = {};
    
    for (let i = 0; i < numericColumns.length; i++) {
      for (let j = i + 1; j < numericColumns.length; j++) {
        const col1 = numericColumns[i];
        const col2 = numericColumns[j];
        
        const pairs = data
          .map(row => [parseFloat(row[col1]), parseFloat(row[col2])])
          .filter(([a, b]) => !isNaN(a) && !isNaN(b));
        
        if (pairs.length > 1) {
          const correlation = this.pearsonCorrelation(pairs);
          correlations[`${col1}_${col2}`] = {
            columns: [col1, col2],
            correlation: correlation,
            sample_size: pairs.length
          };
        }
      }
    }
    
    return correlations;
  }

  pearsonCorrelation(pairs) {
    const n = pairs.length;
    if (n === 0) return 0;

    const sumX = pairs.reduce((sum, [x, _]) => sum + x, 0);
    const sumY = pairs.reduce((sum, [_, y]) => sum + y, 0);
    const sumXY = pairs.reduce((sum, [x, y]) => sum + x * y, 0);
    const sumX2 = pairs.reduce((sum, [x, _]) => sum + x * x, 0);
    const sumY2 = pairs.reduce((sum, [_, y]) => sum + y * y, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  estimateMemoryUsage(data) {
    const jsonString = JSON.stringify(data);
    const bytes = new Blob([jsonString]).size;
    
    return {
      bytes: bytes,
      mb: bytes / (1024 * 1024),
      formatted: this.formatFileSize(bytes)
    };
  }

  formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  recordLoadHistory(filePath, format, executionTime, success, error = null) {
    const record = {
      filePath,
      format,
      executionTime,
      success,
      error: error?.message,
      timestamp: new Date().toISOString()
    };

    this.loadHistory.push(record);

    // 히스토리 크기 제한
    if (this.loadHistory.length > 100) {
      this.loadHistory = this.loadHistory.slice(-50);
    }
  }

  // 배치 로딩
  async loadMultipleFiles(filePaths, options = {}) {
    const results = [];
    const errors = [];

    for (const filePath of filePaths) {
      try {
        const result = await this.loadData(filePath, options);
        results.push(result);
      } catch (error) {
        errors.push({ filePath, error: error.message });
      }
    }

    return {
      results,
      errors,
      summary: {
        total: filePaths.length,
        successful: results.length,
        failed: errors.length,
        success_rate: (results.length / filePaths.length) * 100
      }
    };
  }

  // 형식 지원 확인
  async checkFormatSupport(filePath) {
    const extension = path.extname(filePath).toLowerCase();
    const format = this.detectFileFormat(extension);
    
    const support = {
      supported: this.supportedFormats.includes(format),
      format: format,
      requires_python: ['excel', 'parquet', 'hdf5'].includes(format),
      python_available: !!this.pythonExecutor
    };

    if (support.requires_python && !support.python_available) {
      support.warning = `${format} 파일을 로드하려면 PythonExecutor가 필요합니다.`;
    }

    return support;
  }

  // 통계 정보
  getLoadStatistics() {
    const recentLoads = this.loadHistory.slice(-20);
    
    return {
      total_loads: this.loadHistory.length,
      recent_loads: recentLoads,
      success_rate: this.loadHistory.length > 0 ? 
        (this.loadHistory.filter(h => h.success).length / this.loadHistory.length) * 100 : 0,
      average_load_time: recentLoads.length > 0 ?
        recentLoads.reduce((sum, h) => sum + h.executionTime, 0) / recentLoads.length : 0,
      format_statistics: this.getFormatStatistics()
    };
  }

  getFormatStatistics() {
    const formatCounts = this.loadHistory.reduce((acc, record) => {
      acc[record.format] = (acc[record.format] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(formatCounts)
      .map(([format, count]) => ({ format, count }))
      .sort((a, b) => b.count - a.count);
  }
}