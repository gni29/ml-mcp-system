import { Logger } from '../../utils/logger.js';
import { FileManager } from '../common/file-manager.js';
import fs from 'fs/promises';
import path from 'path';

export class DataLoader {
  constructor() {
    this.logger = new Logger();
    this.fileManager = new FileManager();
    this.supportedFormats = ['.csv', '.xlsx', '.json', '.parquet', '.h5', '.txt'];
    this.cache = new Map();
    this.maxCacheSize = 50; // 최대 50개 파일 캐시
  }

  async loadData(filePath, options = {}) {
    try {
      // 캐시 확인
      const cacheKey = this.getCacheKey(filePath, options);
      if (this.cache.has(cacheKey)) {
        this.logger.info(`캐시에서 데이터 로드: ${filePath}`);
        return this.cache.get(cacheKey);
      }

      await this.fileManager.validateFile(filePath);
      
      const extension = path.extname(filePath).toLowerCase();
      let data;
      
      switch (extension) {
        case '.csv':
          data = await this.loadCSV(filePath, options);
          break;
        case '.xlsx':
        case '.xls':
          data = await this.loadExcel(filePath, options);
          break;
        case '.json':
          data = await this.loadJSON(filePath, options);
          break;
        case '.txt':
          data = await this.loadText(filePath, options);
          break;
        case '.parquet':
          data = await this.loadParquet(filePath, options);
          break;
        case '.h5':
          data = await this.loadHDF5(filePath, options);
          break;
        default:
          throw new Error(`지원하지 않는 파일 형식: ${extension}`);
      }

      // 데이터 후처리
      data = await this.postProcessData(data, options);

      // 캐시 저장
      this.setCache(cacheKey, data);

      return data;
    } catch (error) {
      this.logger.error('데이터 로드 실패:', error);
      throw error;
    }
  }

  async loadCSV(filePath, options = {}) {
    const {
      encoding = 'utf8',
      separator = ',',
      header = true,
      skipRows = 0,
      maxRows = null,
      columns = null,
      dtypes = null
    } = options;
    
    try {
      const data = await fs.readFile(filePath, encoding);
      const lines = data.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
      
      if (lines.length === 0) {
        throw new Error('빈 파일입니다.');
      }

      // 건너뛸 행 처리
      const dataLines = lines.slice(skipRows);
      
      if (dataLines.length === 0) {
        throw new Error('데이터가 없습니다.');
      }

      let headers;
      let dataRows;

      if (header) {
        headers = this.parseCSVLine(dataLines[0], separator);
        dataRows = dataLines.slice(1);
      } else {
        // 첫 번째 행을 기준으로 컬럼 수 파악
        const firstRowCols = this.parseCSVLine(dataLines[0], separator);
        headers = firstRowCols.map((_, index) => `column_${index}`);
        dataRows = dataLines;
      }

      // 선택된 컬럼만 사용
      let selectedColumns = headers;
      let columnIndices = headers.map((_, index) => index);
      
      if (columns) {
        selectedColumns = columns;
        columnIndices = columns.map(col => headers.indexOf(col))
          .filter(index => index !== -1);
      }

      // 최대 행 수 제한
      if (maxRows && dataRows.length > maxRows) {
        dataRows = dataRows.slice(0, maxRows);
      }

      // 데이터 파싱
      const rows = dataRows.map((line, lineIndex) => {
        try {
          const values = this.parseCSVLine(line, separator);
          const row = {};
          
          selectedColumns.forEach((header, index) => {
            const colIndex = columnIndices[index];
            let value = values[colIndex] || null;
            
            // 데이터 타입 변환
            if (dtypes && dtypes[header]) {
              value = this.convertDataType(value, dtypes[header]);
            } else {
              value = this.inferAndConvertDataType(value);
            }
            
            row[header] = value;
          });
          
          return row;
        } catch (error) {
          this.logger.warn(`CSV 라인 파싱 실패 (${lineIndex + 1}):`, error);
          return null;
        }
      }).filter(row => row !== null);

      // 데이터 통계 계산
      const statistics = this.calculateStatistics(rows, selectedColumns);

      return {
        data: rows,
        headers: selectedColumns,
        rowCount: rows.length,
        columnCount: selectedColumns.length,
        filePath: filePath,
        fileType: 'csv',
        statistics: statistics,
        metadata: {
          originalHeaders: headers,
          encoding: encoding,
          separator: separator,
          hasHeader: header,
          skippedRows: skipRows
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
      const data = await fs.readFile(filePath, encoding);
      const jsonData = JSON.parse(data);
      
      let processedData;
      
      if (arrayPath) {
        // 특정 경로의 배열 추출
        processedData = this.extractArrayFromPath(jsonData, arrayPath);
      } else if (Array.isArray(jsonData)) {
        processedData = jsonData;
      } else if (typeof jsonData === 'object' && jsonData !== null) {
        processedData = [jsonData];
      } else {
        throw new Error('JSON 데이터가 객체 또는 배열이 아닙니다.');
      }

      // 플래트닝 처리
      if (flatten) {
        processedData = processedData.map(item =>
          this.flattenObject(item, '', maxDepth)
        );
      }

      // 헤더 추출
      const headers = this.extractJSONHeaders(processedData);
      
      // 데이터 정규화
      const normalizedData = processedData.map(item => {
        const row = {};
        headers.forEach(header => {
          row[header] = this.getNestedValue(item, header);
        });
        return row;
      });

      // 데이터 통계 계산
      const statistics = this.calculateStatistics(normalizedData, headers);

      return {
        data: normalizedData,
        headers: headers,
        rowCount: normalizedData.length,
        columnCount: headers.length,
        filePath: filePath,
        fileType: 'json',
        statistics: statistics,
        metadata: {
          originalStructure: Array.isArray(jsonData) ? 'array' : 'object',
          flattened: flatten,
          arrayPath: arrayPath,
          encoding: encoding
        }
      };
    } catch (error) {
      throw new Error(`JSON 파일 로드 실패: ${error.message}`);
    }
  }

  async loadExcel(filePath, options = {}) {
    const {
      sheetName = null,
      header = true,
      skipRows = 0,
      maxRows = null,
      columns = null
    } = options;

    try {
      // Excel 파일 로드는 외부 라이브러리가 필요하므로 기본 구현
      throw new Error('Excel 파일 로드 기능은 별도 라이브러리(xlsx) 설치가 필요합니다.');
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
      const data = await fs.readFile(filePath, encoding);
      let lines = data.split(delimiter);

      if (skipEmpty) {
        lines = lines.filter(line => line.trim().length > 0);
      }

      if (maxLines) {
        lines = lines.slice(0, maxLines);
      }

      const processedData = lines.map((line, index) => ({
        line_number: index + 1,
        content: line,
        length: line.length,
        word_count: line.split(/\s+/).filter(word => word.length > 0).length
      }));

      return {
        data: processedData,
        headers: ['line_number', 'content', 'length', 'word_count'],
        rowCount: processedData.length,
        columnCount: 4,
        filePath: filePath,
        fileType: 'text',
        statistics: {
          totalLines: processedData.length,
          totalCharacters: data.length,
          totalWords: processedData.reduce((sum, row) => sum + row.word_count, 0),
          averageLineLength: processedData.reduce((sum, row) => sum + row.length, 0) / processedData.length
        },
        metadata: {
          encoding: encoding,
          delimiter: delimiter,
          skipEmpty: skipEmpty
        }
      };
    } catch (error) {
      throw new Error(`텍스트 파일 로드 실패: ${error.message}`);
    }
  }

  async loadParquet(filePath, options = {}) {
    try {
      throw new Error('Parquet 파일 로드 기능은 별도 라이브러리 설치가 필요합니다.');
    } catch (error) {
      throw new Error(`Parquet 파일 로드 실패: ${error.message}`);
    }
  }

  async loadHDF5(filePath, options = {}) {
    try {
      throw new Error('HDF5 파일 로드 기능은 별도 라이브러리 설치가 필요합니다.');
    } catch (error) {
      throw new Error(`HDF5 파일 로드 실패: ${error.message}`);
    }
  }

  async postProcessData(data, options = {}) {
    const {
      removeNulls = false,
      trimStrings = true,
      convertTypes = true,
      validateData = true
    } = options;

    try {
      let processedData = { ...data };

      if (removeNulls) {
        processedData.data = processedData.data.filter(row =>
          Object.values(row).some(value => value !== null && value !== undefined)
        );
      }

      if (trimStrings) {
        processedData.data = processedData.data.map(row => {
          const trimmedRow = {};
          Object.entries(row).forEach(([key, value]) => {
            trimmedRow[key] = typeof value === 'string' ? value.trim() : value;
          });
          return trimmedRow;
        });
      }

      if (validateData) {
        const validation = this.validateDataset(processedData);
        processedData.validation = validation;
      }

      // 행 수 업데이트
      processedData.rowCount = processedData.data.length;

      return processedData;
    } catch (error) {
      this.logger.error('데이터 후처리 실패:', error);
      throw error;
    }
  }

  parseCSVLine(line, separator = ',') {
    const values = [];
    let current = '';
    let inQuotes = false;
    let quoteChar = null;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (!inQuotes && (char === '"' || char === "'")) {
        inQuotes = true;
        quoteChar = char;
      } else if (inQuotes && char === quoteChar) {
        if (line[i + 1] === quoteChar) {
          current += char;
          i++; // 다음 따옴표 건너뛰기
        } else {
          inQuotes = false;
          quoteChar = null;
        }
      } else if (!inQuotes && char === separator) {
        values.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    values.push(current);
    return values;
  }

  convertDataType(value, dtype) {
    if (value === null || value === undefined || value === '') {
      return null;
    }

    try {
      switch (dtype.toLowerCase()) {
        case 'int':
        case 'integer':
          return parseInt(value, 10);
        case 'float':
        case 'number':
          return parseFloat(value);
        case 'bool':
        case 'boolean':
          return value.toLowerCase() === 'true' || value === '1';
        case 'string':
        case 'str':
          return String(value);
        case 'date':
          return new Date(value);
        default:
          return value;
      }
    } catch (error) {
      this.logger.warn(`데이터 타입 변환 실패: ${value} -> ${dtype}`);
      return value;
    }
  }

  inferAndConvertDataType(value) {
    if (value === null || value === undefined || value === '') {
      return null;
    }

    const strValue = String(value).trim();
    
    // 불린 값 확인
    if (strValue.toLowerCase() === 'true' || strValue.toLowerCase() === 'false') {
      return strValue.toLowerCase() === 'true';
    }
    
    // 숫자 확인
    if (/^-?\d+$/.test(strValue)) {
      return parseInt(strValue, 10);
    }
    
    if (/^-?\d+\.\d+$/.test(strValue)) {
      return parseFloat(strValue);
    }
    
    // 날짜 확인
    if (this.isDateString(strValue)) {
      const date = new Date(strValue);
      if (!isNaN(date.getTime())) {
        return date;
      }
    }
    
    return strValue;
  }

  isDateString(value) {
    const datePatterns = [
      /^\d{4}-\d{2}-\d{2}$/,           // YYYY-MM-DD
      /^\d{2}\/\d{2}\/\d{4}$/,         // MM/DD/YYYY
      /^\d{2}-\d{2}-\d{4}$/,           // MM-DD-YYYY
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/ // ISO format
    ];
    
    return datePatterns.some(pattern => pattern.test(value));
  }

  extractArrayFromPath(data, path) {
    const keys = path.split('.');
    let current = data;
    
    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        throw new Error(`경로 '${path}'에서 데이터를 찾을 수 없습니다.`);
      }
    }
    
    if (!Array.isArray(current)) {
      throw new Error(`경로 '${path}'의 데이터가 배열이 아닙니다.`);
    }
    
    return current;
  }

  flattenObject(obj, prefix = '', maxDepth = 10) {
    if (maxDepth <= 0) return obj;
    
    const flattened = {};
    
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const newKey = prefix ? `${prefix}.${key}` : key;
        
        if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
          Object.assign(flattened, this.flattenObject(obj[key], newKey, maxDepth - 1));
        } else {
          flattened[newKey] = obj[key];
        }
      }
    }
    
    return flattened;
  }

  extractJSONHeaders(data) {
    const headers = new Set();
    
    data.forEach(item => {
      if (typeof item === 'object' && item !== null) {
        Object.keys(item).forEach(key => headers.add(key));
      }
    });
    
    return Array.from(headers);
  }

  getNestedValue(obj, path) {
    const keys = path.split('.');
    let current = obj;
    
    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        return null;
      }
    }
    
    return current;
  }

  calculateStatistics(data, headers) {
    const stats = {};
    
    headers.forEach(header => {
      const values = data.map(row => row[header]).filter(val => val !== null && val !== undefined);
      
      if (values.length === 0) {
        stats[header] = { count: 0, type: 'empty' };
        return;
      }
      
      const firstValue = values[0];
      const valueType = typeof firstValue;
      
      stats[header] = {
        count: values.length,
        nullCount: data.length - values.length,
        type: valueType,
        unique: new Set(values).size
      };
      
      if (valueType === 'number') {
        const numValues = values.filter(v => typeof v === 'number');
        if (numValues.length > 0) {
          stats[header].min = Math.min(...numValues);
          stats[header].max = Math.max(...numValues);
          stats[header].mean = numValues.reduce((sum, val) => sum + val, 0) / numValues.length;
          stats[header].median = this.calculateMedian(numValues);
        }
      } else if (valueType === 'string') {
        const lengths = values.map(v => String(v).length);
        stats[header].minLength = Math.min(...lengths);
        stats[header].maxLength = Math.max(...lengths);
        stats[header].avgLength = lengths.reduce((sum, len) => sum + len, 0) / lengths.length;
      }
    });
    
    return stats;
  }

  calculateMedian(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
      return (sorted[middle - 1] + sorted[middle]) / 2;
    } else {
      return sorted[middle];
    }
  }

  validateDataset(dataset) {
    const validation = {
      isValid: true,
      warnings: [],
      errors: [],
      suggestions: []
    };

    // 기본 검증
    if (!dataset.data || dataset.data.length === 0) {
      validation.isValid = false;
      validation.errors.push('데이터가 없습니다.');
      return validation;
    }

    // 컬럼 일관성 검증
    const expectedColumns = dataset.headers;
    const inconsistentRows = dataset.data.filter(row => {
      const rowKeys = Object.keys(row);
      return rowKeys.length !== expectedColumns.length ||
             !expectedColumns.every(col => rowKeys.includes(col));
    });

    if (inconsistentRows.length > 0) {
      validation.warnings.push(`${inconsistentRows.length}개 행의 컬럼 구조가 일관되지 않습니다.`);
    }

    // 데이터 품질 검증
    const nullPercentage = this.calculateNullPercentage(dataset);
    if (nullPercentage > 50) {
      validation.warnings.push(`데이터의 ${nullPercentage.toFixed(1)}%가 null 값입니다.`);
    }

    // 제안사항
    if (dataset.rowCount < 100) {
      validation.suggestions.push('데이터 양이 적습니다. 더 많은 데이터를 수집하는 것을 고려해보세요.');
    }

    return validation;
  }

  calculateNullPercentage(dataset) {
    const totalCells = dataset.rowCount * dataset.columnCount;
    let nullCells = 0;

    dataset.data.forEach(row => {
      Object.values(row).forEach(value => {
        if (value === null || value === undefined) {
          nullCells++;
        }
      });
    });

    return (nullCells / totalCells) * 100;
  }

  getCacheKey(filePath, options) {
    return `${filePath}:${JSON.stringify(options)}`;
  }

  setCache(key, data) {
    // 캐시 크기 제한
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, data);
  }

  clearCache() {
    this.cache.clear();
    this.logger.info('데이터 캐시 정리 완료');
  }

  async loadMultipleFiles(filePaths, options = {}) {
    const results = [];
    const errors = [];

    for (const filePath of filePaths) {
      try {
        const data = await this.loadData(filePath, options);
        results.push(data);
      } catch (error) {
        errors.push({ filePath, error: error.message });
      }
    }

    return {
      results,
      errors,
      successCount: results.length,
      errorCount: errors.length
    };
  }

  async getDataPreview(filePath, options = {}) {
    const previewOptions = {
      ...options,
      maxRows: options.maxRows || 5
    };

    try {
      const data = await this.loadData(filePath, previewOptions);
      return {
        ...data,
        isPreview: true,
        previewRows: previewOptions.maxRows
      };
    } catch (error) {
      throw new Error(`데이터 미리보기 실패: ${error.message}`);
    }
  }
}
