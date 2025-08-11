// tools/data/data-validator.js - 포괄적 데이터 검증 시스템
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import fs from 'fs/promises';
import path from 'path';

export class DataValidator {
  constructor() {
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.validationRules = null;
    this.supportedFormats = ['csv', 'xlsx', 'json', 'parquet', 'txt'];
    this.maxFileSize = 500 * 1024 * 1024; // 500MB
    this.validationHistory = [];
  }

  async initialize() {
    try {
      await this.configLoader.initialize();
      await this.loadValidationRules();
      this.logger.info('DataValidator 초기화 완료');
    } catch (error) {
      this.logger.error('DataValidator 초기화 실패:', error);
      throw error;
    }
  }

  async loadValidationRules() {
    try {
      // 분석 방법 설정에서 검증 규칙 로드
      const analysisConfig = this.configLoader.getConfig('analysis-methods');
      this.validationRules = analysisConfig?.validation || this.getDefaultValidationRules();
      
      this.logger.info('데이터 검증 규칙 로드 완료');
    } catch (error) {
      this.logger.warn('검증 규칙 로드 실패, 기본값 사용:', error);
      this.validationRules = this.getDefaultValidationRules();
    }
  }

  getDefaultValidationRules() {
    return {
      file: {
        max_size_mb: 500,
        min_size_bytes: 100,
        required_extensions: ['.csv', '.xlsx', '.json', '.txt', '.parquet'],
        encoding_check: true
      },
      data: {
        min_rows: 2,
        max_rows: 1000000,
        min_columns: 1,
        max_columns: 10000,
        missing_threshold: 0.9, // 90% 이상 누락 시 경고
        duplicate_threshold: 0.8 // 80% 이상 중복 시 경고
      },
      quality: {
        check_encoding: true,
        check_consistency: true,
        check_outliers: true,
        check_data_types: true,
        check_schema: true
      }
    };
  }

  async validateFile(filePath, options = {}) {
    const startTime = Date.now();
    
    try {
      this.logger.info(`파일 검증 시작: ${filePath}`);

      const validation = {
        filePath,
        timestamp: new Date().toISOString(),
        isValid: true,
        warnings: [],
        errors: [],
        suggestions: [],
        metadata: {},
        dataQuality: {},
        schema: {},
        statistics: {},
        executionTime: 0
      };

      // 1. 파일 존재 및 기본 검증
      await this.validateFileExists(filePath, validation);
      
      if (!validation.isValid) {
        return this.finalizeValidation(validation, startTime);
      }

      // 2. 파일 메타데이터 검증
      await this.validateFileMetadata(filePath, validation);

      // 3. 파일 형식 및 인코딩 검증
      await this.validateFileFormat(filePath, validation);

      // 4. 데이터 구조 검증
      await this.validateDataStructure(filePath, validation);

      // 5. 데이터 품질 검증
      await this.validateDataQuality(filePath, validation);

      // 6. 스키마 검증 (옵션)
      if (options.schema) {
        await this.validateSchema(filePath, validation, options.schema);
      }

      // 7. 비즈니스 규칙 검증 (옵션)
      if (options.businessRules) {
        await this.validateBusinessRules(filePath, validation, options.businessRules);
      }

      return this.finalizeValidation(validation, startTime);

    } catch (error) {
      this.logger.error(`파일 검증 실패: ${filePath}`, error);
      
      return {
        filePath,
        isValid: false,
        errors: [`검증 중 오류 발생: ${error.message}`],
        warnings: [],
        suggestions: ['파일 형식과 권한을 확인하세요.'],
        executionTime: Date.now() - startTime
      };
    }
  }

  async validateFileExists(filePath, validation) {
    try {
      const stats = await fs.stat(filePath);
      
      if (!stats.isFile()) {
        validation.isValid = false;
        validation.errors.push('지정된 경로가 파일이 아닙니다.');
        return;
      }

      validation.metadata.fileSize = stats.size;
      validation.metadata.lastModified = stats.mtime;
      validation.metadata.created = stats.birthtime;

      // 파일 크기 검증
      if (stats.size > this.maxFileSize) {
        validation.isValid = false;
        validation.errors.push(`파일 크기가 너무 큽니다: ${this.formatFileSize(stats.size)} (최대: ${this.formatFileSize(this.maxFileSize)})`);
      }

      if (stats.size < this.validationRules.file.min_size_bytes) {
        validation.warnings.push('파일이 너무 작습니다. 유효한 데이터가 포함되어 있는지 확인하세요.');
      }

    } catch (error) {
      validation.isValid = false;
      validation.errors.push(`파일에 접근할 수 없습니다: ${error.message}`);
    }
  }

  async validateFileMetadata(filePath, validation) {
    const extension = path.extname(filePath).toLowerCase();
    const fileName = path.basename(filePath);
    
    validation.metadata.extension = extension;
    validation.metadata.fileName = fileName;
    validation.metadata.format = this.detectFileFormat(extension);

    // 지원되는 형식인지 확인
    if (!this.validationRules.file.required_extensions.includes(extension)) {
      validation.warnings.push(`지원되지 않는 파일 형식: ${extension}`);
      validation.suggestions.push('CSV, Excel, JSON 형식을 사용하는 것을 권장합니다.');
    }

    // 파일명 패턴 검증
    if (fileName.includes(' ')) {
      validation.warnings.push('파일명에 공백이 포함되어 있습니다.');
      validation.suggestions.push('파일명에 언더스코어(_)나 하이픈(-)을 사용하는 것을 권장합니다.');
    }

    // 특수문자 검증
    const specialChars = /[!@#$%^&*()+=\[\]{};':"\\|,.<>?]/;
    if (specialChars.test(fileName)) {
      validation.warnings.push('파일명에 특수문자가 포함되어 있습니다.');
    }
  }

  async validateFileFormat(filePath, validation) {
    try {
      const extension = validation.metadata.extension;
      const format = validation.metadata.format;

      switch (format) {
        case 'csv':
          await this.validateCSVFormat(filePath, validation);
          break;
        case 'json':
          await this.validateJSONFormat(filePath, validation);
          break;
        case 'excel':
          await this.validateExcelFormat(filePath, validation);
          break;
        case 'text':
          await this.validateTextFormat(filePath, validation);
          break;
        default:
          validation.warnings.push(`알 수 없는 파일 형식: ${format}`);
      }

      // 인코딩 검증
      if (this.validationRules.quality.check_encoding) {
        await this.validateEncoding(filePath, validation);
      }

    } catch (error) {
      validation.errors.push(`파일 형식 검증 실패: ${error.message}`);
    }
  }

  async validateCSVFormat(filePath, validation) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());
      
      if (lines.length === 0) {
        validation.errors.push('CSV 파일이 비어있습니다.');
        validation.isValid = false;
        return;
      }

      // 헤더 검증
      const header = lines[0];
      const delimiter = this.detectCSVDelimiter(header);
      const columns = header.split(delimiter).map(col => col.trim().replace(/"/g, ''));

      validation.schema.columns = columns;
      validation.schema.delimiter = delimiter;
      validation.metadata.rowCount = lines.length - 1; // 헤더 제외
      validation.metadata.columnCount = columns.length;

      // 컬럼명 검증
      const duplicateColumns = this.findDuplicates(columns);
      if (duplicateColumns.length > 0) {
        validation.errors.push(`중복된 컬럼명: ${duplicateColumns.join(', ')}`);
        validation.isValid = false;
      }

      // 빈 컬럼명 검증
      const emptyColumns = columns.filter(col => !col || col.trim() === '');
      if (emptyColumns.length > 0) {
        validation.warnings.push('빈 컬럼명이 있습니다.');
        validation.suggestions.push('모든 컬럼에 의미있는 이름을 지정하세요.');
      }

      // 데이터 일관성 검증 (샘플링)
      const sampleSize = Math.min(100, lines.length - 1);
      const inconsistentRows = [];
      
      for (let i = 1; i <= sampleSize; i++) {
        const row = lines[i];
        const values = row.split(delimiter);
        
        if (values.length !== columns.length) {
          inconsistentRows.push(i + 1); // 1-based line number
        }
      }

      if (inconsistentRows.length > 0) {
        validation.warnings.push(`일관성 없는 행 발견: ${inconsistentRows.slice(0, 5).join(', ')}${inconsistentRows.length > 5 ? '...' : ''}`);
      }

    } catch (error) {
      validation.errors.push(`CSV 형식 검증 실패: ${error.message}`);
    }
  }

  async validateJSONFormat(filePath, validation) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(content);

      validation.metadata.dataType = Array.isArray(data) ? 'array' : 'object';
      validation.metadata.topLevelKeys = Array.isArray(data) ? 
        (data.length > 0 ? Object.keys(data[0]) : []) : 
        Object.keys(data);

      if (Array.isArray(data)) {
        validation.metadata.rowCount = data.length;
        validation.metadata.columnCount = data.length > 0 ? Object.keys(data[0]).length : 0;
        
        // 구조 일관성 검증
        if (data.length > 1) {
          const firstKeys = Object.keys(data[0]).sort();
          const inconsistentItems = [];
          
          for (let i = 1; i < Math.min(data.length, 100); i++) {
            const currentKeys = Object.keys(data[i]).sort();
            if (JSON.stringify(firstKeys) !== JSON.stringify(currentKeys)) {
              inconsistentItems.push(i);
            }
          }
          
          if (inconsistentItems.length > 0) {
            validation.warnings.push(`JSON 객체 구조가 일관되지 않음: 인덱스 ${inconsistentItems.slice(0, 5).join(', ')}`);
          }
        }
      }

    } catch (error) {
      validation.errors.push(`JSON 형식이 유효하지 않습니다: ${error.message}`);
      validation.isValid = false;
    }
  }

  async validateDataStructure(filePath, validation) {
    try {
      const format = validation.metadata.format;
      const rowCount = validation.metadata.rowCount || 0;
      const columnCount = validation.metadata.columnCount || 0;

      // 행 수 검증
      if (rowCount < this.validationRules.data.min_rows) {
        validation.warnings.push(`데이터 행이 너무 적습니다: ${rowCount}행 (최소: ${this.validationRules.data.min_rows}행)`);
      }

      if (rowCount > this.validationRules.data.max_rows) {
        validation.warnings.push(`데이터가 매우 큽니다: ${rowCount}행. 처리 시간이 오래 걸릴 수 있습니다.`);
        validation.suggestions.push('큰 데이터셋의 경우 샘플링을 고려하세요.');
      }

      // 열 수 검증
      if (columnCount < this.validationRules.data.min_columns) {
        validation.warnings.push(`컬럼이 너무 적습니다: ${columnCount}개`);
      }

      if (columnCount > this.validationRules.data.max_columns) {
        validation.warnings.push(`컬럼이 매우 많습니다: ${columnCount}개. 차원 축소를 고려하세요.`);
      }

      // 데이터 형태 분석
      validation.statistics.dataShape = {
        rows: rowCount,
        columns: columnCount,
        density: rowCount * columnCount,
        aspectRatio: columnCount > 0 ? rowCount / columnCount : 0
      };

    } catch (error) {
      validation.warnings.push(`데이터 구조 검증 중 오류: ${error.message}`);
    }
  }

  async validateDataQuality(filePath, validation) {
    try {
      const format = validation.metadata.format;

      if (format === 'csv') {
        await this.validateCSVDataQuality(filePath, validation);
      } else if (format === 'json') {
        await this.validateJSONDataQuality(filePath, validation);
      }

      // 전반적인 품질 점수 계산
      validation.dataQuality.overallScore = this.calculateQualityScore(validation);

    } catch (error) {
      validation.warnings.push(`데이터 품질 검증 중 오류: ${error.message}`);
    }
  }

  async validateCSVDataQuality(filePath, validation) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());
      
      if (lines.length <= 1) return;

      const delimiter = validation.schema.delimiter;
      const columns = validation.schema.columns;
      const sampleSize = Math.min(1000, lines.length - 1);
      
      const qualityMetrics = {
        missingValues: {},
        dataTypes: {},
        uniqueValues: {},
        outliers: {},
        duplicateRows: 0
      };

      // 샘플 데이터 분석
      const sampleData = [];
      for (let i = 1; i <= sampleSize; i++) {
        const values = lines[i].split(delimiter).map(v => v.trim().replace(/"/g, ''));
        sampleData.push(values);
      }

      // 각 컬럼별 품질 분석
      for (let colIndex = 0; colIndex < columns.length; colIndex++) {
        const columnName = columns[colIndex];
        const columnValues = sampleData.map(row => row[colIndex]);

        // 누락값 분석
        const missingCount = columnValues.filter(val => 
          !val || val === '' || val.toLowerCase() === 'null' || val.toLowerCase() === 'na'
        ).length;
        
        qualityMetrics.missingValues[columnName] = {
          count: missingCount,
          percentage: (missingCount / columnValues.length) * 100
        };

        // 데이터 타입 추론
        qualityMetrics.dataTypes[columnName] = this.inferDataType(columnValues);

        // 고유값 분석
        const uniqueValues = new Set(columnValues.filter(val => val && val !== ''));
        qualityMetrics.uniqueValues[columnName] = {
          count: uniqueValues.size,
          percentage: (uniqueValues.size / columnValues.length) * 100
        };

        // 수치형 컬럼의 이상치 분석
        if (qualityMetrics.dataTypes[columnName] === 'numeric') {
          const numericValues = columnValues
            .filter(val => val && !isNaN(val))
            .map(val => parseFloat(val));
          
          if (numericValues.length > 0) {
            qualityMetrics.outliers[columnName] = this.detectOutliers(numericValues);
          }
        }
      }

      // 중복 행 분석
      const rowHashes = new Set();
      let duplicates = 0;
      
      sampleData.forEach(row => {
        const rowHash = row.join('|');
        if (rowHashes.has(rowHash)) {
          duplicates++;
        } else {
          rowHashes.add(rowHash);
        }
      });
      
      qualityMetrics.duplicateRows = duplicates;

      validation.dataQuality = qualityMetrics;

      // 품질 경고 생성
      this.generateQualityWarnings(validation, qualityMetrics);

    } catch (error) {
      validation.warnings.push(`CSV 데이터 품질 분석 실패: ${error.message}`);
    }
  }

  generateQualityWarnings(validation, metrics) {
    // 누락값 경고
    Object.entries(metrics.missingValues).forEach(([column, stats]) => {
      if (stats.percentage > this.validationRules.data.missing_threshold * 100) {
        validation.warnings.push(`컬럼 '${column}'에 누락값이 ${stats.percentage.toFixed(1)}% 있습니다.`);
      }
    });

    // 중복 행 경고
    const duplicatePercentage = (metrics.duplicateRows / validation.metadata.rowCount) * 100;
    if (duplicatePercentage > this.validationRules.data.duplicate_threshold * 100) {
      validation.warnings.push(`중복 행이 ${duplicatePercentage.toFixed(1)}% 발견되었습니다.`);
      validation.suggestions.push('중복 제거를 고려하세요.');
    }

    // 고유값 분석
    Object.entries(metrics.uniqueValues).forEach(([column, stats]) => {
      if (stats.count === 1) {
        validation.warnings.push(`컬럼 '${column}'의 모든 값이 동일합니다.`);
        validation.suggestions.push(`'${column}' 컬럼 제거를 고려하세요.`);
      } else if (stats.percentage < 1) {
        validation.warnings.push(`컬럼 '${column}'에 고유값이 매우 적습니다 (${stats.percentage.toFixed(1)}%).`);
      }
    });

    // 이상치 경고
    Object.entries(metrics.outliers).forEach(([column, outlierInfo]) => {
      if (outlierInfo && outlierInfo.count > 0) {
        validation.warnings.push(`컬럼 '${column}'에서 ${outlierInfo.count}개의 이상치가 발견되었습니다.`);
      }
    });
  }

  // 유틸리티 메서드들
  detectFileFormat(extension) {
    const formatMap = {
      '.csv': 'csv',
      '.xlsx': 'excel',
      '.xls': 'excel',
      '.json': 'json',
      '.txt': 'text',
      '.tsv': 'csv',
      '.parquet': 'parquet'
    };
    
    return formatMap[extension] || 'unknown';
  }

  detectCSVDelimiter(headerLine) {
    const delimiters = [',', ';', '\t', '|'];
    const counts = delimiters.map(delim => ({
      delimiter: delim,
      count: (headerLine.match(new RegExp('\\' + delim, 'g')) || []).length
    }));
    
    counts.sort((a, b) => b.count - a.count);
    return counts[0].count > 0 ? counts[0].delimiter : ',';
  }

  inferDataType(values) {
    const nonEmptyValues = values.filter(val => val && val.trim() !== '');
    
    if (nonEmptyValues.length === 0) return 'empty';

    let numericCount = 0;
    let dateCount = 0;
    let booleanCount = 0;

    for (const value of nonEmptyValues.slice(0, 100)) { // 샘플링
      if (!isNaN(value) && !isNaN(parseFloat(value))) {
        numericCount++;
      } else if (this.isDateString(value)) {
        dateCount++;
      } else if (['true', 'false', '1', '0', 'yes', 'no'].includes(value.toLowerCase())) {
        booleanCount++;
      }
    }

    const total = nonEmptyValues.slice(0, 100).length;
    
    if (numericCount / total > 0.8) return 'numeric';
    if (dateCount / total > 0.8) return 'date';
    if (booleanCount / total > 0.8) return 'boolean';
    
    return 'text';
  }

  isDateString(value) {
    const date = new Date(value);
    return !isNaN(date.getTime()) && value.length > 6;
  }

  detectOutliers(values) {
    if (values.length < 4) return { count: 0, indices: [] };

    values.sort((a, b) => a - b);
    const q1 = values[Math.floor(values.length * 0.25)];
    const q3 = values[Math.floor(values.length * 0.75)];
    const iqr = q3 - q1;
    
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    const outliers = values.filter(val => val < lowerBound || val > upperBound);
    
    return {
      count: outliers.length,
      percentage: (outliers.length / values.length) * 100,
      bounds: { lower: lowerBound, upper: upperBound }
    };
  }

  findDuplicates(array) {
    const seen = new Set();
    const duplicates = new Set();
    
    array.forEach(item => {
      if (seen.has(item)) {
        duplicates.add(item);
      } else {
        seen.add(item);
      }
    });
    
    return Array.from(duplicates);
  }

  calculateQualityScore(validation) {
    let score = 100;
    
    // 오류마다 20점 감점
    score -= validation.errors.length * 20;
    
    // 경고마다 5점 감점
    score -= validation.warnings.length * 5;
    
    // 데이터 품질 기반 감점
    if (validation.dataQuality.missingValues) {
      Object.values(validation.dataQuality.missingValues).forEach(stats => {
        if (stats.percentage > 50) score -= 15;
        else if (stats.percentage > 20) score -= 10;
        else if (stats.percentage > 10) score -= 5;
      });
    }
    
    return Math.max(0, Math.min(100, score));
  }

  async validateEncoding(filePath, validation) {
    try {
      // UTF-8 읽기 시도
      await fs.readFile(filePath, 'utf-8');
      validation.metadata.encoding = 'UTF-8';
    } catch (error) {
      // 다른 인코딩 시도 (간단한 감지)
      try {
        await fs.readFile(filePath, 'latin1');
        validation.warnings.push('파일이 UTF-8이 아닌 인코딩을 사용할 수 있습니다.');
        validation.suggestions.push('UTF-8 인코딩으로 변환하는 것을 권장합니다.');
        validation.metadata.encoding = 'Latin1 (추정)';
      } catch (secondError) {
        validation.warnings.push('파일 인코딩을 확인할 수 없습니다.');
        validation.metadata.encoding = 'Unknown';
      }
    }
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

  finalizeValidation(validation, startTime) {
    validation.executionTime = Date.now() - startTime;
    
    // 검증 히스토리에 추가
    this.validationHistory.push({
      filePath: validation.filePath,
      timestamp: validation.timestamp,
      isValid: validation.isValid,
      score: validation.dataQuality.overallScore,
      executionTime: validation.executionTime
    });

    // 히스토리 크기 제한
    if (this.validationHistory.length > 100) {
      this.validationHistory = this.validationHistory.slice(-50);
    }

    this.logger.info(`파일 검증 완료: ${validation.filePath} (${validation.executionTime}ms)`);
    return validation;
  }

  // 배치 검증
  async validateMultipleFiles(filePaths, options = {}) {
    const results = [];
    
    for (const filePath of filePaths) {
      try {
        const result = await this.validateFile(filePath, options);
        results.push(result);
      } catch (error) {
        results.push({
          filePath,
          isValid: false,
          errors: [error.message],
          warnings: [],
          suggestions: []
        });
      }
    }

    return {
      files: results,
      summary: this.generateBatchSummary(results)
    };
  }

  generateBatchSummary(results) {
    const valid = results.filter(r => r.isValid).length;
    const invalid = results.length - valid;
    const avgScore = results.length > 0 ? 
      results.reduce((sum, r) => sum + (r.dataQuality?.overallScore || 0), 0) / results.length : 0;

    return {
      totalFiles: results.length,
      validFiles: valid,
      invalidFiles: invalid,
      validationRate: (valid / results.length) * 100,
      averageQualityScore: avgScore.toFixed(1),
      commonIssues: this.findCommonIssues(results)
    };
  }

  findCommonIssues(results) {
    const issueCounter = new Map();
    
    results.forEach(result => {
      [...result.errors, ...result.warnings].forEach(issue => {
        const key = issue.split(':')[0]; // 첫 번째 부분만 사용
        issueCounter.set(key, (issueCounter.get(key) || 0) + 1);
      });
    });

    return Array.from(issueCounter.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([issue, count]) => ({ issue, count }));
  }

  // 검증 통계
  getValidationStatistics() {
    const recent = this.validationHistory.slice(-10);
    
    return {
      totalValidations: this.validationHistory.length,
      recentValidations: recent,
      averageExecutionTime: recent.length > 0 ? 
        recent.reduce((sum, v) => sum + v.executionTime, 0) / recent.length : 0,
      successRate: recent.length > 0 ?
        (recent.filter(v => v.isValid).length / recent.length) * 100 : 0,
      averageScore: recent.length > 0 ?
        recent.reduce((sum, v) => sum + (v.score || 0), 0) / recent.length : 0
    };
  }
}