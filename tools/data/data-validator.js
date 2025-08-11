// tools/data/data-validator.js
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';

export class DataValidator {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.validationConfig = null;
    
    this.initializeValidator();
  }

  async initializeValidator() {
    try {
      this.validationConfig = await this.configLoader.loadConfig('python-config.json');
      this.logger.info('DataValidator 초기화 완료');
    } catch (error) {
      this.logger.error('DataValidator 초기화 실패:', error);
      this.validationConfig = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      data_limits: {
        max_file_size_mb: 500,
        max_rows: 1000000,
        max_columns: 1000,
        max_memory_usage_mb: 1500,
        warn_threshold_mb: 1000
      },
      validation_rules: {
        check_missing_values: true,
        check_duplicates: true,
        check_data_types: true,
        check_outliers: true,
        check_encoding: true
      }
    };
  }

  async validateData(data, options = {}) {
    const {
      filePath = null,
      dataType = 'auto',
      checkDuplicates = true,
      checkMissing = true,
      checkTypes = true,
      checkOutliers = false,
      checkEncoding = true,
      generateReport = true,
      strictMode = false
    } = options;

    try {
      this.logger.info('데이터 검증 시작');

      // 데이터 타입에 따른 검증 방법 결정
      let validationResult;
      
      if (filePath) {
        validationResult = await this.validateFile(filePath, options);
      } else if (data) {
        validationResult = await this.validateDataObject(data, options);
      } else {
        throw new Error('데이터 또는 파일 경로가 필요합니다.');
      }

      // 결과 포맷팅
      const formattedResult = await this.formatValidationResult(validationResult, options);

      this.logger.info('데이터 검증 완료');
      return formattedResult;

    } catch (error) {
      this.logger.error('데이터 검증 실패:', error);
      throw error;
    }
  }

  async validateFile(filePath, options = {}) {
    const pythonScript = `
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_file(file_path, options):
    """파일 기반 데이터 검증"""
    result = {
        'file_info': {},
        'data_info': {},
        'validation_results': {},
        'issues': [],
        'recommendations': [],
        'quality_score': 0
    }
    
    try:
        # 파일 정보 수집
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        file_size = file_path.stat().st_size
        result['file_info'] = {
            'path': str(file_path),
            'name': file_path.name,
            'extension': file_path.suffix,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
        
        # 파일 크기 검증
        max_size_mb = options.get('max_file_size_mb', 500)
        if result['file_info']['size_mb'] > max_size_mb:
            result['issues'].append({
                'type': 'file_size',
                'severity': 'error',
                'message': f'파일 크기가 제한을 초과했습니다: {result["file_info"]["size_mb"]}MB > {max_size_mb}MB'
            })
            return result
        
        # 파일 형식에 따른 데이터 로드
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
                
        except UnicodeDecodeError:
            # 인코딩 문제 시 다른 인코딩 시도
            encodings = ['cp949', 'euc-kr', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    result['file_info']['detected_encoding'] = encoding
                    break
                except:
                    continue
            
            if df is None:
                raise ValueError("파일 인코딩을 감지할 수 없습니다.")
        
        # 데이터 기본 정보 수집
        result['data_info'] = {
            'shape': df.shape,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # 데이터 크기 검증
        max_rows = options.get('max_rows', 1000000)
        max_cols = options.get('max_columns', 1000)
        
        if result['data_info']['rows'] > max_rows:
            result['issues'].append({
                'type': 'data_size',
                'severity': 'warning',
                'message': f'행 수가 권장 한계를 초과했습니다: {result["data_info"]["rows"]} > {max_rows}'
            })
        
        if result['data_info']['columns'] > max_cols:
            result['issues'].append({
                'type': 'data_size',
                'severity': 'warning',
                'message': f'열 수가 권장 한계를 초과했습니다: {result["data_info"]["columns"]} > {max_cols}'
            })
        
        # 상세 검증 수행
        if options.get('check_missing', True):
            result['validation_results']['missing_values'] = check_missing_values(df)
        
        if options.get('check_duplicates', True):
            result['validation_results']['duplicates'] = check_duplicates(df)
        
        if options.get('check_types', True):
            result['validation_results']['data_types'] = check_data_types(df)
        
        if options.get('check_outliers', False):
            result['validation_results']['outliers'] = check_outliers(df)
        
        # 품질 점수 계산
        result['quality_score'] = calculate_quality_score(result)
        
        # 권장사항 생성
        result['recommendations'] = generate_recommendations(result)
        
        return result
        
    except Exception as e:
        result['issues'].append({
            'type': 'validation_error',
            'severity': 'error',
            'message': f'검증 중 오류 발생: {str(e)}'
        })
        return result

def check_missing_values(df):
    """결측값 검사"""
    missing_info = {
        'total_missing': int(df.isnull().sum().sum()),
        'missing_by_column': {},
        'missing_percentage': 0,
        'columns_with_missing': []
    }
    
    missing_by_col = df.isnull().sum()
    total_cells = len(df) * len(df.columns)
    
    for col, missing_count in missing_by_col.items():
        if missing_count > 0:
            missing_info['missing_by_column'][col] = {
                'count': int(missing_count),
                'percentage': round((missing_count / len(df)) * 100, 2)
            }
            missing_info['columns_with_missing'].append(col)
    
    missing_info['missing_percentage'] = round((missing_info['total_missing'] / total_cells) * 100, 2)
    
    return missing_info

def check_duplicates(df):
    """중복값 검사"""
    duplicate_info = {
        'total_duplicates': int(df.duplicated().sum()),
        'duplicate_percentage': 0,
        'duplicate_rows': []
    }
    
    duplicate_info['duplicate_percentage'] = round((duplicate_info['total_duplicates'] / len(df)) * 100, 2)
    
    if duplicate_info['total_duplicates'] > 0:
        # 중복 행의 인덱스 수집 (처음 10개만)
        duplicate_indices = df[df.duplicated()].index.tolist()[:10]
        duplicate_info['duplicate_rows'] = [int(idx) for idx in duplicate_indices]
    
    return duplicate_info

def check_data_types(df):
    """데이터 타입 검사"""
    type_info = {
        'dtypes_summary': {},
        'suggested_types': {},
        'type_issues': []
    }
    
    # 데이터 타입 요약
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        type_info['dtypes_summary'][str(dtype)] = {
            'count': len(cols),
            'columns': cols
        }
    
    # 타입 제안
    for col in df.columns:
        current_type = str(df[col].dtype)
        
        # 숫자형으로 변환 가능한지 확인
        if current_type == 'object':
            try:
                # NaN이 아닌 값들만 확인
                non_null_series = df[col].dropna()
                if len(non_null_series) > 0:
                    pd.to_numeric(non_null_series)
                    type_info['suggested_types'][col] = 'numeric'
            except:
                # 날짜형으로 변환 가능한지 확인
                try:
                    pd.to_datetime(non_null_series)
                    type_info['suggested_types'][col] = 'datetime'
                except:
                    pass
    
    return type_info

def check_outliers(df):
    """이상치 검사 (수치형 컬럼만)"""
    outlier_info = {
        'outliers_by_column': {},
        'total_outliers': 0
    }
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) > 0:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) > 0:
                outlier_info['outliers_by_column'][col] = {
                    'count': len(outliers),
                    'percentage': round((len(outliers) / len(series)) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'min_outlier': float(outliers.min()),
                    'max_outlier': float(outliers.max())
                }
                outlier_info['total_outliers'] += len(outliers)
    
    return outlier_info

def calculate_quality_score(result):
    """데이터 품질 점수 계산 (0-100)"""
    score = 100
    
    # 이슈 심각도에 따른 점수 차감
    for issue in result['issues']:
        if issue['severity'] == 'error':
            score -= 20
        elif issue['severity'] == 'warning':
            score -= 10
        elif issue['severity'] == 'info':
            score -= 5
    
    # 결측값에 따른 점수 차감
    if 'missing_values' in result['validation_results']:
        missing_pct = result['validation_results']['missing_values']['missing_percentage']
        if missing_pct > 20:
            score -= 15
        elif missing_pct > 10:
            score -= 10
        elif missing_pct > 5:
            score -= 5
    
    # 중복값에 따른 점수 차감
    if 'duplicates' in result['validation_results']:
        duplicate_pct = result['validation_results']['duplicates']['duplicate_percentage']
        if duplicate_pct > 10:
            score -= 10
        elif duplicate_pct > 5:
            score -= 5
    
    return max(0, score)  # 최소 0점

def generate_recommendations(result):
    """검증 결과에 기반한 권장사항 생성"""
    recommendations = []
    
    # 결측값 관련 권장사항
    if 'missing_values' in result['validation_results']:
        missing_info = result['validation_results']['missing_values']
        if missing_info['missing_percentage'] > 5:
            recommendations.append({
                'type': 'missing_values',
                'priority': 'high' if missing_info['missing_percentage'] > 20 else 'medium',
                'message': f"결측값이 {missing_info['missing_percentage']}% 발견되었습니다. 데이터 정제를 고려하세요.",
                'actions': ['결측값 처리 (제거/보간)', '데이터 수집 과정 점검']
            })
    
    # 중복값 관련 권장사항
    if 'duplicates' in result['validation_results']:
        duplicate_info = result['validation_results']['duplicates']
        if duplicate_info['duplicate_percentage'] > 1:
            recommendations.append({
                'type': 'duplicates',
                'priority': 'medium',
                'message': f"중복 행이 {duplicate_info['duplicate_percentage']}% 발견되었습니다.",
                'actions': ['중복 행 제거', '데이터 수집 과정 검토']
            })
    
    # 데이터 타입 관련 권장사항
    if 'data_types' in result['validation_results']:
        type_info = result['validation_results']['data_types']
        if type_info['suggested_types']:
            recommendations.append({
                'type': 'data_types',
                'priority': 'low',
                'message': "일부 컬럼의 데이터 타입 최적화가 가능합니다.",
                'actions': ['데이터 타입 변환', '메모리 사용량 최적화']
            })
    
    # 파일 크기 관련 권장사항
    if result['data_info']['memory_usage_mb'] > 500:
        recommendations.append({
            'type': 'performance',
            'priority': 'medium',
            'message': f"데이터 크기가 큽니다 ({result['data_info']['memory_usage_mb']}MB). 성능에 영향을 줄 수 있습니다.",
            'actions': ['데이터 샘플링', '청크 단위 처리', '불필요한 컬럼 제거']
        })
    
    return recommendations

# 메인 실행
if __name__ == "__main__":
    import sys
    import json
    
    try:
        # 인자 파싱
        file_path = "${filePath}"
        options = ${JSON.stringify(options)}
        
        # 검증 실행
        result = validate_file(file_path, options)
        
        # 결과 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            'error': True,
            'message': str(e),
            'traceback': str(e)
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)
`;

    try {
      const result = await this.pythonExecutor.execute(pythonScript, {
        timeout: 30000,
        cwd: './python'
      });

      return JSON.parse(result.stdout);
    } catch (error) {
      this.logger.error('파일 검증 실패:', error);
      throw error;
    }
  }

  async validateDataObject(data, options = {}) {
    // JavaScript 객체로 전달된 데이터 검증
    try {
      const validationResult = {
        data_info: {},
        validation_results: {},
        issues: [],
        recommendations: [],
        quality_score: 0
      };

      // 기본 데이터 정보
      if (Array.isArray(data)) {
        validationResult.data_info = {
          type: 'array',
          length: data.length,
          sample: data.slice(0, 3)
        };
      } else if (typeof data === 'object') {
        validationResult.data_info = {
          type: 'object',
          keys: Object.keys(data),
          key_count: Object.keys(data).length
        };
      } else {
        validationResult.data_info = {
          type: typeof data,
          value: data
        };
      }

      // 간단한 검증 로직
      if (Array.isArray(data) && data.length === 0) {
        validationResult.issues.push({
          type: 'empty_data',
          severity: 'warning',
          message: '데이터가 비어있습니다.'
        });
      }

      validationResult.quality_score = validationResult.issues.length === 0 ? 100 : 80;

      return validationResult;

    } catch (error) {
      this.logger.error('데이터 객체 검증 실패:', error);
      throw error;
    }
  }

  async checkDataTypes(data, options = {}) {
    const {
      suggestOptimization = true,
      checkConsistency = true
    } = options;

    try {
      if (!data || typeof data !== 'object') {
        throw new Error('유효한 데이터 객체가 필요합니다.');
      }

      const typeCheckResult = {
        column_types: {},
        type_issues: [],
        optimization_suggestions: []
      };

      // 컬럼별 타입 분석
      for (const [column, values] of Object.entries(data)) {
        if (Array.isArray(values)) {
          const typeAnalysis = this.analyzeColumnTypes(values);
          typeCheckResult.column_types[column] = typeAnalysis;

          // 타입 일관성 검사
          if (checkConsistency && typeAnalysis.mixed_types) {
            typeCheckResult.type_issues.push({
              column: column,
              issue: 'mixed_types',
              message: `${column} 컬럼에 혼합된 데이터 타입이 있습니다.`
            });
          }
        }
      }

      return typeCheckResult;

    } catch (error) {
      this.logger.error('데이터 타입 검사 실패:', error);
      throw error;
    }
  }

  analyzeColumnTypes(values) {
    const typeCount = {};
    const sampleValues = [];
    
    for (const value of values.slice(0, 100)) { // 샘플 100개만 검사
      const type = this.getDetailedType(value);
      typeCount[type] = (typeCount[type] || 0) + 1;
      
      if (sampleValues.length < 5) {
        sampleValues.push(value);
      }
    }

    const dominantType = Object.keys(typeCount).reduce((a, b) => 
      typeCount[a] > typeCount[b] ? a : b
    );

    return {
      dominant_type: dominantType,
      type_distribution: typeCount,
      mixed_types: Object.keys(typeCount).length > 1,
      sample_values: sampleValues,
      total_values: values.length
    };
  }

  getDetailedType(value) {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'number') {
      return Number.isInteger(value) ? 'integer' : 'float';
    }
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'string') {
      // 날짜 형식 확인
      if (/^\d{4}-\d{2}-\d{2}/.test(value)) return 'date';
      // 숫자 문자열 확인
      if (/^\d+$/.test(value)) return 'numeric_string';
      return 'string';
    }
    return typeof value;
  }

  async formatValidationResult(result, options = {}) {
    const {
      format = 'detailed',
      includeRecommendations = true,
      includeSummary = true
    } = options;

    try {
      const formatted = await this.resultFormatter.formatResult(result, {
        type: 'validation',
        format: format,
        metadata: {
          validation_timestamp: new Date().toISOString(),
          validator_version: '1.0.0'
        }
      });

      // 요약 정보 추가
      if (includeSummary) {
        formatted.summary = this.generateValidationSummary(result);
      }

      // 권장사항 포함
      if (includeRecommendations && result.recommendations) {
        formatted.recommendations = result.recommendations;
      }

      return formatted;

    } catch (error) {
      this.logger.error('검증 결과 포맷팅 실패:', error);
      return result;
    }
  }

  generateValidationSummary(result) {
    const summary = {
      overall_status: 'unknown',
      total_issues: 0,
      critical_issues: 0,
      data_quality: 'unknown'
    };

    if (result.issues) {
      summary.total_issues = result.issues.length;
      summary.critical_issues = result.issues.filter(
        issue => issue.severity === 'error'
      ).length;
    }

    // 전체 상태 결정
    if (summary.critical_issues > 0) {
      summary.overall_status = 'failed';
    } else if (summary.total_issues > 0) {
      summary.overall_status = 'warning';
    } else {
      summary.overall_status = 'passed';
    }

    // 데이터 품질 등급
    const qualityScore = result.quality_score || 0;
    if (qualityScore >= 90) {
      summary.data_quality = 'excellent';
    } else if (qualityScore >= 75) {
      summary.data_quality = 'good';
    } else if (qualityScore >= 60) {
      summary.data_quality = 'fair';
    } else {
      summary.data_quality = 'poor';
    }

    return summary;
  }

  // 빠른 검증 (기본적인 검사만)
  async quickValidate(data, options = {}) {
    try {
      const quickResult = {
        is_valid: true,
        basic_info: {},
        quick_checks: {},
        issues: []
      };

      // 데이터 존재 여부
      if (!data) {
        quickResult.is_valid = false;
        quickResult.issues.push('데이터가 없습니다.');
        return quickResult;
      }

      // 기본 정보
      if (Array.isArray(data)) {
        quickResult.basic_info = {
          type: 'array',
          length: data.length,
          empty: data.length === 0
        };
      } else if (typeof data === 'object') {
        quickResult.basic_info = {
          type: 'object',
          keys: Object.keys(data).length,
          empty: Object.keys(data).length === 0
        };
      }

      // 빠른 검사
      quickResult.quick_checks = {
        has_data: !!data && (Array.isArray(data) ? data.length > 0 : Object.keys(data).length > 0),
        data_type: typeof data,
        is_structured: Array.isArray(data) || (typeof data === 'object' && data !== null)
      };

      return quickResult;

    } catch (error) {
      this.logger.error('빠른 검증 실패:', error);
      return {
        is_valid: false,
        error: error.message,
        issues: ['검증 중 오류 발생']
      };
    }
  }

  // 스키마 검증
  async validateSchema(data, schema, options = {}) {
    try {
      const schemaResult = {
        schema_valid: true,
        schema_errors: [],
        missing_fields: [],
        extra_fields: [],
        type_mismatches: []
      };

      if (!schema || typeof schema !== 'object') {
        throw new Error('유효한 스키마가 필요합니다.');
      }

      // 필수 필드 검증
      for (const field of Object.keys(schema)) {
        if (!(field in data)) {
          schemaResult.missing_fields.push(field);
          schemaResult.schema_valid = false;
        }
      }

      // 타입 검증
      for (const [field, value] of Object.entries(data)) {
        if (field in schema) {
          const expectedType = schema[field];
          const actualType = typeof value;
          
          if (expectedType !== actualType) {
            schemaResult.type_mismatches.push({
              field: field,
              expected: expectedType,
              actual: actualType
            });
            schemaResult.schema_valid = false;
          }
        } else {
          schemaResult.extra_fields.push(field);
        }
      }

      return schemaResult;

    } catch (error) {
      this.logger.error('스키마 검증 실패:', error);
      throw error;
    }
  }
}