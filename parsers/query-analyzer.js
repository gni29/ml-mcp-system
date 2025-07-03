import { Logger } from '../../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class QueryAnalyzer {
  constructor() {
    this.logger = new Logger();
    this.patterns = {
      // 파일 참조 패턴
      file_patterns: {
        explicit: /(?:파일|file|데이터)\s*[:=]?\s*['"]([^'"]+)['"]?/gi,
        implicit: /\.(?:csv|xlsx|json|txt|parquet|h5)(?:\s|$)/gi,
        path: /(?:\.\/|\/|\\)([^\\\/\s]+\.(?:csv|xlsx|json|txt|parquet|h5))/gi
      },
      
      // 컬럼 참조 패턴
      column_patterns: {
        explicit: /(?:컬럼|column|필드|field)\s*[:=]?\s*['"]([^'"]+)['"]?/gi,
        target: /(?:목표|target|타겟|결과|결과값)\s*[:=]?\s*['"]([^'"]+)['"]?/gi,
        features: /(?:특성|feature|변수|입력)\s*[:=]?\s*\[([^\]]+)\]/gi
      },
      
      // 숫자 패턴
      numeric_patterns: {
        integer: /\b(\d+)\b/g,
        float: /\b(\d+\.\d+)\b/g,
        percentage: /\b(\d+(?:\.\d+)?)\s*%/g,
        range: /\b(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)/g
      },
      
      // 파라미터 패턴
      parameter_patterns: {
        model: /(?:모델|model)\s*[:=]?\s*['"]([^'"]+)['"]?/gi,
        algorithm: /(?:알고리즘|algorithm)\s*[:=]?\s*['"]([^'"]+)['"]?/gi,
        method: /(?:방법|method)\s*[:=]?\s*['"]([^'"]+)['"]?/gi
      }
    };
    
    this.filePatterns = {
      csv: /\.csv$/i,
      excel: /\.xlsx?$/i,
      json: /\.json$/i,
      text: /\.txt$/i,
      image: /\.(jpg|jpeg|png|gif|bmp|tiff)$/i,
      parquet: /\.parquet$/i,
      hdf5: /\.h5$/i
    };
  }

  async analyzeQuery(query) {
    try {
      const analysis = {
        original_query: query,
        timestamp: new Date().toISOString(),
        file_references: this.extractFileReferences(query),
        column_references: this.extractColumnReferences(query),
        numeric_values: this.extractNumericValues(query),
        parameters: this.extractParameters(query),
        data_requirements: this.analyzeDataRequirements(query),
        detected_files: await this.detectAvailableFiles(),
        context: this.analyzeContext(query),
        validation: this.validateQuery(query),
        resolved_references: {},
        suggested_corrections: []
      };

      // 참조 해결
      analysis.resolved_references = await this.resolveReferences(analysis);
      
      // 수정 제안
      analysis.suggested_corrections = this.suggestCorrections(analysis);

      return analysis;
    } catch (error) {
      this.logger.error('쿼리 분석 실패:', error);
      return this.getErrorAnalysis(query, error);
    }
  }

  extractFileReferences(query) {
    const references = {
      explicit: [],
      implicit: [],
      paths: []
    };

    // 명시적 파일 참조
    let match;
    while ((match = this.patterns.file_patterns.explicit.exec(query)) !== null) {
      references.explicit.push({
        filename: match[1],
        position: match.index,
        confidence: 0.9
      });
    }

    // 암시적 파일 참조 (확장자 기반)
    this.patterns.file_patterns.implicit.lastIndex = 0;
    while ((match = this.patterns.file_patterns.implicit.exec(query)) !== null) {
      const extension = match[0].trim();
      references.implicit.push({
        extension: extension,
        position: match.index,
        confidence: 0.6
      });
    }

    // 경로 참조
    this.patterns.file_patterns.path.lastIndex = 0;
    while ((match = this.patterns.file_patterns.path.exec(query)) !== null) {
      references.paths.push({
        path: match[0],
        filename: match[1],
        position: match.index,
        confidence: 0.8
      });
    }

    return references;
  }

  extractColumnReferences(query) {
    const references = {
      explicit: [],
      target: [],
      feature_lists: []
    };

    // 명시적 컬럼 참조
    let match;
    while ((match = this.patterns.column_patterns.explicit.exec(query)) !== null) {
      references.explicit.push({
        column: match[1],
        position: match.index,
        confidence: 0.9
      });
    }

    // 타겟 컬럼
    this.patterns.column_patterns.target.lastIndex = 0;
    while ((match = this.patterns.column_patterns.target.exec(query)) !== null) {
      references.target.push({
        column: match[1],
        position: match.index,
        confidence: 0.8
      });
    }

    // 특성 리스트
    this.patterns.column_patterns.features.lastIndex = 0;
    while ((match = this.patterns.column_patterns.features.exec(query)) !== null) {
      const features = match[1].split(',').map(f => f.trim().replace(/['"]/g, ''));
      references.feature_lists.push({
        features: features,
        position: match.index,
        confidence: 0.7
      });
    }

    return references;
  }

  extractNumericValues(query) {
    const values = {
      integers: [],
      floats: [],
      percentages: [],
      ranges: []
    };

    // 정수
    let match;
    while ((match = this.patterns.numeric_patterns.integer.exec(query)) !== null) {
      values.integers.push({
        value: parseInt(match[1]),
        position: match.index,
        raw: match[1]
      });
    }

    // 실수
    this.patterns.numeric_patterns.float.lastIndex = 0;
    while ((match = this.patterns.numeric_patterns.float.exec(query)) !== null) {
      values.floats.push({
        value: parseFloat(match[1]),
        position: match.index,
        raw: match[1]
      });
    }

    // 백분율
    this.patterns.numeric_patterns.percentage.lastIndex = 0;
    while ((match = this.patterns.numeric_patterns.percentage.exec(query)) !== null) {
      values.percentages.push({
        value: parseFloat(match[1]),
        position: match.index,
        raw: match[1]
      });
    }

    // 범위
    this.patterns.numeric_patterns.range.lastIndex = 0;
    while ((match = this.patterns.numeric_patterns.range.exec(query)) !== null) {
      values.ranges.push({
        min: parseFloat(match[1]),
        max: parseFloat(match[2]),
        position: match.index,
        raw: match[0]
      });
    }

    return values;
  }

  extractParameters(query) {
    const parameters = {
      model: {},
      visualization: {},
      data: {}
    };

    // 모델 파라미터
    let match;
    while ((match = this.patterns.parameter_patterns.model.exec(query)) !== null) {
      parameters.model.type = match[1];
    }

    // 알고리즘 파라미터
    this.patterns.parameter_patterns.algorithm.lastIndex = 0;
    while ((match = this.patterns.parameter_patterns.algorithm.exec(query)) !== null) {
      parameters.model.algorithm = match[1];
    }

    // 방법 파라미터
    this.patterns.parameter_patterns.method.lastIndex = 0;
    while ((match = this.patterns.parameter_patterns.method.exec(query)) !== null) {
      parameters.model.method = match[1];
    }

    // 컨텍스트 기반 파라미터 추론
    this.inferParameters(query, parameters);

    return parameters;
  }

  inferParameters(query, parameters) {
    const lowerQuery = query.toLowerCase();

    // 머신러닝 모델 추론
    if (lowerQuery.includes('분류') || lowerQuery.includes('classification')) {
      parameters.model.task = 'classification';
    } else if (lowerQuery.includes('회귀') || lowerQuery.includes('regression')) {
      parameters.model.task = 'regression';
    } else if (lowerQuery.includes('클러스터') || lowerQuery.includes('cluster')) {
      parameters.model.task = 'clustering';
    }

    // 시각화 유형 추론
    if (lowerQuery.includes('산점도') || lowerQuery.includes('scatter')) {
      parameters.visualization.type = 'scatter';
    } else if (lowerQuery.includes('히스토그램') || lowerQuery.includes('histogram')) {
      parameters.visualization.type = 'histogram';
    } else if (lowerQuery.includes('박스플롯') || lowerQuery.includes('boxplot')) {
      parameters.visualization.type = 'boxplot';
    } else if (lowerQuery.includes('히트맵') || lowerQuery.includes('heatmap')) {
      parameters.visualization.type = 'heatmap';
    }

    // 데이터 처리 파라미터
    if (lowerQuery.includes('정규화') || lowerQuery.includes('normalize')) {
      parameters.data.normalize = true;
    }
    if (lowerQuery.includes('스케일') || lowerQuery.includes('scale')) {
      parameters.data.scale = true;
    }
  }

  analyzeDataRequirements(query) {
    const requirements = {
      file_type: 'unknown',
      data_type: 'unknown',
      preprocessing: [],
      validation: []
    };

    const lowerQuery = query.toLowerCase();

    // 파일 타입 추론
    if (lowerQuery.includes('csv')) {
      requirements.file_type = 'csv';
    } else if (lowerQuery.includes('excel') || lowerQuery.includes('xlsx')) {
      requirements.file_type = 'excel';
    } else if (lowerQuery.includes('json')) {
      requirements.file_type = 'json';
    }

    // 데이터 타입 추론
    if (lowerQuery.includes('숫자') || lowerQuery.includes('numeric')) {
      requirements.data_type = 'numeric';
    } else if (lowerQuery.includes('텍스트') || lowerQuery.includes('text')) {
      requirements.data_type = 'text';
    } else if (lowerQuery.includes('범주') || lowerQuery.includes('categorical')) {
      requirements.data_type = 'categorical';
    }

    // 전처리 요구사항
    if (lowerQuery.includes('정규화') || lowerQuery.includes('normalize')) {
      requirements.preprocessing.push('normalization');
    }
    if (lowerQuery.includes('스케일') || lowerQuery.includes('scale')) {
      requirements.preprocessing.push('scaling');
    }
    if (lowerQuery.includes('이상치') || lowerQuery.includes('outlier')) {
      requirements.preprocessing.push('outlier_detection');
    }
    if (lowerQuery.includes('결측값') || lowerQuery.includes('missing')) {
      requirements.preprocessing.push('missing_value_handling');
    }

    return requirements;
  }

  analyzeContext(query) {
    const context = {
      domain: 'general',
      urgency: 'normal',
      complexity: 'medium'
    };

    const lowerQuery = query.toLowerCase();

    // 도메인 분석
    if (lowerQuery.includes('금융') || lowerQuery.includes('financial')) {
      context.domain = 'finance';
    } else if (lowerQuery.includes('의료') || lowerQuery.includes('medical')) {
      context.domain = 'healthcare';
    } else if (lowerQuery.includes('마케팅') || lowerQuery.includes('marketing')) {
      context.domain = 'marketing';
    } else if (lowerQuery.includes('IoT') || lowerQuery.includes('센서')) {
      context.domain = 'iot';
    }

    // 긴급도 분석
    if (lowerQuery.includes('긴급') || lowerQuery.includes('urgent') || lowerQuery.includes('빨리')) {
      context.urgency = 'high';
    } else if (lowerQuery.includes('천천히') || lowerQuery.includes('나중에')) {
      context.urgency = 'low';
    }

    // 복잡도 분석
    const complexityKeywords = ['복잡', '상세', '정교', 'complex', 'detailed'];
    const simpleKeywords = ['간단', '빠르게', 'simple', 'quick'];
    
    if (complexityKeywords.some(keyword => lowerQuery.includes(keyword))) {
      context.complexity = 'high';
    } else if (simpleKeywords.some(keyword => lowerQuery.includes(keyword))) {
      context.complexity = 'low';
    }

    return context;
  }

  validateQuery(query) {
    const validation = {
      is_valid: true,
      warnings: [],
      errors: [],
      suggestions: []
    };

    // 기본 검증
    if (!query || query.trim().length === 0) {
      validation.is_valid = false;
      validation.errors.push('빈 쿼리입니다.');
      return validation;
    }

    // 길이 검증
    if (query.length > 1000) {
      validation.warnings.push('쿼리가 너무 깁니다. 간단히 요약해주세요.');
    }

    // 파일 참조 검증
    const fileRefs = this.extractFileReferences(query);
    if (fileRefs.explicit.length === 0 && fileRefs.implicit.length === 0) {
      validation.suggestions.push('분석할 데이터 파일을 명시해주세요.');
    }

    // 액션 검증
    const actionKeywords = ['분석', '시각화', '훈련', '예측', '비교'];
    const hasAction = actionKeywords.some(keyword => query.includes(keyword));
    if (!hasAction) {
      validation.suggestions.push('수행할 작업을 명확히 해주세요 (예: 분석, 시각화, 모델 훈련).');
    }

    return validation;
  }

  async resolveReferences(analysis) {
    const resolved = {
      files: [],
      columns: [],
      parameters: {}
    };

    // 파일 참조 해결
    const fileRefs = analysis.file_references;
    const detectedFiles = analysis.detected_files;

    // 명시적 파일 참조 해결
    for (const ref of fileRefs.explicit) {
      const matchedFiles = this.findMatchingFiles(ref.filename, detectedFiles);
      resolved.files.push(...matchedFiles);
    }

    // 암시적 파일 참조 해결
    if (fileRefs.implicit.length > 0 && resolved.files.length === 0) {
      // 가장 최근 파일 또는 가장 적합한 파일 선택
      const candidateFiles = this.selectCandidateFiles(detectedFiles);
      resolved.files.push(...candidateFiles);
    }

    // 컬럼 참조 해결 (파일이 있는 경우)
    if (resolved.files.length > 0) {
      // 실제 파일에서 컬럼 정보 추출 (CSV의 경우)
      for (const file of resolved.files) {
        if (file.type === 'csv') {
          try {
            const columns = await this.extractColumnsFromCSV(file.path);
            resolved.columns.push(...columns);
          } catch (error) {
            this.logger.warn(`컬럼 추출 실패: ${file.path}`, error);
          }
        }
      }
    }

    // 파라미터 해결
    resolved.parameters = this.resolveParameters(analysis.parameters, analysis.numeric_values);

    return resolved;
  }

  findMatchingFiles(filename, detectedFiles) {
    const matches = [];
    
    for (const [location, files] of Object.entries(detectedFiles)) {
      for (const file of files) {
        if (file.name === filename || file.name.includes(filename)) {
          matches.push({
            ...file,
            location: location,
            match_type: file.name === filename ? 'exact' : 'partial'
          });
        }
      }
    }
    
    return matches;
  }

  selectCandidateFiles(detectedFiles) {
    const candidates = [];
    
    // 모든 파일 수집
    for (const files of Object.values(detectedFiles)) {
      candidates.push(...files);
    }
    
    // CSV 파일 우선, 그 다음 크기 순으로 정렬
    candidates.sort((a, b) => {
      if (a.type === 'csv' && b.type !== 'csv') return -1;
      if (a.type !== 'csv' && b.type === 'csv') return 1;
      return 0; // 크기 정보가 있다면 크기 순으로 정렬
    });
    
    return candidates.slice(0, 3); // 상위 3개만 반환
  }

  async extractColumnsFromCSV(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n');
      
      if (lines.length === 0) return [];
      
      const headers = lines[0].split(',').map(header => header.trim().replace(/"/g, ''));
      return headers;
    } catch (error) {
      this.logger.warn(`CSV 컬럼 추출 실패: ${filePath}`, error);
      return [];
    }
  }

  resolveParameters(extractedParams, numericValues) {
    const resolved = { ...extractedParams };
    
    // 숫자 값들을 적절한 파라미터로 매핑
    if (numericValues.integers.length > 0) {
      const firstInt = numericValues.integers[0].value;
      
      // 일반적인 범위에 따라 파라미터 추정
      if (firstInt >= 2 && firstInt <= 20) {
        resolved.model.n_clusters = resolved.model.n_clusters || firstInt;
        resolved.model.n_components = resolved.model.n_components || firstInt;
      } else if (firstInt >= 10 && firstInt <= 1000) {
        resolved.model.epochs = resolved.model.epochs || firstInt;
        resolved.model.batch_size = resolved.model.batch_size || firstInt;
      }
    }
    
    if (numericValues.floats.length > 0) {
      const firstFloat = numericValues.floats[0].value;
      
      if (firstFloat > 0 && firstFloat < 1) {
        resolved.model.learning_rate = resolved.model.learning_rate || firstFloat;
        resolved.model.test_size = resolved.model.test_size || firstFloat;
      }
    }
    
    return resolved;
  }

  suggestCorrections(analysis) {
    const suggestions = [];
    
    // 파일 참조 수정 제안
    if (analysis.file_references.explicit.length === 0 && analysis.detected_files.current_directory.length > 0) {
      suggestions.push({
        type: 'file_reference',
        message: '다음 파일들을 사용할 수 있습니다:',
        options: analysis.detected_files.current_directory.map(f => f.name)
      });
    }
    
    // 컬럼 참조 수정 제안
    if (analysis.column_references.explicit.length === 0 && analysis.resolved_references.columns.length > 0) {
      suggestions.push({
        type: 'column_reference',
        message: '다음 컬럼들을 사용할 수 있습니다:',
        options: analysis.resolved_references.columns
      });
    }
    
    // 파라미터 수정 제안
    if (Object.keys(analysis.parameters.model).length === 0) {
      suggestions.push({
        type: 'parameters',
        message: '분석 파라미터를 지정하면 더 정확한 결과를 얻을 수 있습니다.',
        options: ['n_clusters', 'n_components', 'test_size', 'random_state']
      });
    }
    
    return suggestions;
  }

  determineFileType(filename) {
    const ext = path.extname(filename).toLowerCase();
    
    if (this.filePatterns.csv.test(filename)) return 'csv';
    if (this.filePatterns.excel.test(filename)) return 'excel';
    if (this.filePatterns.json.test(filename)) return 'json';
    if (this.filePatterns.text.test(filename)) return 'text';
    if (this.filePatterns.image.test(filename)) return 'image';
    if (this.filePatterns.parquet.test(filename)) return 'parquet';
    if (this.filePatterns.hdf5.test(filename)) return 'hdf5';
    
    return 'unknown';
  }

  isDataFile(filename) {
    return Object.values(this.filePatterns).some(pattern => pattern.test(filename));
  }

  async detectAvailableFiles() {
    const files = {
      current_directory: [],
      uploads: [],
      data_directory: []
    };

    try {
      // 현재 디렉토리 스캔
      const currentFiles = await fs.readdir('./');
      files.current_directory = currentFiles
        .filter(file => this.isDataFile(file))
        .map(file => ({
          name: file,
          path: `./${file}`,
          type: this.determineFileType(file),
          size: null // 크기는 필요시 별도 조회
        }));

      // uploads 디렉토리 스캔
      try {
        const uploadFiles = await fs.readdir('./uploads');
        files.uploads = uploadFiles
          .filter(file => this.isDataFile(file))
          .map(file => ({
            name: file,
            path: `./uploads/${file}`,
            type: this.determineFileType(file),
            size: null
          }));
      } catch (error) {
        // uploads 디렉토리가 없으면 무시
      }

      // data 디렉토리 스캔
      try {
        const dataFiles = await fs.readdir('./data');
        files.data_directory = dataFiles
          .filter(file => this.isDataFile(file))
          .map(file => ({
            name: file,
            path: `./data/${file}`,
            type: this.determineFileType(file),
            size: null
          }));
      } catch (error) {
        // data 디렉토리가 없으면 무시
      }

    } catch (error) {
      this.logger.warn('파일 감지 실패:', error);
    }

    return files;
  }

  getErrorAnalysis(query, error) {
    return {
      original_query: query,
      timestamp: new Date().toISOString(),
      error: error.message,
      file_references: { explicit: [], implicit: [], paths: [] },
      column_references: { explicit: [], target: [], feature_lists: [] },
      numeric_values: { integers: [], floats: [], percentages: [], ranges: [] },
      parameters: { model: {}, visualization: {}, data: {} },
      data_requirements: { file_type: 'unknown', data_type: 'unknown' },
      detected_files: { current_directory: [], uploads: [], data_directory: [] },
      context: { domain: 'general', urgency: 'normal', complexity: 'medium' },
      validation: { is_valid: false, warnings: [], errors: [error.message], suggestions: [] },
      resolved_references: { files: [], columns: [], parameters: {} },
      suggested_corrections: []
    };
  }
}
