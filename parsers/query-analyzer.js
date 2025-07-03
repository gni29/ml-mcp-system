// parsers/query-analyzer.js
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class QueryAnalyzer {
  constructor() {
    this.logger = new Logger();
    this.filePatterns = this.initializeFilePatterns();
    this.dataTypePatterns = this.initializeDataTypePatterns();
    this.parameterPatterns = this.initializeParameterPatterns();
  }

  initializeFilePatterns() {
    return {
      csv: /\.csv$/i,
      excel: /\.(xlsx?|xls)$/i,
      json: /\.json$/i,
      text: /\.txt$/i,
      image: /\.(png|jpg|jpeg|gif|bmp|tiff)$/i,
      parquet: /\.parquet$/i,
      hdf5: /\.h5$/i
    };
  }

  initializeDataTypePatterns() {
    return {
      file_references: {
        explicit: /([a-zA-Z0-9_-]+\.(csv|xlsx?|json|txt|png|jpg|jpeg))/gi,
        implicit: /(이 파일|현재 파일|업로드된 파일|데이터 파일)/gi,
        path_like: /([\.\/]?[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+)/gi
      },
      column_references: {
        explicit: /([a-zA-Z0-9_]+)\s*(컬럼|열|변수|피처|feature|column)/gi,
        target: /(타겟|target|목표|예측할|분류할)\s*([a-zA-Z0-9_]+)/gi,
        feature_list: /\[([a-zA-Z0-9_,\s]+)\]/gi
      },
      numeric_values: {
        integers: /\b(\d+)\b/g,
        floats: /\b(\d+\.\d+)\b/g,
        percentages: /(\d+(?:\.\d+)?)\s*%/g,
        ranges: /(\d+)\s*[-~]\s*(\d+)/g
      },
      time_references: {
        dates: /\b(\d{4}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4})\b/g,
        periods: /(지난|최근|이번|다음)\s*(\d+)\s*(일|주|월|년|day|week|month|year)/gi,
        time_units: /(시간|분|초|hour|minute|second)/gi
      }
    };
  }

  initializeParameterPatterns() {
    return {
      model_parameters: {
        epochs: /epoch[s]?\s*[:=]?\s*(\d+)/gi,
        batch_size: /(batch[_\s]?size|배치[_\s]?사이즈)\s*[:=]?\s*(\d+)/gi,
        learning_rate: /(learning[_\s]?rate|학습률)\s*[:=]?\s*(\d+\.?\d*)/gi,
        n_clusters: /(n[_\s]?clusters?|클러스터\s*수)\s*[:=]?\s*(\d+)/gi,
        n_components: /(n[_\s]?components?|컴포넌트\s*수|주성분\s*수)\s*[:=]?\s*(\d+)/gi,
        test_size: /(test[_\s]?size|테스트\s*비율)\s*[:=]?\s*(\d+\.?\d*)/gi,
        random_state: /(random[_\s]?state|시드)\s*[:=]?\s*(\d+)/gi
      },
      visualization_parameters: {
        figsize: /(figsize|크기)\s*[:=]?\s*\[?(\d+),?\s*(\d+)\]?/gi,
        color: /(color|색상|컬러)\s*[:=]?\s*([a-zA-Z0-9#]+)/gi,
        alpha: /(alpha|투명도)\s*[:=]?\s*(\d+\.?\d*)/gi,
        title: /(title|제목)\s*[:=]?\s*['"]([^'"]+)['"]/gi
      },
      data_parameters: {
        separator: /(sep|separator|구분자)\s*[:=]?\s*['"]([^'"]+)['"]/gi,
        encoding: /(encoding|인코딩)\s*[:=]?\s*['"]([^'"]+)['"]/gi,
        header: /(header|헤더)\s*[:=]?\s*(\d+|None|True|False)/gi,
        index_col: /(index[_\s]?col|인덱스\s*컬럼)\s*[:=]?\s*(\d+|None)/gi
      }
    };
  }

  async analyzeQuery(query) {
    try {
      const analysis = {
        original_query: query,
        timestamp: new Date().toISOString(),
        
        // 기본 분석
        file_references: this.extractFileReferences(query),
        column_references: this.extractColumnReferences(query),
        numeric_values: this.extractNumericValues(query),
        time_references: this.extractTimeReferences(query),
        
        // 파라미터 분석
        parameters: this.extractParameters(query),
        
        // 데이터 요구사항 분석
        data_requirements: await this.analyzeDataRequirements(query),
        
        // 자동 파일 감지
        detected_files: await this.detectAvailableFiles(),
        
        // 컨텍스트 분석
        context: this.analyzeContext(query),
        
        // 검증
        validation: this.validateQuery(query)
      };

      // 분석 결과 후처리
      analysis.resolved_references = await this.resolveReferences(analysis);
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
    const explicitMatches = query.match(this.dataTypePatterns.file_references.explicit);
    if (explicitMatches) {
      references.explicit = explicitMatches.map(match => ({
        filename: match,
        extension: path.extname(match).toLowerCase(),
        type: this.determineFileType(match)
      }));
    }

    // 암시적 파일 참조
    const implicitMatches = query.match(this.dataTypePatterns.file_references.implicit);
    if (implicitMatches) {
      references.implicit = implicitMatches.map(match => ({
        phrase: match,
        type: 'implicit'
      }));
    }

    // 경로 형태 참조
    const pathMatches = query.match(this.dataTypePatterns.file_references.path_like);
    if (pathMatches) {
      references.paths = pathMatches.map(match => ({
        path: match,
        type: 'path'
      }));
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
    const explicitMatches = [...query.matchAll(this.dataTypePatterns.column_references.explicit)];
    references.explicit = explicitMatches.map(match => ({
      column: match[1],
      context: match[0]
    }));

    // 타겟 변수 참조
    const targetMatches = [...query.matchAll(this.dataTypePatterns.column_references.target)];
    references.target = targetMatches.map(match => ({
      column: match[2],
      context: match[0]
    }));

    // 피처 리스트
    const featureMatches = [...query.matchAll(this.dataTypePatterns.column_references.feature_list)];
    references.feature_lists = featureMatches.map(match => ({
      features: match[1].split(',').map(f => f.trim()),
      context: match[0]
    }));

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
    const intMatches = [...query.matchAll(this.dataTypePatterns.numeric_values.integers)];
    values.integers = intMatches.map(match => parseInt(match[1]));

    // 실수
    const floatMatches = [...query.matchAll(this.dataTypePatterns.numeric_values.floats)];
    values.floats = floatMatches.map(match => parseFloat(match[1]));

    // 백분율
    const percentMatches = [...query.matchAll(this.dataTypePatterns.numeric_values.percentages)];
    values.percentages = percentMatches.map(match => parseFloat(match[1]));

    // 범위
    const rangeMatches = [...query.matchAll(this.dataTypePatterns.numeric_values.ranges)];
    values.ranges = rangeMatches.map(match => ({
      min: parseInt(match[1]),
      max: parseInt(match[2])
    }));

    return values;
  }

  extractTimeReferences(query) {
    const references = {
      dates: [],
      periods: [],
      time_units: []
    };

    // 날짜
    const dateMatches = [...query.matchAll(this.dataTypePatterns.time_references.dates)];
    references.dates = dateMatches.map(match => ({
      date: match[1],
      parsed: new Date(match[1])
    }));

    // 기간
    const periodMatches = [...query.matchAll(this.dataTypePatterns.time_references.periods)];
    references.periods = periodMatches.map(match => ({
      direction: match[1],
      amount: parseInt(match[2]),
      unit: match[3],
      context: match[0]
    }));

    // 시간 단위
    const timeMatches = [...query.matchAll(this.dataTypePatterns.time_references.time_units)];
    references.time_units = timeMatches.map(match => match[1]);

    return references;
  }

  extractParameters(query) {
    const parameters = {
      model: {},
      visualization: {},
      data: {}
    };

    // 모델 파라미터
    for (const [param, pattern] of Object.entries(this.parameterPatterns.model_parameters)) {
      const matches = [...query.matchAll(pattern)];
      if (matches.length > 0) {
        const value = matches[0][matches[0].length - 1];
        parameters.model[param] = isNaN(value) ? value : parseFloat(value);
      }
    }

    // 시각화 파라미터
    for (const [param, pattern] of Object.entries(this.parameterPatterns.visualization_parameters)) {
      const matches = [...query.matchAll(pattern)];
      if (matches.length > 0) {
        if (param === 'figsize') {
          parameters.visualization[param] = [parseInt(matches[0][2]), parseInt(matches[0][3])];
        } else {
          const value = matches[0][matches[0].length - 1];
          parameters.visualization[param] = isNaN(value) ? value : parseFloat(value);
        }
      }
    }

    // 데이터 파라미터
    for (const [param, pattern] of Object.entries(this.parameterPatterns.data_parameters)) {
      const matches = [...query.matchAll(pattern)];
      if (matches.length > 0) {
        const value = matches[0][matches[0].length - 1];
        parameters.data[param] = value;
      }
    }

    return parameters;
  }

  async analyzeDataRequirements(query) {
    const requirements = {
      file_type: 'unknown',
      data_type: 'tabular',
      size_estimate: 'medium',
      columns_needed: [],
      preprocessing: [],
      validation: []
    };

    // 파일 타입 결정
    const fileRefs = this.extractFileReferences(query);
    if (fileRefs.explicit.length > 0) {
      requirements.file_type = fileRefs.explicit[0].type;
    }

    // 데이터 타입 추정
    if (query.includes('이미지') || query.includes('image') || query.includes('jpg') || query.includes('png')) {
      requirements.data_type = 'image';
    } else if (query.includes('텍스트') || query.includes('text') || query.includes('nlp')) {
      requirements.data_type = 'text';
    } else if (query.includes('시계열') || query.includes('time') || query.includes('날짜')) {
      requirements.data_type = 'time_series';
    }

    // 필요한 컬럼 추출
    const columnRefs = this.extractColumnReferences(query);
    requirements.columns_needed = [
      ...columnRefs.explicit.map(ref => ref.column),
      ...columnRefs.target.map(ref => ref.column),
      ...columnRefs.feature_lists.flatMap(ref => ref.features)
    ];

    // 전처리 요구사항 분석
    if (query.includes('정규화') || query.includes('normalize')) {
      requirements.preprocessing.push('normalization');
    }
    if (query.includes('스케일링') || query.includes('scale')) {
      requirements.preprocessing.push('scaling');
    }
    if (query.includes('이상치') || query.includes('outlier')) {
      requirements.preprocessing.push('outlier_detection');
    }
    if (query.includes('결측값') || query.includes('missing')) {
      requirements.preprocessing.push('missing_value_handling');
    }

    return requirements;
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

  analyzeContext(query) {
    const context = {
      domain: 'general',
      urgency: 'normal',
      complexity: 'medium',
      user_expertise: 'intermediate',
      output_preference: 'comprehensive'
    };

    // 도메인 분석
    const domainKeywords = {
      finance: ['금융', '주식', '투자', '수익', '리스크'],
      healthcare: ['의료', '건강', '환자', '진료', '치료'],
      marketing: ['마케팅', '고객', '판매', '광고', '캠페인'],
      manufacturing: ['제조', '생산', '품질', '공정', '설비'],
      education: ['교육', '학생', '성적', '학습', '평가']
    };

    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      if (keywords.some(keyword => query.includes(keyword))) {
        context.domain = domain;
        break;
      }
    }

    // 긴급도 분석
    const urgencyKeywords = {
      high: ['급해', '긴급', '빨리', '즉시', 'urgent'],
      low: ['천천히', '나중에', '여유있게', 'when convenient']
    };

    for (const [level, keywords] of Object.entries(urgencyKeywords)) {
      if (keywords.some(keyword => query.includes(keyword))) {
        context.urgency = level;
        break;
      }
    }

    // 복잡도 분석
    const complexityIndicators = {
      simple: ['간단히', '빠르게', '대충', 'quick', 'simple'],
      complex: ['자세히', '정확히', '완전히', 'detailed', 'comprehensive']
    };

    for (const [level, keywords] of Object.entries(complexityIndicators)) {
      if (keywords.some(keyword => query.includes(keyword))) {
        context.complexity = level;
        break;
      }
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

    // 길이 검증
    if (query.length < 5) {
      validation.errors.push('쿼리가 너무 짧습니다.');
      validation.is_valid = false;
    } else if (query.length > 1000) {
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
      const firstInt = numericValues.integers[0];
      
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
      const firstFloat = numericValues.floats[0];
      
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
