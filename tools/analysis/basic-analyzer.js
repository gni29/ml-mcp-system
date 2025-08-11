// tools/analysis/basic-analyzer.js
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';

export class BasicAnalyzer {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.analysisConfig = null;
    
    this.initializeAnalyzer();
  }

  async initializeAnalyzer() {
    try {
      this.analysisConfig = await this.configLoader.loadConfig('analysis-methods.json');
      this.logger.info('BasicAnalyzer 초기화 완료');
    } catch (error) {
      this.logger.error('BasicAnalyzer 초기화 실패:', error);
      this.analysisConfig = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      basic: {
        descriptive_stats: {
          complexity: 0.2,
          estimated_time_ms: 1000,
          python_script: 'python/analysis/basic/descriptive_stats.py'
        },
        correlation: {
          complexity: 0.3,
          estimated_time_ms: 2000,
          python_script: 'python/analysis/basic/correlation.py'
        },
        distribution: {
          complexity: 0.4,
          estimated_time_ms: 3000,
          python_script: 'python/analysis/basic/distribution.py'
        },
        frequency: {
          complexity: 0.2,
          estimated_time_ms: 1500,
          python_script: 'python/analysis/basic/frequency.py'
        }
      }
    };
  }

  async descriptiveStats(data, options = {}) {
    const {
      includeQuartiles = true,
      includeSkewness = true,
      includeKurtosis = true,
      numericOnly = false,
      percentiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    } = options;

    try {
      this.logger.info('기술통계 분석 시작');

      const config = this.getAnalysisConfig('descriptive_stats');
      const scriptPath = config.python_script || 'python/analysis/basic/descriptive_stats.py';

      // Python 스크립트 실행
      const result = await this.pythonExecutor.executeScript(scriptPath, {
        data: data,
        options: {
          include_quartiles: includeQuartiles,
          include_skewness: includeSkewness,
          include_kurtosis: includeKurtosis,
          numeric_only: numericOnly,
          percentiles: percentiles
        }
      });

      if (result.error) {
        throw new Error(result.message || '기술통계 분석 실패');
      }

      this.logger.info('기술통계 분석 완료');
      return await this.resultFormatter.formatResult(result, {
        type: 'descriptive_stats',
        timestamp: new Date().toISOString(),
        method: 'descriptive_statistics'
      });

    } catch (error) {
      this.logger.error('기술통계 분석 실패:', error);
      throw error;
    }
  }

  async correlation(data, options = {}) {
    const {
      method = 'pearson',
      threshold = 0.5,
      generateHeatmap = true,
      onlySignificant = false,
      pValueThreshold = 0.05
    } = options;

    try {
      this.logger.info('상관관계 분석 시작');

      const config = this.getAnalysisConfig('correlation');
      const scriptPath = config.python_script || 'python/analysis/basic/correlation.py';

      // Python 스크립트 실행
      const result = await this.pythonExecutor.executeScript(scriptPath, {
        data: data,
        options: {
          method: method,
          threshold: threshold,
          generate_heatmap: generateHeatmap,
          only_significant: onlySignificant,
          p_value_threshold: pValueThreshold
        }
      });

      if (result.error) {
        throw new Error(result.message || '상관관계 분석 실패');
      }

      this.logger.info('상관관계 분석 완료');
      return await this.resultFormatter.formatResult(result, {
        type: 'correlation',
        timestamp: new Date().toISOString(),
        method: method
      });

    } catch (error) {
      this.logger.error('상관관계 분석 실패:', error);
      throw error;
    }
  }

  async distribution(data, options = {}) {
    const {
      plotType = 'all',
      bins = 30,
      testNormality = true,
      columns = null,
      generatePlots = true
    } = options;

    try {
      this.logger.info('분포 분석 시작');

      const config = this.getAnalysisConfig('distribution');
      const scriptPath = config.python_script || 'python/analysis/basic/distribution.py';

      // Python 스크립트 실행
      const result = await this.pythonExecutor.executeScript(scriptPath, {
        data: data,
        options: {
          plot_type: plotType,
          bins: bins,
          test_normality: testNormality,
          columns: columns,
          generate_plots: generatePlots
        }
      });

      if (result.error) {
        throw new Error(result.message || '분포 분석 실패');
      }

      this.logger.info('분포 분석 완료');
      return await this.resultFormatter.formatResult(result, {
        type: 'distribution',
        timestamp: new Date().toISOString(),
        plot_type: plotType
      });

    } catch (error) {
      this.logger.error('분포 분석 실패:', error);
      throw error;
    }
  }

  async frequency(data, options = {}) {
    const {
      includePercentages = true,
      sortBy = 'frequency',
      topN = 20,
      columns = null
    } = options;

    try {
      this.logger.info('빈도 분석 시작');

      const config = this.getAnalysisConfig('frequency');
      const scriptPath = config.python_script || 'python/analysis/basic/frequency.py';

      // Python 스크립트 실행
      const result = await this.pythonExecutor.executeScript(scriptPath, {
        data: data,
        options: {
          include_percentages: includePercentages,
          sort_by: sortBy,
          top_n: topN,
          columns: columns
        }
      });

      if (result.error) {
        throw new Error(result.message || '빈도 분석 실패');
      }

      this.logger.info('빈도 분석 완료');
      return await this.resultFormatter.formatResult(result, {
        type: 'frequency',
        timestamp: new Date().toISOString(),
        sort_by: sortBy
      });

    } catch (error) {
      this.logger.error('빈도 분석 실패:', error);
      throw error;
    }
  }

  // 통합 분석 (모든 기본 분석을 한 번에)
  async comprehensiveAnalysis(data, options = {}) {
    const {
      includeDescriptive = true,
      includeCorrelation = true,
      includeDistribution = true,
      includeFrequency = true,
      generateSummary = true
    } = options;

    try {
      this.logger.info('종합 기본 분석 시작');

      const results = {
        analysis_type: 'comprehensive_basic',
        timestamp: new Date().toISOString(),
        results: {},
        execution_info: {
          started_at: new Date().toISOString(),
          analyses_requested: []
        }
      };

      // 요청된 분석 목록 기록
      if (includeDescriptive) results.execution_info.analyses_requested.push('descriptive_stats');
      if (includeCorrelation) results.execution_info.analyses_requested.push('correlation');
      if (includeDistribution) results.execution_info.analyses_requested.push('distribution');
      if (includeFrequency) results.execution_info.analyses_requested.push('frequency');

      // 기술통계
      if (includeDescriptive) {
        try {
          this.logger.info('기술통계 분석 실행 중...');
          results.results.descriptive_stats = await this.descriptiveStats(data, options);
        } catch (error) {
          this.logger.warn('기술통계 분석 실패:', error);
          results.results.descriptive_stats = { 
            error: true, 
            message: error.message,
            analysis_type: 'descriptive_stats'
          };
        }
      }

      // 상관관계
      if (includeCorrelation) {
        try {
          this.logger.info('상관관계 분석 실행 중...');
          results.results.correlation = await this.correlation(data, options);
        } catch (error) {
          this.logger.warn('상관관계 분석 실패:', error);
          results.results.correlation = { 
            error: true, 
            message: error.message,
            analysis_type: 'correlation'
          };
        }
      }

      // 분포 분석
      if (includeDistribution) {
        try {
          this.logger.info('분포 분석 실행 중...');
          results.results.distribution = await this.distribution(data, options);
        } catch (error) {
          this.logger.warn('분포 분석 실패:', error);
          results.results.distribution = { 
            error: true, 
            message: error.message,
            analysis_type: 'distribution'
          };
        }
      }

      // 빈도 분석
      if (includeFrequency) {
        try {
          this.logger.info('빈도 분석 실행 중...');
          results.results.frequency = await this.frequency(data, options);
        } catch (error) {
          this.logger.warn('빈도 분석 실패:', error);
          results.results.frequency = { 
            error: true, 
            message: error.message,
            analysis_type: 'frequency'
          };
        }
      }

      results.execution_info.completed_at = new Date().toISOString();
      results.execution_info.total_duration_ms = 
        new Date(results.execution_info.completed_at) - new Date(results.execution_info.started_at);

      // 종합 요약 생성
      if (generateSummary) {
        results.summary = this.generateComprehensiveSummary(results.results);
      }

      this.logger.info('종합 기본 분석 완료');
      return results;

    } catch (error) {
      this.logger.error('종합 기본 분석 실패:', error);
      throw error;
    }
  }

  generateComprehensiveSummary(results) {
    const summary = {
      analysis_completed: [],
      analysis_failed: [],
      key_findings: [],
      data_quality: 'unknown',
      recommendations: [],
      success_rate: 0
    };

    let successful = 0;
    let total = 0;

    // 완료된 분석과 실패한 분석 분류
    for (const [analysisType, result] of Object.entries(results)) {
      total++;
      if (result && !result.error) {
        summary.analysis_completed.push(analysisType);
        successful++;
      } else {
        summary.analysis_failed.push({
          type: analysisType,
          error: result?.message || '알 수 없는 오류'
        });
      }
    }

    summary.success_rate = total > 0 ? Math.round((successful / total) * 100) : 0;

    // 주요 발견사항 추출 (에러가 없는 결과에서만)
    if (results.descriptive_stats && !results.descriptive_stats.error) {
      const stats = results.descriptive_stats.data || results.descriptive_stats;
      if (stats.summary) {
        summary.key_findings.push(`총 ${stats.summary.total_rows || 'N/A'}행, ${stats.summary.total_columns || 'N/A'}개 컬럼 분석됨`);
        if (stats.summary.numeric_columns !== undefined) {
          summary.key_findings.push(`숫자형 컬럼: ${stats.summary.numeric_columns}개`);
        }
      }
    }

    if (results.correlation && !results.correlation.error) {
      const corr = results.correlation.data || results.correlation;
      if (corr.summary && corr.summary.strong_correlations_count > 0) {
        summary.key_findings.push(`강한 상관관계 ${corr.summary.strong_correlations_count}개 발견됨`);
      }
    }

    if (results.distribution && !results.distribution.error) {
      const dist = results.distribution.data || results.distribution;
      if (dist.summary && dist.summary.normality_summary) {
        const normalPct = dist.summary.normality_summary.percentage_normal;
        summary.key_findings.push(`${normalPct}%의 컬럼이 정규분포를 따름`);
      }
    }

    // 데이터 품질 평가
    if (summary.success_rate >= 75) {
      summary.data_quality = 'good';
    } else if (summary.success_rate >= 50) {
      summary.data_quality = 'fair';
    } else {
      summary.data_quality = 'poor';
    }

    // 권장사항 생성
    if (summary.analysis_failed.length > 0) {
      summary.recommendations.push('일부 분석이 실패했습니다. 데이터 형식과 내용을 확인해주세요.');
    }

    if (results.correlation && results.correlation.data?.strong_correlations?.length > 0) {
      summary.recommendations.push('강한 상관관계가 있는 변수들에 대한 추가 분석을 고려하세요.');
    }

    if (summary.analysis_completed.length < 4) {
      summary.recommendations.push('모든 기본 분석을 완료하여 데이터를 종합적으로 이해하세요.');
    }

    if (summary.success_rate === 100) {
      summary.recommendations.push('모든 기본 분석이 성공적으로 완료되었습니다. 고급 분석을 고려해보세요.');
    }

    return summary;
  }

  // 빠른 데이터 개요 분석 (JavaScript로 즉시 처리)
  async quickOverview(data, options = {}) {
    try {
      this.logger.info('빠른 데이터 개요 분석 시작');

      const overview = {
        basic_info: {},
        quick_stats: {},
        data_quality: {},
        recommendations: [],
        analysis_type: 'quick_overview',
        timestamp: new Date().toISOString()
      };

      // 데이터 타입별 처리
      if (typeof data === 'string') {
        // 파일 경로인 경우
        overview.basic_info = {
          type: 'file_path',
          path: data,
          note: 'Python 스크립트를 통해 상세 분석이 필요합니다.'
        };
        overview.recommendations.push('파일 데이터는 상세 분석 기능을 사용하세요.');
        return overview;
      }

      // JavaScript 객체 데이터 처리
      if (Array.isArray(data)) {
        overview.basic_info = {
          type: 'array',
          length: data.length,
          sample: data.slice(0, 3),
          has_data: data.length > 0
        };

        if (data.length === 0) {
          overview.recommendations.push('데이터가 비어있습니다.');
        }
      } else if (typeof data === 'object' && data !== null) {
        const keys = Object.keys(data);
        overview.basic_info = {
          type: 'object',
          columns: keys,
          column_count: keys.length,
          has_data: keys.length > 0
        };

        // 각 컬럼의 기본 정보
        for (const key of keys.slice(0, 10)) { // 처음 10개 컬럼만 분석
          if (Array.isArray(data[key])) {
            const values = data[key];
            const nonNullValues = values.filter(v => v !== null && v !== undefined);
            
            overview.quick_stats[key] = {
              count: values.length,
              non_null_count: nonNullValues.length,
              type: this.inferDataType(nonNullValues),
              sample: values.slice(0, 3),
              unique_count: new Set(nonNullValues).size,
              null_count: values.length - nonNullValues.length,
              null_percentage: Math.round(((values.length - nonNullValues.length) / values.length) * 100)
            };
          }
        }
      } else {
        overview.basic_info = {
          type: typeof data,
          value: data,
          note: '단일 값입니다.'
        };
      }

      // 데이터 품질 평가
      let qualityScore = 100;
      let qualityIssues = [];

      for (const [col, stats] of Object.entries(overview.quick_stats)) {
        if (stats.null_percentage > 10) {
          qualityScore -= 10;
          qualityIssues.push(`${col} 컬럼에 결측값이 ${stats.null_percentage}% 있습니다.`);
        }
        
        if (stats.unique_count === 1) {
          qualityScore -= 5;
          qualityIssues.push(`${col} 컬럼이 모두 동일한 값입니다.`);
        }
      }

      overview.data_quality = {
        score: Math.max(0, qualityScore),
        level: qualityScore >= 90 ? 'excellent' : 
               qualityScore >= 75 ? 'good' : 
               qualityScore >= 60 ? 'fair' : 'poor',
        issues: qualityIssues
      };

      // 권장사항 추가
      overview.recommendations.push(...qualityIssues);
      
      if (overview.basic_info.column_count > 0) {
        overview.recommendations.push('상세한 분석을 위해 기술통계나 상관관계 분석을 수행하세요.');
      }

      this.logger.info('빠른 데이터 개요 분석 완료');
      return overview;

    } catch (error) {
      this.logger.error('빠른 데이터 개요 분석 실패:', error);
      return {
        error: true,
        message: error.message,
        analysis_type: 'quick_overview',
        timestamp: new Date().toISOString()
      };
    }
  }

  inferDataType(values) {
    if (!values || values.length === 0) return 'unknown';
    
    const sampleSize = Math.min(100, values.length);
    const sample = values.slice(0, sampleSize);
    
    const types = {
      number: 0,
      string: 0,
      boolean: 0,
      date: 0
    };

    for (const value of sample) {
      if (typeof value === 'number' && !isNaN(value)) {
        types.number++;
      } else if (typeof value === 'boolean') {
        types.boolean++;
      } else if (typeof value === 'string') {
        // 날짜 형식 확인
        if (/^\d{4}-\d{2}-\d{2}/.test(value) || !isNaN(Date.parse(value))) {
          types.date++;
        } else {
          types.string++;
        }
      } else {
        types.string++;
      }
    }

    // 가장 많은 타입 반환
    return Object.keys(types).reduce((a, b) => types[a] > types[b] ? a : b);
  }

  // 분석 설정 가져오기
  getAnalysisConfig(methodName) {
    if (this.analysisConfig?.basic?.[methodName]) {
      return this.analysisConfig.basic[methodName];
    }
    return this.getDefaultConfig().basic[methodName] || {};
  }

  // 지원되는 분석 방법 목록
  getSupportedMethods() {
    return [
      {
        name: 'descriptiveStats',
        display_name: '기술통계',
        description: '데이터의 기본적인 통계량 계산 (평균, 표준편차, 분위수 등)',
        complexity: 0.2,
        estimated_time: '1-2초',
        requires_python: true
      },
      {
        name: 'correlation',
        display_name: '상관관계 분석',
        description: '변수 간의 선형 상관관계 분석',
        complexity: 0.3,
        estimated_time: '2-3초',
        requires_python: true
      },
      {
        name: 'distribution',
        display_name: '분포 분석',
        description: '데이터의 분포 특성 및 정규성 검정',
        complexity: 0.4,
        estimated_time: '3-5초',
        requires_python: true
      },
      {
        name: 'frequency',
        display_name: '빈도 분석',
        description: '범주형 데이터의 빈도 및 비율 분석',
        complexity: 0.2,
        estimated_time: '1-2초',
        requires_python: true
      },
      {
        name: 'comprehensiveAnalysis',
        display_name: '종합 분석',
        description: '모든 기본 분석을 한 번에 수행',
        complexity: 0.5,
        estimated_time: '5-10초',
        requires_python: true
      },
      {
        name: 'quickOverview',
        display_name: '빠른 개요',
        description: '데이터의 기본 정보를 빠르게 파악 (JavaScript)',
        complexity: 0.1,
        estimated_time: '1초 미만',
        requires_python: false
      }
    ];
  }

  // 리소스 사용량 추정
  estimateResourceUsage(data, method) {
    let estimatedMemory = 50; // 기본 50MB
    let estimatedTime = 1000; // 기본 1초

    // 데이터 크기에 따른 추정
    if (typeof data === 'object' && data !== null) {
      const dataSize = JSON.stringify(data).length;
      estimatedMemory += Math.floor(dataSize / (1024 * 1024)) * 2; // 2MB per MB of data
      estimatedTime += Math.floor(dataSize / 100000) * 100; // 100ms per 100KB
    }

    // 분석 방법에 따른 추정
    const methodConfig = this.getAnalysisConfig(method);
    if (methodConfig.complexity) {
      estimatedTime *= (1 + methodConfig.complexity);
      estimatedMemory *= (1 + methodConfig.complexity * 0.5);
    }

    return {
      estimated_memory_mb: Math.round(estimatedMemory),
      estimated_time_ms: Math.round(estimatedTime),
      method: method,
      complexity: methodConfig.complexity || 0.3,
      requires_python: method !== 'quickOverview'
    };
  }

  // 분석 상태 확인
  getAnalysisStatus() {
    return {
      initialized: !!this.analysisConfig,
      python_executor_ready: !!this.pythonExecutor,
      supported_methods: this.getSupportedMethods().length,
      last_initialized: new Date().toISOString()
    };
  }
}