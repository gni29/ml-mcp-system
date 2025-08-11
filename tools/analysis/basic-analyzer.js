// tools/analysis/basic-analyzer.js - 기본 통계 분석 인터페이스
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class BasicAnalyzer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.analysisHistory = [];
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('BasicAnalyzer 초기화 완료');
    } catch (error) {
      this.logger.error('BasicAnalyzer 초기화 실패:', error);
      throw error;
    }
  }

  async analyzeData(data, options = {}) {
    try {
      this.logger.info('기본 데이터 분석 시작');

      const {
        analysis_type = 'comprehensive',
        target_columns = null,
        statistical_level = 'basic',
        generate_plots = true,
        include_outliers = true
      } = options;

      // Python 스크립트 호출
      const scriptPath = 'python/analysis/basic/descriptive_stats.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        analysis_type,
        target_columns,
        statistical_level,
        generate_plots,
        include_outliers
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 120000
      });

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        this.recordAnalysisHistory(data, options, analysisResult);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'basic_analysis');
      } else {
        throw new Error(`분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('기본 분석 실패:', error);
      throw error;
    }
  }

  async calculateDescriptiveStats(data, columns = null) {
    try {
      this.logger.info('기술통계 계산 시작');

      const scriptPath = 'python/analysis/basic/descriptive_stats.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        analysis_type: 'descriptive_only',
        target_columns: columns,
        statistical_level: 'advanced',
        generate_plots: false,
        include_outliers: false
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        const statsResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(statsResult, 'descriptive_stats');
      } else {
        throw new Error(`기술통계 계산 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('기술통계 계산 실패:', error);
      throw error;
    }
  }

  async calculateCorrelations(data, method = 'pearson', columns = null) {
    try {
      this.logger.info('상관관계 분석 시작');

      const scriptPath = 'python/analysis/basic/correlation.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        method: method,
        target_columns: columns
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        const correlationResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(correlationResult, 'correlation_analysis');
      } else {
        throw new Error(`상관관계 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('상관관계 분석 실패:', error);
      throw error;
    }
  }

  async analyzeDistribution(data, columns = null) {
    try {
      this.logger.info('분포 분석 시작');

      const scriptPath = 'python/analysis/basic/distribution.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_columns: columns,
        include_plots: true
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 90000
      });

      if (result.success) {
        const distributionResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(distributionResult, 'distribution_analysis');
      } else {
        throw new Error(`분포 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('분포 분석 실패:', error);
      throw error;
    }
  }

  async analyzeFrequency(data, columns = null) {
    try {
      this.logger.info('빈도 분석 시작');

      const scriptPath = 'python/analysis/basic/frequency.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_columns: columns,
        include_plots: true
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        const frequencyResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(frequencyResult, 'frequency_analysis');
      } else {
        throw new Error(`빈도 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('빈도 분석 실패:', error);
      throw error;
    }
  }

  async detectOutliers(data, method = 'iqr', columns = null) {
    try {
      this.logger.info('이상치 탐지 시작');

      const scriptPath = 'python/analysis/advanced/outlier_detection.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        method: method,
        target_columns: columns,
        include_plots: true
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 90000
      });

      if (result.success) {
        const outlierResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(outlierResult, 'outlier_analysis');
      } else {
        throw new Error(`이상치 탐지 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('이상치 탐지 실패:', error);
      throw error;
    }
  }

  async checkDataQuality(data) {
    try {
      this.logger.info('데이터 품질 검사 시작');

      const scriptPath = 'python/data/validate_data.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        check_type: 'quality_assessment'
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        const qualityResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(qualityResult, 'data_quality');
      } else {
        throw new Error(`데이터 품질 검사 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('데이터 품질 검사 실패:', error);
      throw error;
    }
  }

  recordAnalysisHistory(data, options, result) {
    const record = {
      timestamp: new Date().toISOString(),
      data_type: typeof data,
      data_info: typeof data === 'string' ? data : `${Object.keys(data).length} keys`,
      options,
      success: !result.error,
      execution_time: Date.now()
    };

    this.analysisHistory.push(record);
    
    // 히스토리 크기 제한 (최대 50개)
    if (this.analysisHistory.length > 50) {
      this.analysisHistory = this.analysisHistory.slice(-25);
    }
  }

  // 유틸리티 메서드들
  getAnalysisHistory(limit = 10) {
    return this.analysisHistory.slice(-limit);
  }

  getSupportedAnalysisTypes() {
    return [
      'comprehensive',
      'descriptive_only', 
      'correlation_only',
      'distribution_only',
      'frequency_only',
      'outlier_only',
      'quality_check'
    ];
  }

  getSupportedStatisticalLevels() {
    return ['basic', 'advanced', 'comprehensive'];
  }

  getSupportedCorrelationMethods() {
    return ['pearson', 'spearman', 'kendall'];
  }

  getSupportedOutlierMethods() {
    return ['iqr', 'zscore', 'isolation_forest', 'local_outlier_factor'];
  }

  async getAnalysisStatus() {
    return {
      python_executor_status: await this.pythonExecutor.getExecutionStats(),
      analysis_history_count: this.analysisHistory.length,
      supported_features: {
        analysis_types: this.getSupportedAnalysisTypes(),
        statistical_levels: this.getSupportedStatisticalLevels(),
        correlation_methods: this.getSupportedCorrelationMethods(),
        outlier_methods: this.getSupportedOutlierMethods()
      }
    };
  }

  async cleanup() {
    await this.pythonExecutor.shutdown();
    this.logger.info('BasicAnalyzer 정리 완료');
  }
}